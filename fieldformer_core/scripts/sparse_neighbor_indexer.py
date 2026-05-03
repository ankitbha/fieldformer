from __future__ import annotations

import torch


class SplitAwareSparseNeighborIndexer:
    """
    Sparse sensor-time neighbor indexer with an optional visible-index set.

    If allowed_indices is supplied, observed and continuous queries can only
    gather neighbors whose linear indices are in that set. This keeps sparse
    FieldFormer from reading validation/test observation values as context.
    """

    def __init__(
        self,
        sensors_xy: torch.Tensor,
        t_grid: torch.Tensor,
        time_radius: int,
        k_neighbors: int,
        allowed_indices: torch.Tensor | None = None,
    ):
        self.sensors_xy = sensors_xy
        self.t_grid = t_grid
        self.S = sensors_xy.shape[0]
        self.Nt = t_grid.shape[0]
        self.time_radius = int(time_radius)
        self.k_neighbors = int(k_neighbors)
        self.allowed_indices: torch.Tensor | None = None
        self.allowed_mask: torch.Tensor | None = None
        self.fallback_index: torch.Tensor | None = None

        sensor_ids = torch.arange(self.S, dtype=torch.long)
        offsets = torch.arange(-self.time_radius, self.time_radius + 1, dtype=torch.long)
        s_mesh, dt_mesh = torch.meshgrid(sensor_ids, offsets, indexing="ij")
        self.base_sensor = s_mesh.reshape(-1)
        self.base_dt = dt_mesh.reshape(-1)

        if allowed_indices is not None:
            self.set_allowed_indices(allowed_indices)

    def set_allowed_indices(self, allowed_indices: torch.Tensor | None) -> None:
        if allowed_indices is None:
            self.allowed_indices = None
            self.allowed_mask = None
            self.fallback_index = None
            return
        allowed = allowed_indices.detach().long().flatten()
        if allowed.numel() == 0:
            raise ValueError("allowed_indices must contain at least one observation")
        mask = torch.zeros(self.S * self.Nt, dtype=torch.bool, device=allowed.device)
        mask[allowed] = True
        self.allowed_indices = allowed
        self.allowed_mask = mask
        self.fallback_index = allowed[0]

    def lin_to_sk(self, lin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = lin // self.Nt
        k = lin % self.Nt
        return s, k

    def sk_to_lin(self, s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return s * self.Nt + k

    def _filter_and_pad(self, lin_nb: torch.Tensor, lin_q: torch.Tensor | None = None, exclude_self: bool = False) -> torch.Tensor:
        valid = torch.ones_like(lin_nb, dtype=torch.bool)
        if self.allowed_mask is not None:
            valid &= self.allowed_mask.to(lin_nb.device)[lin_nb]
        if exclude_self and lin_q is not None:
            valid &= lin_nb != lin_q[:, None]

        cand_count = lin_nb.shape[1]
        order = torch.arange(cand_count, device=lin_nb.device).unsqueeze(0).expand_as(lin_nb)
        sort_key = torch.where(valid, order, order.new_full(order.shape, cand_count + 1))
        take = torch.argsort(sort_key, dim=1)[:, : min(self.k_neighbors, cand_count)]
        lin_nb = torch.gather(lin_nb, 1, take)
        valid = torch.gather(valid, 1, take)

        has_valid = valid.any(dim=1, keepdim=True)
        first_valid_pos = valid.long().argmax(dim=1, keepdim=True)
        row_fallback = torch.gather(lin_nb, 1, first_valid_pos)
        if self.allowed_indices is not None:
            allowed = self.allowed_indices.to(lin_nb.device)
            primary = allowed[0].expand_as(row_fallback)
            if lin_q is not None and allowed.numel() > 1:
                secondary = allowed[1].expand_as(row_fallback)
                global_fallback = torch.where(primary == lin_q[:, None], secondary, primary)
            else:
                global_fallback = primary
        elif lin_q is not None:
            global_fallback = lin_q[:, None]
        else:
            global_fallback = lin_nb[:, :1]
        row_fallback = torch.where(has_valid, row_fallback, global_fallback)
        lin_nb = torch.where(valid, lin_nb, row_fallback.expand_as(lin_nb))

        if lin_nb.shape[1] < self.k_neighbors:
            pad = lin_nb[:, -1:].expand(-1, self.k_neighbors - lin_nb.shape[1])
            lin_nb = torch.cat([lin_nb, pad], dim=1)
        return lin_nb

    def gather_observed_neighbors(self, lin_q: torch.Tensor, exclude_self: bool = True) -> torch.Tensor:
        _, k_q = self.lin_to_sk(lin_q)
        bsz = lin_q.shape[0]
        s_nb = self.base_sensor.to(lin_q.device).unsqueeze(0).expand(bsz, -1)
        k_nb = (k_q[:, None] + self.base_dt.to(lin_q.device)[None, :]).clamp_(0, self.Nt - 1)
        lin_nb = self.sk_to_lin(s_nb, k_nb)
        return self._filter_and_pad(lin_nb, lin_q=lin_q, exclude_self=exclude_self)

    def gather_continuous_neighbors(self, xyt_q: torch.Tensor) -> torch.Tensor:
        t_q = xyt_q[:, 2]
        dist = torch.abs(t_q[:, None] - self.t_grid[None, :].to(xyt_q.device))
        k_hat = torch.argmin(dist, dim=1)

        bsz = xyt_q.shape[0]
        s_nb = self.base_sensor.to(xyt_q.device).unsqueeze(0).expand(bsz, -1)
        k_nb = (k_hat[:, None] + self.base_dt.to(xyt_q.device)[None, :]).clamp_(0, self.Nt - 1)
        lin_nb = self.sk_to_lin(s_nb, k_nb)
        return self._filter_and_pad(lin_nb)
