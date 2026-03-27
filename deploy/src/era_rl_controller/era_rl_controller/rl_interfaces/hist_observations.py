import numpy as np


class RingHistory:
    def __init__(self, capacity: int, dim: int):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.buf = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self.head = 0
        self.size = 0

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(
            self.dim,
        )
        tail = (self.head + self.size) % self.capacity
        self.buf[tail] = x
        if self.size < self.capacity:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.capacity

    def get_matrix(self) -> np.ndarray:
        if self.size < self.capacity:
            out = np.zeros((self.capacity, self.dim), dtype=np.float32)
            if self.size > 0:
                out[-self.size :] = self.buf[: self.size]
            return out
        return np.concatenate([self.buf[self.head :], self.buf[: self.head]], axis=0)

    def flatten(self) -> np.ndarray:
        return self.get_matrix().reshape(-1)


class ObsHistory:
    def __init__(self, history_cfg):
        self.dims = {name: cfg["dim"] for name, cfg in history_cfg.items()}
        self.rings = {name: RingHistory(cfg["history_length"], cfg["dim"]) for name, cfg in history_cfg.items()}
        self.order = list(history_cfg.keys())

    @property
    def obs_dim(self) -> int:
        return int(sum(r.capacity * r.dim for r in self.rings.values()))

    def push_item(self, name: str, vec: np.ndarray):
        self.rings[name].push(vec)

    def build_obs(self) -> np.ndarray:
        parts = [self.rings[name].flatten() for name in self.order]
        return np.concatenate(parts, axis=0).astype(np.float32)
