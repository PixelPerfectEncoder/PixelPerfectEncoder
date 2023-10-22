import numpy as np
import bisect

class ResidualProcessor:
    def __init__(self, max_val=255):
        self.max_val = max_val
        self.max_exp = int(np.log2(self.max_val)) + 1
        self.two_to_n = np.array([2 ** i for i in range(0, self.max_exp + 1)], dtype=np.int16)
        biggest_n = 2 ** self.max_exp
        self.n_to_two = np.empty(biggest_n + 1, dtype=np.uint16)
        for v in range(0, biggest_n + 1):
            pos = bisect.bisect_left(self.two_to_n, v)
            if pos != 0 and v - self.two_to_n[pos - 1] <= self.two_to_n[pos] - v:
                self.n_to_two[v] = pos - 1
            else:
                self.n_to_two[v] = pos
                
    def decode(self, residual : np.ndarray) -> np.ndarray:
        decoded = self.two_to_n[np.abs(residual)] * np.sign(residual)
        return np.clip(decoded, -self.max_val, self.max_val)

    def encode(self, residual : np.ndarray) -> np.ndarray:
       return self.n_to_two[np.abs(residual)] * np.sign(residual)
        
