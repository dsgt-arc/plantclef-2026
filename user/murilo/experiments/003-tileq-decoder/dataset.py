"""PyTorch ``IterableDataset`` wrapping ``SamplerA`` / ``SamplerB`` (spec §2.3).

Yields ``(tile_cls (n_tiles, 768) torch.float32, label (n_classes,) torch.float32)``
indefinitely. The DataLoader's collate stacks these into ``(B, n_tiles, 768)``
+ ``(B, n_classes)`` batches.

Multi-worker note: each worker process gets a forked copy of the sampler (and
its ~4.3 GB embeddings array, which on Linux is shared via copy-on-write).
``__iter__`` reseeds the in-process sampler's RNG with a worker-id-derived seed
so workers don't emit duplicate samples.

For the spec's batch sizes (32 @ RTX 6000 / 64 @ H100) and CPU sampling rate
(~2k samples/sec from the smoke tests), a single worker is sufficient
(``num_workers=0``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data


class PseudoQuadratDataset(torch.utils.data.IterableDataset):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Independent per-worker RNG. Derive from the parent RNG so a global
            # `--seed` still controls reproducibility across workers.
            base = int(self.sampler.rng.integers(0, 2**31))
            self.sampler.rng = np.random.default_rng(base + worker_info.id)

        while True:
            tiles, label = self.sampler.sample()
            yield torch.from_numpy(tiles), torch.from_numpy(label)
