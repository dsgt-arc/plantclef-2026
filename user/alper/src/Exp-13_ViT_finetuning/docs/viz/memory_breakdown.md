# GPU Memory Breakdown — H200 (141 GB), bs=24, 518x518

Side-by-side stacked bars. 1 block = ~2 GB. Scale 0-80 GB.

```
  GB:       0     10    20    30    40    50    60    70    80
            |-----|-----|-----|-----|-----|-----|-----|-----|

  Full FT   AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGOOWRRRR           ~69 GB
            activations=60  G=1.2  O=2.4  W=0.6  R=5.0

  LoRA r16  AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwRRRR              ~65.6 GB
            activations=60  G=0.01 O=0.02 W=0.608 R=5.0

            |-----|-----|-----|-----|-----|-----|-----|-----|
  GB:       0     10    20    30    40    50    60    70    80

  Legend
  ------
  A  activations (forward cache for backward) ~60 GB
  G  parameter gradients
  O  optimizer state (AdamW m,v)
  W  model weights (fp16/bf16 ViT-L ~0.6 GB)
  w  model weights + LoRA adapters (~0.608 GB)
  R  other: CUDA ctx, cudnn, workspaces, allgather ~5 GB
```

Absolute deltas (Full FT -> LoRA r16)
-------------------------------------
  grads            1.20 GB  ->  0.01 GB    save 1.19 GB
  optimizer state  2.40 GB  ->  0.02 GB    save 2.38 GB
  weights          0.60 GB  ->  0.61 GB    +0.01 GB (adapters)
  activations     60.00 GB  -> 60.00 GB    unchanged
  other            5.00 GB  ->  5.00 GB    unchanged
  -----------------------------------------------------------
  TOTAL           69.20 GB  -> 65.62 GB    save 3.58 GB (~5%)

Caption
-------
At 518x518 input with batch 24, activations dominate the
memory budget (~87% of the ~69 GB Full-FT footprint).
LoRA only shrinks gradients + optimizer state, which
together are <6% of that footprint, so switching to LoRA
r16 saves only ~3.6 GB out of 141 GB available. The H200's
141 GB budget is not the bottleneck that motivates LoRA --
the real lever for memory is activation checkpointing or
smaller input resolution, not trainable-param count.
