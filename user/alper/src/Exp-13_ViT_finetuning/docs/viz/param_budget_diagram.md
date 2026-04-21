# Trainable Parameter Budget — ViT-L (~304M total)

Bar width proportional to log10(trainable params).
Scale: each '#' ~ 0.5 decades of log10(params).

```
                     log10(params)
                  4    5    6    7    8    9
                  |    |    |    |    |    |
  Full FT         ############################  304.0M  (100.0%)
  Partial 12+LN+H ##########################    160.0M  ( 52.6%)
  Phase1 head     ######################          8.0M  (  2.6%)
  DoRA r32        #######################        15.0M  (  4.9%)
  LoRA r16 attn   ###################             2.0M  (  0.7%)

                  |    |    |    |    |    |
                 10k 100k   1M  10M 100M   1B
```

Legend
------
  Full FT         : unfreeze every weight in backbone + head
  Partial 12+LN+H : last 12 transformer blocks + all LayerNorms
                    + classifier head
  Phase1 head     : classifier only (7806 x 1024 = 7.99M)
  DoRA r32        : LoRA+magnitude on attn.qkv, attn.proj,
                    mlp.fc1, mlp.fc2 (rank 32) + head
  LoRA r16 attn   : LoRA on attn.qkv and attn.proj only
                    (rank 16) + head

Notes
-----
- Bars are log-scaled so small-param methods stay visible; the
  linear ratio Full:LoRA is ~150:1, not the ~1.5:1 shown.
- Phase 1 head alone (~8M) already exceeds LoRA r16 (~2M);
  a LoRA run is mostly re-training the head.
- DoRA r32 on attn+MLP is the sweet spot if we want a small
  trainable footprint that still touches the MLP feature mixer.
