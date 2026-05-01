# Phase 2 Finetuning Strategy — Decision Matrix

ViT-Large (~304M params), PlantCLEF 2024, 7806 classes, 1.4M imgs.
Baseline = Phase 1 (head-only, head ~8M params).

| Strategy                    | F1 delta vs base | Trainable % | Mem save | Wallclock save | Ensemble diversity | PlantCLEF precedent | Impl. effort |
|-----------------------------|------------------|-------------|----------|----------------|--------------------|---------------------|--------------|
| (a) Full FT + LLRD 0.75     | High (+0.03-0.06) strongest signal | 100% (304M) | 0% baseline | 0% baseline | Low same backbone | High Top-3 teams used it | Low already scripted |
| (b) Vanilla LoRA r16 (attn) | Med (+0.01-0.03) undertrains MLP | ~0.7% (2M) | ~3% activ. dominate | ~10-15% smaller bwd | High orthogonal deltas | Low rarely reported | Med peft wiring |
| (c) DoRA r32 (attn+MLP)     | Med-High (+0.02-0.04) closer to full | ~5% (15M) | ~3% activ. dominate | ~5-10% | High distinct from full | Low emerging method | Med-High peft+DoRA layer |
| (d) Partial unfreeze (last 12 + LN + head) | Med-High (+0.02-0.05) | ~52% (160M) | ~1-2% | ~15-25% fewer grads | Low-Med same backbone | Med common baseline | Low freeze loop |

**Recommendation:** stick with (a) Full FT + LLRD 0.75 as primary; run (c) DoRA r32 as a cheap second finetune for ensemble diversity if compute allows — skip (b) and (d).
