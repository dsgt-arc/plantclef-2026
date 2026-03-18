# SAM3 Analysis Results

**Date:** 2026-03-11
**Data:** `~/scratch/plantclef-2026/sam3_outputs/{plant,flower}/task_XXXX/results.parquet`
**Coverage:** ~1.3M images, 20 tasks each, prompts: `plant` and `flower`

---

## 1. Summary Statistics

### Flower prompt

```
task               images      w/det     no-det     det%  masks/det    size MB
------------------------------------------------------------------------------
task_0000         101,736     48,884     52,852    48.0%      12.13      107.1
task_0001          69,451     47,498     21,953    68.4%      12.47      117.5
task_0002          74,408     47,015     27,393    63.2%      14.08      116.3
task_0003          84,739     50,969     33,770    60.1%      11.74      116.8
task_0004          76,354     39,089     37,265    51.2%      11.53       93.7
task_0005          83,980     49,910     34,070    59.4%      11.98      124.5
task_0006         111,049     55,948     55,101    50.4%      13.22      119.6
task_0007          62,450     33,467     28,983    53.6%      12.29       85.6
task_0008          74,208     43,920     30,288    59.2%      12.35      116.7
task_0009          68,834     44,128     24,706    64.1%      12.37      113.1
task_0010          72,538     45,196     27,342    62.3%      12.06      112.8
task_0011          72,042     40,238     31,804    55.9%      13.19      102.9
task_0012          72,787     44,622     28,165    61.3%      12.60      112.8
task_0013          63,141     39,026     24,115    61.8%      12.99      103.7
task_0014          65,228     34,439     30,789    52.8%      12.14       84.2
task_0015          22,013     15,307      6,706    69.5%      11.01       37.4
task_0016          22,767     14,290      8,477    62.8%      10.58       37.4
task_0017          53,244     27,300     25,944    51.3%      11.40       63.6
task_0018          41,825     27,167     14,658    65.0%      12.43       73.5
task_0019          15,838     10,139      5,699    64.0%      11.33       26.5
------------------------------------------------------------------------------
TOTAL           1,308,632    758,552    550,080    58.0%      12.36     1865.5
```

### Plant prompt

```
task               images      w/det     no-det     det%  masks/det    size MB
------------------------------------------------------------------------------
task_0000         101,736     83,429     18,307    82.0%       2.73      603.6
task_0001          69,451     50,481     18,970    72.7%       3.16      350.8
task_0002          74,408     56,411     17,997    75.8%       2.67      393.6
task_0003          84,739     61,964     22,775    73.1%       2.67      438.0
task_0004          76,354     55,888     20,466    73.2%       2.88      379.4
task_0005          83,980     60,551     23,429    72.1%       2.80      410.9
task_0006         111,049     84,026     27,023    75.7%       2.61      597.1
task_0007          62,717     46,912     15,805    74.8%       2.94      331.6
task_0008          74,208     53,968     20,240    72.7%       2.79      368.4
task_0009          68,834     49,269     19,565    71.6%       2.98      343.5
task_0010          72,538     52,783     19,755    72.8%       3.06      358.5
task_0011          72,042     54,079     17,963    75.1%       3.03      379.2
task_0012          72,787     52,539     20,248    72.2%       2.81      365.6
task_0013          63,141     46,598     16,543    73.8%       3.11      324.4
task_0014          65,228     48,585     16,643    74.5%       2.90      343.4
task_0015          22,013     15,281      6,732    69.4%       3.18      105.3
task_0016          22,767     14,959      7,808    65.7%       2.90       94.3
task_0017          53,244     39,147     14,097    73.5%       2.79      277.9
task_0018          41,825     28,771     13,054    68.8%       2.91      197.6
task_0019          15,838     11,169      4,669    70.5%       3.43       77.4
------------------------------------------------------------------------------
TOTAL           1,308,899    966,810    342,089    73.9%       2.86     6740.4
```

**Key observations:**
- `plant` detects substantially more images (73.9% vs 58.0%) but returns far fewer masks per detected image (2.86 vs 12.36), suggesting it segments whole plants as single objects while `flower` finds many flower instances per image.
- `flower` data is 3.6× smaller on disk (1.9 GB vs 6.7 GB) despite lower detection rate.
- task_0000 and task_0006 are the largest tasks (~100K images each); task_0015/0016/0019 are the smallest.

---

## 2. Threshold Sensitivity (flower and plant prompts)

### flower prompt

```
 threshold     detected     det%    total_masks    masks/det
------------------------------------------------------------
       0.5      758,552    58.0%      9,379,478        12.36
       0.6      712,305    54.4%      6,949,091         9.76
       0.7      646,002    49.4%      4,774,055         7.39
       0.8      541,299    41.4%      2,876,444         5.31
       0.9      341,006    26.1%      1,053,353         3.09
------------------------------------------------------------
Note: threshold=0.5 is baseline (inference already filtered at this level)
```

### plant prompt

```
 threshold     detected     det%    total_masks    masks/det
------------------------------------------------------------
       0.5      966,810    73.9%      2,769,388         2.86
       0.6      825,567    63.1%      1,782,133         2.16
       0.7      643,882    49.2%      1,073,121         1.67
       0.8      428,824    32.8%        571,602         1.33
       0.9      163,348    12.5%        179,154         1.10
------------------------------------------------------------
Note: threshold=0.5 is baseline (inference already filtered at this level)
```

### Side-by-side comparison

| threshold | flower det% | flower masks/det | plant det% | plant masks/det |
|-----------|-------------|------------------|------------|-----------------|
| 0.5 | 58.0% | 12.36 | 73.9% | 2.86 |
| 0.6 | 54.4% | 9.76 | 63.1% | 2.16 |
| 0.7 | 49.4% | 7.39 | 49.2% | 1.67 |
| 0.8 | 41.4% | 5.31 | 32.8% | 1.33 |
| 0.9 | 26.1% | 3.09 | 12.5% | 1.10 |

**Key observations:**
- At **0.7**, both prompts converge to nearly identical detection rates (~49%) — a notable crossover. This is likely the best single operating threshold if both prompts are used together.
- Above 0.7, `plant` degrades much faster than `flower` (49% → 13% vs 49% → 26%), indicating `plant` accumulates more low-to-mid-confidence detections in the 0.5–0.7 band.
- `flower` retains higher detection rates at high thresholds, implying flower detections are generally higher-confidence when they fire — even though the overall rate is lower.
- The steep mask/det drop for `flower` (12.36 → 3.09) vs the gentle drop for `plant` (2.86 → 1.10) reflects that `flower` finds many instances per image at low confidence; `plant` is already conservative.
- **Recommended operating threshold**: 0.7 for `flower` (halves mask count, retains ~85% of detected images); 0.5–0.6 for `plant` (drop-off is steeper so raising threshold loses more coverage than it buys in precision).

---

## 3. Species Detection Rate (plant and flower prompts)

### Summary

| | plant | flower |
|--|-------|--------|
| Total species | 7,806 | 7,805 |
| Total images | 1,308,899 | 1,308,632 |
| Species with 0% detection | **106 (1.4%)** | **239 (3.1%)** |

### plant prompt

```
Total species: 7,806
Total images:  1,308,899
Species with 0% detection rate: 106
```

Top 20 species by detection rate (100%, selected large-count):
```
species_id             images   detected     det%
-------------------------------------------------
1356390                   196        196   100.0%
1743350                   118        118   100.0%
1360640                    76         76   100.0%
1546033                    64         64   100.0%
1357575                    59         59   100.0%
1360654                    46         46   100.0%
... (many single-image species also at 100%)
```

Bottom 20 species by detection rate (0%):
```
species_id             images   detected     det%
-------------------------------------------------
1743922                     6          0     0.0%
1741956                     5          0     0.0%
1744525                     3          0     0.0%
1628862                     3          0     0.0%
1658959                     2          0     0.0%
1502086                     2          0     0.0%
... (majority are 1–6 image species)
```

### flower prompt

```
Total species: 7,805
Total images:  1,308,632
Species with 0% detection rate: 239
```

Top 20 species by detection rate (100%, selected):
```
species_id             images   detected     det%
-------------------------------------------------
1609681                    28         28   100.0%
1628826                    22         22   100.0%
1359717                    20         20   100.0%
1743365                    15         15   100.0%
1502750                    11         11   100.0%
1389250                    11         11   100.0%
1407920                     9          9   100.0%
... (many small-count species also at 100%)
```

Bottom 20 species by detection rate (0%, notable large-count):
```
species_id             images   detected     det%
-------------------------------------------------
1424927                    22          0     0.0%
1743916                    21          0     0.0%
1361527                    10          0     0.0%
1701520                     9          0     0.0%
1365482                     9          0     0.0%
1739061                     7          0     0.0%
1531358                     8          0     0.0%
1392307                     4          0     0.0%
... (others are 1–5 image species)
```

### Side-by-side comparison and key observations

- `flower` has **2.3× more zero-detection species** (239 vs 106), confirming it is a harder/narrower concept — many plant images simply do not show flowers.
- `plant` zero-detection failures are concentrated in **very small species (1–6 images)**, suggesting isolated edge cases (unusual framing, non-standard photography) rather than systematic model failure.
- `flower` zero-detection failures include **larger species** with 10–22 images at 0% (e.g. `1424927`: 22 images, `1743916`: 21 images) — these likely represent species systematically photographed without visible flowers (herbarium sheets, bark/foliage-only shots). High-priority candidates for visual inspection.
- Top-performing species follow the same pattern: `plant` reaches 100% on much larger image counts (up to 196) than `flower` (up to 28), reflecting that the plant concept is easier to satisfy across diverse image types.
- The **zero-detection sets are largely non-overlapping** between prompts — species missed by `plant` are mostly different from those missed by `flower`. This is consistent with the 54.5% overall prompt disagreement rate and reinforces the union strategy: combining both prompts covers failures of each.

---

## 4. Prompt Agreement (plant vs flower)

```
Total images (union): 1,308,899

  both detected:       505,689  (38.6%)
  neither:              89,226  ( 6.8%)
  plant only:          461,121  (35.2%)
  flower only:         252,863  (19.3%)

  disagreement:        713,984  (54.5%)
```

### Top 20 most-disagreeing species (by disagreement rate)

```
species_id             images     both  neither   plant_only  flower_only   disagree%
-------------------------------------------------------------------------------------
1389508                    11        0        0            0           11      100.0%
1391352                    12        0        0           12            0      100.0%
1489237                     5        0        0            2            3      100.0%
1743424                     4        0        0            2            2      100.0%
1685292                     5        0        0            5            0      100.0%
1397316                     5        0        0            5            0      100.0%
1492778                     5        0        0            0            5      100.0%
... (all at 100% disagreement, mostly small species)
```

**Key observations:**
- Only 38.6% of images are detected by **both** prompts — a surprisingly small overlap given that plants generally have flowers.
- 6.8% are detected by **neither** prompt — these images are likely the hardest cases (poor framing, background clutter, non-standard views).
- `plant` is much more aggressive: it detects 35.2% of images that `flower` misses, whereas `flower` captures only 19.3% that `plant` misses.
- The two prompts are highly complementary: combining both would cover ~93.2% of images (only the "neither" bucket is uncovered).
- Species with 100% disagreement (all images detected by exactly one prompt) tend to be small-image-count species — likely those with unusual morphology where one prompt's semantic concept applies clearly but not the other.

---

## Summary & Downstream Recommendations

| Metric | plant | flower |
|---|---|---|
| Total images | 1,308,899 | 1,308,632 |
| Detection rate | 73.9% | 58.0% |
| Masks / detected img | 2.86 | 12.36 |
| Disk size | 6.7 GB | 1.9 GB |
| Species with 0% det | 106 | 239 |

- **Use `plant` as primary mask source** for whole-plant segmentation tasks; it has higher recall and cleaner per-image mask counts.
- **Use `flower` as a complementary source** for fine-grained flower-region crops, especially after threshold tightening to ≥0.7 to reduce noise.
- **Union strategy** (`plant` OR `flower`) would cover ~93% of the dataset. For the remaining 6.8% ("neither"), consider a fallback (e.g. full-image crop).
- **Score threshold 0.7** is a reasonable operating point for `flower`: halves mask count while retaining ~85% of detected images.
- **106 zero-detection species** under `plant` warrant manual review — check if images are mislabeled, non-standard (herbarium scans, etc.), or genuinely hard.
