"""Build a biogeographic-impossibility per-source mask (Exp-14 Stage 13).

Companion to build_habitat_mask.py. The aquatic mask (Stage 12) zeroed
91 fully-aquatic species globally. Stage 13 adds species that are
biogeographically impossible on specific sources but not globally:

- Alpine sources (CBN-PdlC, CBN-Pla): zero five species that don't occur
  at alpine summit elevation but were predicted there 21-51x by the
  Stage 12 winner:
    * Axonopus compressus (sid 1402995, class_idx 5524) — tropical
      American C4 carpetgrass; not naturalized in temperate alpine.
    * Equisetum x moorei (sid 1706074, class_idx 6797) — coastal/dune
      horsetail hybrid (E. hyemale x E. ramosissimum); coastal
      distribution, not alpine summit.
    * Cerastium tomentosum (sid 1391092, class_idx 3192) — Mediterranean
      lowland to montane garden escape ("snow-in-summer"); native to
      Apennines mid-elevations, not alpine summit grassland.
    * Armeria maritima (sid 1413349, class_idx 5574) — coastal sea-thrift;
      its var./subsp. *alpina* is the alpine taxon, but the *species* level
      Armeria maritima is the wrong taxon at alpine summit habitat.
    * Trifolium suffocatum (sid 1363741, class_idx 2554) — Mediterranean
      coastal annual; sandy/disturbed habitats, not alpine summit.

Other sources (RNNB, GUARDEN, LISAH, OPTMix, 2024-CEV3): no change.

Output: bool[NUM_CLASSES] mask per source group, packed in an npz.
False = blacklisted on that source group, True = admissible.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

NUM_CLASSES = 7806

ALPINE_BIOGEO_BLACKLIST = [
    {
        "species_id": 1402995,
        "class_idx": 5524,
        "name": "Axonopus compressus",
        "rationale": "Tropical American C4 carpetgrass; not naturalized in temperate alpine summit grassland.",
    },
    {
        "species_id": 1706074,
        "class_idx": 6797,
        "name": "Equisetum x moorei",
        "rationale": "Coastal/dune horsetail hybrid (E. hyemale x E. ramosissimum); coastal distribution, not alpine.",
    },
    {
        "species_id": 1391092,
        "class_idx": 3192,
        "name": "Cerastium tomentosum",
        "rationale": "Mediterranean lowland-to-montane garden escape; native Apennines mid-elevations, not alpine summit.",
    },
    {
        "species_id": 1413349,
        "class_idx": 5574,
        "name": "Armeria maritima",
        "rationale": "Coastal sea-thrift; alpine taxon is var./subsp. alpina, not the species-level A. maritima.",
    },
    {
        "species_id": 1363741,
        "class_idx": 2554,
        "name": "Trifolium suffocatum",
        "rationale": "Mediterranean coastal annual on sandy/disturbed soils; not alpine summit.",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--species-map",
        required=True,
        help="JSON list mapping class_idx -> species_id (class_to_species_id.json).",
    )
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-list-json", default=None)
    args = ap.parse_args()

    with open(args.species_map, encoding="utf-8") as f:
        c2s = json.load(f)
    if isinstance(c2s, list):
        sid_to_class = {int(s): i for i, s in enumerate(c2s)}
    else:
        sid_to_class = {int(v): int(k) for k, v in c2s.items()}

    mask_alpine = np.ones(NUM_CLASSES, dtype=np.bool_)
    audit = []
    for entry in ALPINE_BIOGEO_BLACKLIST:
        sid = int(entry["species_id"])
        cls = sid_to_class.get(sid)
        if cls is None:
            raise ValueError(f"species_id {sid} not in species_map")
        if cls != int(entry["class_idx"]):
            raise ValueError(
                f"class_idx mismatch for sid={sid}: file says {entry['class_idx']}, "
                f"species_map says {cls}"
            )
        mask_alpine[cls] = False
        audit.append({**entry, "blocked_on": ["CBN-PdlC", "CBN-Pla"]})

    mask_default = np.ones(NUM_CLASSES, dtype=np.bool_)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)) or ".", exist_ok=True)
    np.savez(args.out_npz, mask_alpine=mask_alpine, mask_default=mask_default)
    print(f"  alpine mask blacklisted: {(~mask_alpine).sum()} species")
    print(f"  default mask blacklisted: {(~mask_default).sum()} species")
    print(f"  wrote {args.out_npz}")

    if args.out_list_json:
        with open(args.out_list_json, "w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)
        print(f"  wrote {args.out_list_json}")


if __name__ == "__main__":
    main()
