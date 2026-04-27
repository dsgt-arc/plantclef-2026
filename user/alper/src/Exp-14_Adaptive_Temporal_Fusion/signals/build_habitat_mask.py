"""Build a habitat-impossible species mask.

Marks fully-aquatic species (submerged or floating) that cannot
plausibly appear in a dry-soil quadrat photograph. Output: npy bool
array [NUM_CLASSES] where False = blacklisted.

Run once. Output is consumed by submission generation as a second mask
applied alongside the existing geo mask (sw_europe_mask_bool.npy).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

NUM_CLASSES = 7806

AQUATIC_GENERA = {
    'Posidonia', 'Zostera', 'Cymodocea', 'Halodule', 'Halophila', 'Phyllospadix',
    'Ruppia', 'Lemna', 'Spirodela', 'Wolffia', 'Wolffiella',
    'Potamogeton', 'Stuckenia', 'Groenlandia',
    'Myriophyllum', 'Najas', 'Vallisneria', 'Ceratophyllum', 'Hydrilla',
    'Egeria', 'Elodea', 'Hydrocharis', 'Stratiotes',
    'Cabomba', 'Limnobium', 'Salvinia', 'Azolla',
    'Nymphaea', 'Nuphar', 'Nymphoides',
    'Pistia', 'Eichhornia', 'Pontederia',
    'Aldrovanda', 'Utricularia',
    'Aponogeton', 'Hottonia',
}

AQUATIC_EPITHETS = {
    'oceanica', 'submersa', 'submersum', 'demersum', 'fluitans', 'natans',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species-map", required=True,
                    help="JSON list mapping class_idx -> species_id (e.g. class_to_species_id.json).")
    ap.add_argument("--species-names", required=True,
                    help="JSON dict species_id -> Latin name (utf-8).")
    ap.add_argument("--out-npy", required=True)
    ap.add_argument("--out-list-json", default=None,
                    help="Optional JSON of blacklisted (sid, name) entries for review.")
    args = ap.parse_args()

    with open(args.species_map, encoding="utf-8") as f:
        species_map = json.load(f)
    with open(args.species_names, encoding="utf-8") as f:
        names = {int(k): v for k, v in json.load(f).items()}

    mask = np.ones(NUM_CLASSES, dtype=np.bool_)
    blacklist = []
    for class_idx, sid in enumerate(species_map):
        nm = names.get(int(sid), "")
        parts = nm.split()
        if len(parts) < 2:
            continue
        genus = parts[0]
        epithet = parts[1].rstrip(',').lower()
        if genus in AQUATIC_GENERA or epithet in AQUATIC_EPITHETS:
            mask[class_idx] = False
            blacklist.append({"class_idx": int(class_idx),
                              "species_id": int(sid), "name": nm})

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npy)) or ".",
                exist_ok=True)
    np.save(args.out_npy, mask)
    n_blacklisted = (~mask).sum()
    print(f"  blacklisted {n_blacklisted} / {NUM_CLASSES} species")
    print(f"  wrote {args.out_npy}")

    if args.out_list_json:
        with open(args.out_list_json, "w", encoding="utf-8") as f:
            json.dump(blacklist, f, ensure_ascii=False, indent=2)
        print(f"  wrote {args.out_list_json}")


if __name__ == "__main__":
    main()
