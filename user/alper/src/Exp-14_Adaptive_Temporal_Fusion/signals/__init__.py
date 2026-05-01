"""Per-image signals for adaptive temporal fusion.

Each signal produces a dict {quadrat_id: scalar or vector} cached to a
parquet/npz file. Signals are stage-independent — compute once, reuse
across all controller variants.
"""
from __future__ import annotations

KNOWN_SOURCES = ("CBN-PdlC", "CBN-Pla", "RNNB", "GUARDEN", "LISAH", "OPTMix")


def parse_source(quadrat_id: str) -> str:
    """Return source-prefix label for a quadrat_id.

    Mapping (longest-prefix-wins, since CBN-PdlC and CBN-Pla share CBN):
        CBN-PdlC-A1-20130807 -> 'CBN-PdlC'
        CBN-Pla-3-20140601   -> 'CBN-Pla'
        RNNB-3-1-20230512    -> 'RNNB'
        GUARDEN-CBNMed-...   -> 'GUARDEN'
        LISAH-PEC-...        -> 'LISAH'
        OPTMix-...           -> 'OPTMix'
        2024-CEV3-20240602   -> 'OTHER'
    """
    for src in KNOWN_SOURCES:
        if quadrat_id.startswith(src + "-") or quadrat_id.startswith(src):
            return src
    return "OTHER"


def parse_date(quadrat_id: str) -> str | None:
    """Return trailing 8-digit YYYYMMDD or None."""
    parts = quadrat_id.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[1]
    return None
