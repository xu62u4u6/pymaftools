"""Domain grouping of MAF ``Variant_Classification`` values.

This is a *domain fact* (which raw variant class belongs to which coarse
functional bucket), not a plotting concern, so it lives in the core layer.
The plot layer (``ColorManager``) imports these and attaches colours; keeping
the mapping here is the single source of truth and lets core compute grouped
statistics (e.g. per-sample TMB by functional group) without importing plot.
"""

from __future__ import annotations

# Coarse functional grouping of Variant_Classification. The full ~18 raw
# categories are too noisy for stacked bars / legends; this buckets them into
# six functionally meaningful groups.
FUNCTIONAL_GROUP = {
    "Missense_Mutation": "Missense",
    "Nonsense_Mutation": "Truncating",
    "Frame_Shift_Del": "Truncating",
    "Frame_Shift_Ins": "Truncating",
    "Translation_Start_Site": "Truncating",
    "Nonstop_Mutation": "Truncating",
    "Splice_Site": "Splice",
    "Splice_Region": "Splice",
    "In_Frame_Del": "In-frame",
    "In_Frame_Ins": "In-frame",
    "Silent": "Silent",
    "3'UTR": "Other",
    "5'UTR": "Other",
    "3'Flank": "Other",
    "5'Flank": "Other",
    "Intron": "Other",
    "IGR": "Other",
    "RNA": "Other",
    "Targeted_Region": "Other",
}

# Stack / legend order, most-damaging first. Variant classes not in
# FUNCTIONAL_GROUP fall back to "Other".
FUNCTIONAL_ORDER = ["Truncating", "Splice", "Missense", "In-frame", "Silent", "Other"]
