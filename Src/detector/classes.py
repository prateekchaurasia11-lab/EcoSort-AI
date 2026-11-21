"""Class order and general disposal guidance for the EcoShot AI detector.

The class order must match the established classifier order (E2 and earlier) and the
detector dataset's ``names`` list. Never reorder it.
"""
from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

GUIDANCE_NOTICE = (
    "General guidance only. Recycling rules differ by city and collector, so check "
    "your local rules. Model scores are uncalibrated detector scores, not "
    "probabilities of being correct."
)

GUIDANCE: dict[str, str] = {
    "cardboard": (
        "Flatten and keep dry. Clean cardboard is usually recyclable; greasy or "
        "food-soiled pieces usually are not. Lined drink cartons are a separate case."
    ),
    "glass": (
        "Empty and rinse bottles and jars for the glass stream where it exists. "
        "Wrap broken glass safely. Ceramics, mirrors and bulbs usually do not go with bottles."
    ),
    "metal": (
        "Empty and rinse cans and tins. Clean foil can often be recycled. "
        "Aerosols should be fully empty. Check local rules for mixed-material items."
    ),
    "paper": (
        "Keep clean paper dry and dispose of it with recyclable paper. Tissues, napkins "
        "and food-soiled paper usually belong in general waste."
    ),
    "plastic": (
        "Empty and rinse rigid plastic. Thin film, bags and snack wrappers are often "
        "not accepted in kerbside recycling, so check local rules."
    ),
    "trash": (
        "Treat as general (residual) waste unless local rules say otherwise. "
        "This class includes mixed-material and hard-to-recycle items."
    ),
}

assert set(GUIDANCE) == set(CLASS_NAMES)
