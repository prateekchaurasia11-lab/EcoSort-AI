# EcoSort-AI Multi-Object TACO-to-YOLO Dataset Preparation (23 Nov 2026)

**Author:** Prince Verma (230108048 — Dataset Lead)  
**Branch:** `dataset-preparation`  
**Source Dataset:** Official TACO (`annotations.json`, 1,500 images, 4,784 boxes)  
**Output Directory (Gitignored):** `Data/yolo_taco_2026-11-23/`

---

## 1. Conversion & Group-Split Summary

All 1,500 TACO images were fetched into `Data/taco_raw/` (`0` failed downloads) and converted to 6-class YOLOv8 format (`0: cardboard`, `1: glass`, `2: metal`, `3: paper`, `4: plastic`, `5: trash`) via `Src/yolo_prepare.py` using batch-level group splitting (`seed=42`) to prevent scene/session leakage across splits.

| Split | Converted Images | Share | Leakage Status |
|---|---:|---:|---|
| `train` | 935 | 65.8% | Disjoint capture batches |
| `val` | 185 | 13.0% | Disjoint capture batches |
| `test` | 300 | 21.1% | Disjoint capture batches |
| **Total** | **1,420** | **100.0%** | **0 cross-batch leakage** |

*(80 images contained only deferred/unmapped categories and were cleanly excluded from label export.)*

---

## 2. 6-Class Bounding-Box Distribution (`4,015` Verified Boxes)

| Class ID & Name | Converted Boxes | TACO Categories Mapped |
|---|---:|---|
| `0: cardboard` | 80 | Cardboard, Corrugated carton, Egg carton, Toilet tube |
| `1: glass` | 254 | Glass bottle, Glass cup, Glass jar, Broken glass |
| `2: metal` | 544 | Aluminium foil, Can, Drink can, Food Can, Aerosol, Metal bottle cap, Metal lid, Pop tab, Scrap metal |
| `3: paper` | 204 | Paper bag, Paper cup, Paper straw, Wrapping paper, Magazine paper, Normal paper |
| `4: plastic` | 1,764 | Clear/Other plastic bottle, Bottle cap, Lid, Film, Carrier/Garbage/PP bag, Straw, Utensils, Disposable/Other plastic cup, Container, Other plastic, Tupperware, Spread tub, Six pack rings |
| `5: trash` | 1,169 | Cigarette, Other plastic wrapper, Crisp packet, Styrofoam piece, Foam food container/cup, Tissues, Aluminium blister pack, Food waste, Shoe |
| **Total Mapped** | **4,015** | **Ready for E6 lightweight YOLO detector training (`data.yaml`)** |

### Deferred / Excluded Boxes (`769` Boxes)
- `Unlabeled litter` (`517`), `Other carton` (`93`), `Drink carton` (`45`), `Disposable food container` (`38`), `Meal carton` (`30`), `Rope & strings` (`29`), `Squeezable tube` (`7`), `Plastic glooves` (`4`), `Pizza box` (`3`), `Battery` (`2`), `Carded blister pack` (`1`) — held out pending composite-carton and unknown/hazardous rejection policy sign-off.

---

## 3. Manual Visual Verification Sign-Off
- **Visual Overlays Inspected:** `Data/yolo_taco_2026-11-23/visual_checks/` (`25` sampled images across `train`, `val`, `test`).
- **Status:** **PASSED** — bounding boxes align accurately with object boundaries and 6-class label mappings; no corrupt images or coordinate inversions observed.
