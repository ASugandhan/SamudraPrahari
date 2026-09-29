# SAMUDRA PRAHARI — MASTER BUILD PROMPT (v3 Architecture)

**Project:** AI-Powered Underwater Marine Debris & Anomaly Detection from Side-Scan Sonar (SIH 2026 · PS 26057 · Team NEXUS)
**Executor:** AI coding agent (Claude Code / Antigravity)
**Codebase:** Fresh repository
**Training compute:** Google Colab / Kaggle GPU (notebooks), inference on CPU / edge via ONNX Runtime
**Timeline:** 3+ weeks, full architecture

---

## 0. HOW TO USE THIS DOCUMENT (read first, agent)

You are a senior ML + full-stack engineer building Samudra Prahari from an empty repository.
This document is a sequence of **tasks**. Each task has four parts:

| Part | Meaning |
|---|---|
| **TASK** | What to build — the goal in one or two lines |
| **DO** | Exact instructions: what the task must do, files to create, constraints |
| **EXPECTED RESULT** | Measurable acceptance criteria — what "done" looks like |
| **VERIFY** | Commands/checks you run after execution to compare the ACTUAL result against the EXPECTED RESULT |

### Execution loop — follow this for EVERY task

```
1. Read the task fully, including its "Depends on" list. Confirm dependencies are PASS in the ledger.
2. Execute DO.
3. Run every VERIFY step. Capture real command output.
4. Write reports/T<ID>.md using the Result Report Template (Section 3).
5. Compare ACTUAL vs EXPECTED line by line.
   - All criteria met with evidence  → status PASS, update ledger, move on.
   - Any criterion missed            → status FAIL. Fix and re-run VERIFY. Max 3 attempts.
   - Still failing after 3 attempts  → status BLOCKED. Stop. Write what failed, what you tried,
                                       and what decision you need from the human. Do not continue
                                       to tasks that depend on it.
6. Never mark PASS without pasted evidence (test output, metric file, screenshot path).
```

### Model-training tasks (Colab)
The agent cannot run a Colab GPU itself. For any task marked **[COLAB]**:
1. The agent writes the notebook in `notebooks/` and every script it calls.
2. The human runs it on Colab/Kaggle and places the outputs (weights + `metrics.json`) in the paths stated.
3. The agent then runs VERIFY on those outputs and writes the report.
Mark the report `AWAITING_HUMAN_RUN` until the outputs exist.

---

## 1. GLOBAL RULES (apply to every task — violating any rule = automatic FAIL)

| # | Rule | Simple example |
|---|---|---|
| R1 | **Never fabricate coordinates.** No navigation data → `lat/lon = null` and `geo_status = "no_nav"`. | A PNG upload with no GPS must show "no nav data", never a made-up lat/lon. |
| R2 | **Never fabricate measurements.** Altitude comes from bottom tracking or file metadata, never a hard-coded constant. If unknown → field is `null`. | No `altitude_px = 30` defaults anywhere. |
| R3 | **Segmentation metrics = IoU / F1 of the positive class only.** Never report pixel "accuracy" for segmentation. | 99.99% of pixels are seabed; "99.9% accuracy" is meaningless. |
| R4 | **Real and synthetic results are always reported separately.** Headline numbers use real data only. | "IoU 0.42 (real, AI4Shipwrecks val)" and separately "IoU 0.71 (synthetic nets)". |
| R5 | **Split by source, not by random image.** Tiles from the same survey/wreck never appear in both train and test. | Wreck #12 tiles all go to train OR all to test. |
| R6 | **Unknown is a valid answer.** If the system is not confident, it outputs `UNKNOWN — operator review`, never a forced label. | A tyre (not a trained class) → UNKNOWN, not "cylinder". |
| R7 | **Every output carries provenance:** which model, which version, which scores produced it. | Each contact lists `yolo_conf`, `anomaly_score`, `shadow_score`, `match_distance`, `model_versions`. |
| R8 | **One canonical preprocessing order**, enforced by a test. No duplicate pipeline functions. | Only `pipeline.run()` exists; a test fails if step order changes. |
| R9 | **Config over constants.** All thresholds live in `configs/*.yaml`, not in code. | `anomaly_threshold: 0.62` in `configs/stage_a.yaml`. |
| R10 | **Estimates are labelled as estimates.** Any derived physical quantity that depends on assumptions carries `method` and `assumptions` fields. | `height_m_estimate` with `method: "shadow_geometry"`. |
| R11 | **Edge-first.** Every deployed model exports to ONNX; total model footprint target < 100 MB; inference runs on CPU. | `onnxruntime` CPU run must succeed in CI. |
| R12 | **No AI-generated images** anywhere in the repo, docs, or UI assets (SIH rule). | UI icons from an open icon set only. |

---

## 2. TARGET ARCHITECTURE (what we are building)

```
                        ┌──────────────────────────────────────────┐
 XTF / JSF (full mode)  │ INGEST  → SonarFrame (pixels + per-ping    │
 PNG / JPG (degraded)   │           nav + altitude + resolution)     │
                        └───────────────────┬──────────────────────┘
                                            ▼
┌────────────────────────── PREPROCESSING (canonical order) ─────────────────────────┐
│ 1 Bottom tracking → per-ping altitude    5 Lee despeckle (keeps thin net lines)    │
│ 2 Slant-range correction (real altitude) 6 Percentile normalise + mild CLAHE       │
│ 3 Nadir / water-column mask              7 QUALITY GATE → low tiles flagged        │
│ 4 Gain normalisation (BAC/EGN) + FFT     8 Overlapping tiling (≈20%)               │
│   stripe removal                                                                   │
└───────────────────────────────────────────┬────────────────────────────────────────┘
                                            ▼
 STAGE A — "Is anything here?"   Autoencoder anomaly map  ∪  YOLO11n boxes → candidates
                                            ▼
 STAGE B — "What is it?"         YOLO11n class · UNet mask (sprawling) · shadow verification
                                  · geometry features (area m², L/W, height estimate)
                                            ▼
 STAGE C — "Do we know it?"      Signature library match → KNOWN class  or  UNKNOWN
                                  + location cross-check vs known-objects DB → "charted" / "new"
                                            ▼
 GEO + REPORT                    pyproj geo-referencing (null if no nav) · severity score
                                  · GeoJSON / CSV / KML / PDF
                                            ▼
 BACKEND (FastAPI + ONNX Runtime)  ⇄  FRONTEND (React dashboard: upload, waterfall,
                                       map, contact queue, unknown queue, feedback, export)
```

### Repository layout (create exactly this)
```
samudra-prahari/
├── configs/                 # all thresholds & paths (YAML)
├── data/                    # gitignored; manifests committed
│   └── manifests/
├── notebooks/               # [COLAB] training notebooks
├── models/                  # ONNX weights + registry.json (git-lfs or release assets)
├── src/samudra/
│   ├── ingest/              # xtf/jsf/image readers → SonarFrame
│   ├── preprocess/          # bottom, slant, nadir, gain, despeckle, quality, tiling, pipeline.py
│   ├── stage_a/             # autoencoder, anomaly map, candidate extraction, fusion
│   ├── stage_b/             # yolo, unet, routing, shadow, geometry
│   ├── stage_c/             # signature library, open-set matcher, known-object DB
│   ├── geo/                 # georeferencing
│   ├── report/              # severity, exporters
│   └── common/              # dataclasses, config loader, logging, provenance
├── api/                     # FastAPI app
├── web/                     # React + TypeScript frontend
├── tests/                   # pytest (unit + integration)
├── reports/                 # T<ID>.md result reports + LEDGER.md
└── docs/
```

---

## 3. RESULT REPORT TEMPLATE (copy into `reports/T<ID>.md` after every task)

```markdown
# T<ID> — <Task name>
Status: PASS | FAIL | BLOCKED | AWAITING_HUMAN_RUN
Attempt: 1/3
Date:

## Expected vs Actual
| # | Expected (from master prompt) | Actual (measured) | Evidence | Met? |
|---|---|---|---|---|
| 1 | ... | ... | cmd output / file path | ✅/❌ |

## Commands run
<paste commands and trimmed real output>

## Deviations / honest notes
<anything that differs from the plan, any assumption made, any number that is weaker than hoped>

## Rules check
R1–R12 violated? (list any, or "none")
```

And maintain `reports/LEDGER.md`:
```markdown
| Task | Status | Key result | Date |
|---|---|---|---|
```

---

# PHASE 0 — FOUNDATION

## T0.1 — Repository scaffold
**Depends on:** —
**TASK:** Create the fresh repository with structure, tooling, and CI.
**DO:**
- Create the layout in Section 2. Python ≥ 3.10, `pyproject.toml`, `ruff`, `pytest`, `pre-commit`.
- `src/samudra/common/config.py`: YAML config loader with schema validation (pydantic).
- `src/samudra/common/types.py`: dataclasses `SonarFrame`, `Tile`, `Candidate`, `Contact` (fields listed in T1.1, T2.4, T5.2).
- `src/samudra/common/provenance.py`: helper that stamps model name, version, and config hash onto outputs.
- GitHub Actions CI: lint + tests on every push.
- `.gitignore` excludes `data/raw`, `data/processed`, large weights.
- `reports/LEDGER.md` initialised.
**EXPECTED RESULT:**
1. `pytest` runs and passes (≥ 1 smoke test).
2. `ruff check .` returns zero errors.
3. CI workflow file exists and is valid YAML.
4. Importing `samudra` works: `python -c "import samudra"`.
**VERIFY:**
```bash
pytest -q && ruff check . && python -c "import samudra; print('ok')"
tree -L 3 -I 'node_modules|.git'
```

## T0.2 — Dataset acquisition + manifests
**Depends on:** T0.1
**TASK:** Download public sonar datasets and create leak-free manifests.
**DO:**
- Scripts in `scripts/data/` to fetch:
  - SCTD — https://github.com/freepoet/SCTD
  - Seabed Objects-KLSG — https://github.com/YDY-andy/Sonar-dataset (and/or Kaggle mirror)
  - AI4Shipwrecks (pixel masks + terrain-only extras) — https://umfieldrobotics.github.io/ai4shipwrecks/
- Normalise labels to one class map: `shipwreck, aircraft, pipe, cylinder, ghost_net` (map dataset-specific names; document every mapping in `configs/class_map.yaml`; drop classes that don't map, e.g. "human", and log how many).
- `data/manifests/*.csv` with columns: `image_path, source_dataset, source_group_id, split, label_type (box|mask|none), sha256`.
- Split by `source_group_id` (survey/wreck), 70/15/15, fixed seed (Rule R5).
- `scripts/data/dataset_card.py` prints counts per dataset × class × split.
**EXPECTED RESULT:**
1. Manifests exist for all three datasets.
2. **Zero** `source_group_id` appears in more than one split (leakage test passes).
3. Dataset card printed; image counts recorded in the report (real numbers, not guesses).
4. Class mapping file documents every source label.
**VERIFY:**
```bash
python scripts/data/dataset_card.py
pytest tests/test_manifests.py -q   # includes leakage test + sha256 existence check
```

## T0.3 — Colab training bridge + model registry
**Depends on:** T0.1
**TASK:** Standard way to train on Colab and bring weights back.
**DO:**
- `notebooks/_template.ipynb`: mounts Drive, clones repo, installs `-e .`, pulls manifests, trains, exports ONNX, writes `metrics.json`.
- `models/registry.json`: `{model_name: {version, onnx_path, sha256, trained_on, metrics_file, date}}`.
- `scripts/register_model.py`: validates ONNX loads in onnxruntime (CPU), computes sha256, updates registry.
**EXPECTED RESULT:**
1. Template notebook runs end-to-end on Colab with a dummy model (human confirms).
2. `register_model.py` rejects a corrupt/missing ONNX with a clear error.
3. Registry entry written for the dummy model.
**VERIFY:**
```bash
python scripts/register_model.py --onnx models/dummy.onnx --name dummy
pytest tests/test_registry.py -q
```

---

# PHASE 1 — INGEST & PREPROCESSING

## T1.1 — Ingest: XTF/JSF + image mode → SonarFrame
**Depends on:** T0.1
**TASK:** Read raw sonar files into one common data structure.
**DO:**
- `ingest/xtf_reader.py` using `pyxtf`; `ingest/jsf_reader.py` (EdgeTech JSF); `ingest/image_reader.py` for PNG/JPG.
- `SonarFrame` fields: `pixels (H×W, port|starboard layout)`, `mode ("full"|"degraded")`, `ping_nav (lat, lon, heading, time per ping | None)`, `altitude_m per ping | None`, `slant_res_m_per_px | None`, `along_res_m_per_px | None`, `frequency_khz | None`, `source_file`.
- Image mode sets `mode="degraded"` and all nav/altitude fields to `None` (Rules R1, R2).
**EXPECTED RESULT:**
1. A sample XTF file loads with per-ping nav and altitude populated (use any public sample XTF; record its source).
2. A PNG loads with `mode="degraded"` and nav/altitude = `None`.
3. Unit tests cover both paths.
**VERIFY:**
```bash
pytest tests/ingest -q
python -m samudra.ingest.inspect <sample.xtf>   # prints ping count, nav present, altitude range
```

## T1.2 — Bottom tracking → per-ping altitude
**Depends on:** T1.1
**TASK:** Detect the seabed first return in every ping to get real altitude. (Replaces the old hard-coded altitude.)
**DO:**
- `preprocess/bottom_tracking.py`: per ping, smooth intensity, find first strong rise after the water column on each side (threshold + gradient), median-filter the result across pings to remove spikes.
- Output `altitude_px` per ping; convert to metres when `slant_res_m_per_px` exists.
- If the file already contains altitude, compute both and log the difference.
- In degraded mode, still estimate `altitude_px` from the image (flag `altitude_source="image_estimate"`).
**EXPECTED RESULT:**
1. On a file with recorded altitude: median absolute error between tracked and recorded altitude ≤ 10% (record actual).
2. No NaN in output; spikes suppressed (max ping-to-ping jump ≤ configured limit).
3. `grep -rn "altitude_px *= *30"` returns nothing in `src/`.
**VERIFY:**
```bash
pytest tests/preprocess/test_bottom_tracking.py -q
python scripts/eval/bottom_tracking_eval.py --file <sample.xtf>   # prints MAE vs recorded altitude
grep -rn "= *30" src/samudra/preprocess || echo "no hard-coded altitude"
```

## T1.3 — Slant-range correction (per-ping altitude)
**Depends on:** T1.2
**TASK:** Convert slant range to ground range using each ping's own altitude.
**DO:**
- `preprocess/slant_range.py`: `ground = sqrt(slant² − altitude²)` per ping (row), via `cv2.remap`, handles port/starboard.
- Pixels inside the water column (slant < altitude) become masked, not invented.
**EXPECTED RESULT:**
1. Synthetic test: a known flat seabed with a target at known ground range lands within ±1 px after correction.
2. Uses per-ping altitude (test with varying altitude passes).
**VERIFY:**
```bash
pytest tests/preprocess/test_slant_range.py -q
```

## T1.4 — Nadir / water-column masking (after slant)
**Depends on:** T1.3
**TASK:** Mask the nadir gap after slant correction.
**DO:** `preprocess/nadir.py` — mask region derived from per-ping altitude; masked pixels set to 0 with a boolean mask kept alongside (never inpainted).
**EXPECTED RESULT:**
1. Mask width follows altitude per ping (test with varying altitude).
2. A boolean `valid_mask` is returned and propagated.
**VERIFY:** `pytest tests/preprocess/test_nadir.py -q`

## T1.5 — Gain normalisation (BAC/EGN) + stripe removal
**Depends on:** T1.4
**TASK:** Remove brightness fall-off across the swath and stripe noise.
**DO:**
- `preprocess/gain.py`: Beam Angle Correction (mean intensity per incidence angle) and Empirical Gain Normalisation (mean per angle × range bin), computed over the survey; divide/normalise.
- 2D-FFT notch filter for along-track stripe noise.
- Reference implementation idea: GEOMAR `sidescantools` (GPL-3). **Do not copy its code** unless the repo is licensed GPL-3; implement from the method description.
**EXPECTED RESULT:**
1. Across-track mean-intensity profile flatness improves: coefficient of variation of the column-mean profile reduced by ≥ 50% on a sample survey (record before/after).
2. Stripe energy at the notch frequency reduced (record before/after).
**VERIFY:**
```bash
pytest tests/preprocess/test_gain.py -q
python scripts/eval/gain_eval.py --file <sample>   # prints CV before/after, saves before/after PNG
```

## T1.6 — Despeckle + contrast normalisation
**Depends on:** T1.5
**TASK:** Reduce speckle without erasing thin ghost-net lines.
**DO:**
- `preprocess/despeckle.py`: adaptive Lee filter (optionally Frost, selectable in config).
- `preprocess/contrast.py`: percentile clip (1–99) + mild CLAHE (config clip limit).
**EXPECTED RESULT:**
1. Test: flat speckle patch std decreases; a synthetic 1-px bright line keeps ≥ 3× the background intensity (median filter fails this — include that assertion as a comparison).
2. Output dtype uint8, range 0–255.
**VERIFY:** `pytest tests/preprocess/test_despeckle.py -q`

## T1.7 — Image quality gate
**Depends on:** T1.6
**TASK:** Score each tile's quality; low-quality tiles are flagged "re-survey", not analysed with guesses.
**DO:**
- `preprocess/quality.py`: no-reference score from interpretable proxies — estimated SNR, contrast (p95−p5), edge/contour density, dropout fraction (rows/cols of zeros), saturation fraction. Weighted into `quality_score ∈ [0,1]` (weights in config).
- Human labels 60 tiles as good/poor (script provides a quick labelling CLI); calibrate the threshold on those.
- Tiles below threshold get `quality_flag="low"`; downstream stages still run but contacts inherit a warning and severity cap.
**EXPECTED RESULT:**
1. On the 60 labelled tiles: balanced accuracy ≥ 0.75 for good vs poor (record actual + confusion matrix).
2. Threshold stored in `configs/quality.yaml`.
**VERIFY:**
```bash
python scripts/label_quality.py   # human step
python scripts/eval/quality_eval.py   # prints balanced accuracy + confusion matrix
```

## T1.8 — Tiling + canonical pipeline
**Depends on:** T1.2–T1.7
**TASK:** One pipeline, one order, overlapping tiles with coordinate mapping back to the full image.
**DO:**
- `preprocess/tiling.py`: tiles (default 640 px, 20% overlap), each tile stores its offset and `valid_mask`.
- `preprocess/pipeline.py`: the **only** entry point. Order: bottom → slant → nadir → gain → despeckle → contrast → quality → tiling. Each step toggleable via config but order fixed.
- Test asserts the exact step order and that no other public pipeline function exists (Rule R8).
**EXPECTED RESULT:**
1. Tile → full-image coordinate round-trip error = 0 px.
2. Order test passes; only one pipeline function exported.
3. Runtime on a 4000×2000 image logged (record actual seconds on CPU).
**VERIFY:**
```bash
pytest tests/preprocess/test_pipeline_order.py tests/preprocess/test_tiling.py -q
python -m samudra.preprocess.pipeline --file <sample> --save-steps out/   # saves one PNG per step
```

---

# PHASE 2 — STAGE A: "IS ANYTHING HERE?" (detect everything)

## T2.1 — Autoencoder training on normal seabed [COLAB]
**Depends on:** T0.2, T0.3, T1.8
**TASK:** Learn what "normal seabed" looks like so anything unusual stands out.
**DO:**
- Training data: **object-free** tiles only (AI4Shipwrecks terrain-only extras + background tiles from SCTD/KLSG with no boxes, after preprocessing). Record tile count.
- Model: convolutional autoencoder (or a small patch-feature model such as a pretrained-backbone PaDiM-style baseline if the AE underperforms — record which one is used and why).
- Loss: SSIM + L1 combined. Anomaly map = per-pixel reconstruction error, smoothed.
- Export ONNX; register (T0.3). Save `metrics.json`.
**EXPECTED RESULT:**
1. ONNX file ≤ 30 MB, loads on CPU.
2. On a held-out set of object-free tiles, mean anomaly score is clearly lower than on tiles containing objects: AUROC (tile-level, object vs no-object) ≥ 0.80 (record actual).
3. Training/validation loss curves saved.
**VERIFY:**
```bash
python scripts/register_model.py --onnx models/stage_a_ae.onnx --name stage_a_ae
python scripts/eval/ae_tile_auroc.py   # prints AUROC on held-out split
```

## T2.2 — Anomaly map → candidate regions
**Depends on:** T2.1
**TASK:** Turn the anomaly heatmap into candidate boxes/regions.
**DO:**
- `stage_a/candidates.py`: threshold anomaly map (config), morphological cleanup, connected components, discard tiny blobs below min area (config, in m² when resolution known else px), respect `valid_mask`.
- Each candidate: box, polygon, `anomaly_score` (peak and mean).
**EXPECTED RESULT (object-level, on test split with known objects):**
1. Recall ≥ 0.80 of annotated objects hit by at least one candidate (IoU ≥ 0.1 or centre-inside), **at** a recorded false-candidate rate per image. Report the full recall-vs-false-candidates curve and the operating point chosen.
2. Numbers reported separately per dataset (R4, R5).
**VERIFY:** `python scripts/eval/stage_a_recall.py --split test`

## T2.3 — YOLO11n detector training [COLAB]
**Depends on:** T0.2, T0.3, T1.8
**TASK:** Train YOLO11n for the 5 known classes.
**DO:**
- Ultralytics YOLO11n, **COCO-pretrained** initialisation (literature: pretraining gave > 54% mAP50-95 gain on sonar).
- Train on preprocessed tiles from manifests (source-group split). Sonar-safe augmentation only (flip along-track, small brightness/contrast, speckle noise; no hue/colour augmentation).
- Report per-class AP, precision, recall, mAP50, mAP50-95 on the **test** split. Export ONNX; register.
- Also run the same recipe with YOLOv8n once and record both (so the version switch is justified with our own numbers, not assumed).
**EXPECTED RESULT:**
1. YOLO11n mAP50 on SCTD test recorded (literature reference ≈ 0.90; our target ≥ 0.85). Record actual even if lower.
2. Per-class table present; ghost_net class reported separately with its real-vs-synthetic source noted (R4).
3. YOLO11n vs YOLOv8n comparison table recorded.
4. ONNX ≤ 15 MB, CPU inference works.
**VERIFY:**
```bash
python scripts/register_model.py --onnx models/stage_b_yolo11n.onnx --name yolo11n
python scripts/eval/yolo_eval.py --split test   # per-class table
```

## T2.4 — Candidate fusion
**Depends on:** T2.2, T2.3
**TASK:** Merge autoencoder candidates and YOLO boxes into one candidate list.
**DO:**
- `stage_a/fusion.py`: union of AE candidates and YOLO boxes; merge overlapping ones (IoU ≥ config) keeping both scores; candidates found only by the AE are tagged `origin="anomaly_only"` (these are the future UNKNOWN pool).
- `Candidate` fields: `box, polygon, origin (yolo|anomaly|both), yolo_class|None, yolo_conf|None, anomaly_score, tile_id`.
**EXPECTED RESULT:**
1. Fused recall ≥ max(recall of YOLO alone, recall of AE alone) on test split (record all three).
2. No duplicate candidates for the same object (duplicate rate recorded, target ≤ 5%).
**VERIFY:** `python scripts/eval/fusion_eval.py --split test`

---

# PHASE 3 — STAGE B: "WHAT IS IT?" (classify + measure)

## T3.1 — UNet segmentation for sprawling objects [COLAB]
**Depends on:** T0.2, T0.3, T1.8
**TASK:** Pixel masks for sprawling debris to get true shape and area.
**DO:**
- UNet (lightweight encoder, e.g. ResNet18/MobileNet, ImageNet-pretrained) trained on AI4Shipwrecks real masks. Optional second run including synthetic ghost-net masks (from a procedural generator in `scripts/synth/`) — report separately.
- Loss: Dice + BCE (or focal) for the extreme class imbalance.
- Export ONNX; register.
**EXPECTED RESULT:**
1. IoU and F1 of the positive class on AI4Shipwrecks **real** test split recorded. Benchmark: published SOTA ≈ 0.445 IoU (SOD), UNet ≈ 0.411. Target ≥ 0.40. Record actual even if lower. No pixel accuracy (R3).
2. Synthetic results in a separate table (R4).
3. ONNX ≤ 50 MB, CPU inference works.
**VERIFY:** `python scripts/eval/unet_eval.py --split test --real-only`

## T3.2 — Routing: rigid vs sprawling
**Depends on:** T2.4, T3.1
**TASK:** Decide which candidates need a mask.
**DO:**
- `stage_b/routing.py`: rule from config — if YOLO class ∈ {ghost_net} OR candidate aspect/solidity indicates irregular sprawl OR `origin="anomaly_only"` with large area → run UNet on the crop; else keep box.
**EXPECTED RESULT:**
1. Unit tests for each routing branch.
2. On test split, % of ghost_net-labelled objects routed to UNet ≥ 90% (record).
**VERIFY:** `pytest tests/stage_b/test_routing.py -q`

## T3.3 — Shadow verification + height estimate
**Depends on:** T3.2
**TASK:** Use the acoustic shadow to reject false positives and estimate object height.
**DO:**
- `stage_b/shadow.py`: read the shadow on the far-from-nadir side of each candidate; score solidity, width regularity, and contrast vs local seabed; class-aware (ghost nets: chaotic/weak shadow expected).
- Height estimate (only when altitude and ground range are known): `h ≈ (L_shadow × H_altitude) / (R_ground + L_shadow)`. Output `height_m_estimate`, `method="shadow_geometry"`, `assumptions=[flat seabed, ...]` (R10). Otherwise `null`.
- `calibrated_conf` = combination of detector/anomaly score and shadow score (weights in config).
**EXPECTED RESULT:**
1. On test split: false-positive rate with shadow filter vs without, at the same recall, recorded. **This is the real source for any "fewer false positives" claim in the deck** — report the actual % change, even if small.
2. Height estimate test on synthetic geometry within ±15%.
**VERIFY:**
```bash
pytest tests/stage_b/test_shadow.py -q
python scripts/eval/shadow_fp_eval.py --split test   # prints FP rate with/without at fixed recall
```

## T3.4 — Geometry features
**Depends on:** T3.3
**TASK:** Physical description of every candidate.
**DO:**
- `stage_b/geometry.py`: `area_m2` (from mask if present, else box), `length_m, width_m, aspect_ratio, solidity, orientation_deg, height_m_estimate, shadow_regularity`. Metres only when resolution known; otherwise px with `units="px"` (R2).
**EXPECTED RESULT:**
1. Synthetic shapes of known size measured within ±5%.
2. Units flag always present.
**VERIFY:** `pytest tests/stage_b/test_geometry.py -q`

## T3.5 — (OPTIONAL) Mass estimate
**Depends on:** T3.4
**TASK:** Only if the team wants it: estimate mass for nets.
**DO:** `mass_kg_estimate = area_m2 × areal_density_kg_per_m2[class]`, density table in `configs/density.yaml` **with a citation per value**, output `method` + `assumptions` (R10). Disabled by default.
**EXPECTED RESULT:** Every density value has a source; output is always labelled "estimate". If no sourced densities are found, mark task SKIPPED and state so in the deck.
**VERIFY:** `pytest tests/stage_b/test_mass.py -q`

---

# PHASE 4 — STAGE C: "DO WE KNOW IT?" (open-set)

## T4.1 — Signature library
**Depends on:** T3.4, T2.3
**TASK:** Build the reference library of what known objects look like.
**DO:**
- `stage_c/signatures.py`: for every labelled training object, store (a) an embedding (pooled YOLO11n backbone features of the crop, or a small CNN embedder), and (b) the geometry feature vector from T3.4.
- Per-class prototypes (mean + covariance or k-medoids), saved to `models/signature_library.npz` with version.
**EXPECTED RESULT:** Library contains all 5 classes with sample counts recorded; nearest-prototype classification on the **known-class** test split reaches accuracy within 5 points of YOLO's own classification (record both).
**VERIFY:** `python scripts/eval/signature_knn_eval.py --split test`

## T4.2 — Open-set rejection (UNKNOWN)
**Depends on:** T4.1, T2.4
**TASK:** Decide KNOWN vs UNKNOWN for every candidate.
**DO:**
- `stage_c/openset.py`: distance to nearest prototype (Mahalanobis or cosine) + anomaly score + YOLO confidence → `decision ∈ {KNOWN:<class>, UNKNOWN}` and `match_distance`.
- Threshold calibrated on validation.
- **Leave-one-class-out test:** retrain/rebuild the library without one class (e.g. `pipe`), then check those objects come out UNKNOWN. Repeat for each class.
**EXPECTED RESULT:**
1. Unknown-detection AUROC (held-out class vs known classes) recorded per left-out class; target mean ≥ 0.75.
2. Known-class accuracy drop from adding rejection ≤ 5 points (record).
3. This result is the evidence for the "open-set / detects never-seen debris" novelty claim — the deck must quote these real numbers.
**VERIFY:** `python scripts/eval/openset_lococ.py`

## T4.3 — Known-objects database cross-reference
**Depends on:** T5.1
**TASK:** Check if a detection is already on the chart.
**DO:**
- `stage_c/known_db.py`: load a GeoJSON of known objects (charted wrecks, pipelines, cables, previous-survey contacts). Provide a sample file `data/known_objects/sample.geojson` (clearly labelled SAMPLE, not real charts).
- For each geo-referenced contact: within `match_radius_m` (config) of a known feature → `chart_status="charted:<id>"`; else `"new"`. No nav → `"unknown_location"` (R1).
- Previous-survey contacts can be added from exported GeoJSON → enables change detection ("new since last survey").
**EXPECTED RESULT:**
1. Unit tests: inside radius → charted, outside → new, no nav → unknown_location.
2. Spatial index used (R-tree/STRtree); 10,000 known features queried in < 1 s (record).
**VERIFY:** `pytest tests/stage_c/test_known_db.py -q`

---

# PHASE 5 — GEO-REFERENCING & REPORTING

## T5.1 — Geo-referencing
**Depends on:** T1.1, T1.8
**TASK:** Pixel → lat/lon using real per-ping navigation.
**DO:** `geo/georef.py` with pyproj (WGS84 geodesic forward solve): ping position + heading + across-track ground range → lat/lon; layback/offset from config. No nav → `null` + `geo_status="no_nav"` (R1).
**EXPECTED RESULT:**
1. Synthetic track test: known offset reproduced within 0.5 m.
2. Degraded-mode input returns null coordinates, never invented.
**VERIFY:** `pytest tests/geo -q`

## T5.2 — Contacts + severity score
**Depends on:** T3.4, T4.2, T4.3, T5.1
**TASK:** Final contact records with a transparent priority score.
**DO:**
- `Contact` fields: `id, lat, lon, geo_status, decision (KNOWN:<class>|UNKNOWN), class_conf, calibrated_conf, anomaly_score, shadow_score, match_distance, area_m2, dims, height_m_estimate, chart_status, quality_flag, chip_path, provenance`.
- `report/severity.py`: documented weighted formula (weights in config), e.g. ghost_net and UNKNOWN weighted high, `chart_status="new"` boosts, `quality_flag="low"` caps. Every contact stores its score breakdown.
**EXPECTED RESULT:**
1. Severity breakdown sums to total for every contact (test).
2. Formula written in `docs/severity.md` in plain language with one worked example.
**VERIFY:** `pytest tests/report/test_severity.py -q`

## T5.3 — Exporters
**Depends on:** T5.2
**TASK:** Operator-ready outputs.
**DO:** GeoJSON (EPSG:4326), CSV, KML, and a PDF contact report (one page per contact: chip image, fields, map snippet if geo). Null coordinates exported as null / "no nav data".
**EXPECTED RESULT:** Each export validates (GeoJSON schema valid, KML opens in a validator, CSV row count = contact count, PDF page count = contact count + summary).
**VERIFY:** `pytest tests/report/test_exporters.py -q`

---

# PHASE 6 — BACKEND

## T6.1 — FastAPI service
**Depends on:** T5.3
**TASK:** Serve the full pipeline to the dashboard.
**DO:**
- `api/`: FastAPI + ONNX Runtime (CPU). Endpoints:
  - `POST /surveys` (upload XTF/JSF/PNG) → job id
  - `GET /jobs/{id}` status; `WS /jobs/{id}/progress` per-stage progress
  - `GET /surveys/{id}/contacts`, `GET /contacts/{id}`
  - `GET /surveys/{id}/tiles/{z}/{x}/{y}` waterfall/mosaic tiles
  - `POST /contacts/{id}/feedback` {action: confirm|reject|relabel, label?}
  - `GET /surveys/{id}/export?format=geojson|csv|kml|pdf`
  - `POST /known-objects` upload GeoJSON
- SQLite (default) via SQLModel; feedback stored as future training labels.
- Background worker (in-process queue is fine) for jobs.
- OpenAPI docs auto-generated.
**EXPECTED RESULT:**
1. Integration test: upload sample → job completes → contacts returned → feedback saved → export downloads.
2. Works fully offline (no external network calls at runtime).
**VERIFY:**
```bash
pytest tests/api -q
uvicorn api.main:app & sleep 3 && curl -s localhost:8000/docs | head -5
```

---

# PHASE 7 — FRONTEND DASHBOARD

**Stack:** React + TypeScript + Vite, TanStack Query, MapLibre GL (offline-capable base map), OpenSeadragon (deep-zoom waterfall/mosaic), Tailwind or CSS variables for tokens, an open-source icon set (R12). Dark "ops console" theme; must stay readable on a ship-bridge laptop.

## T7.1 — Frontend scaffold + design tokens
**Depends on:** T6.1
**DO:** Vite React-TS app in `web/`; API client generated from OpenAPI; design tokens (colours incl. severity scale, spacing, type); layout shell with left nav: Surveys · Map · Contacts · Unknowns · Reports · Settings.
**EXPECTED RESULT:** `npm run build` succeeds; `npm run lint` clean; typed API client compiles.
**VERIFY:** `cd web && npm ci && npm run lint && npm run build`

## T7.2 — Upload + job queue
**Depends on:** T7.1
**DO:** Drag-drop upload (XTF/JSF/PNG), per-job card with live stage progress over WebSocket (Ingest → Preprocess → Stage A → B → C → Report), degraded-mode badge for image-only uploads.
**EXPECTED RESULT:** Uploading the sample shows live progress through every stage and ends in "Complete" (screenshot saved).
**VERIFY:** Playwright e2e test `web/e2e/upload.spec.ts`.

## T7.3 — Waterfall / mosaic viewer
**Depends on:** T7.2
**DO:** Deep-zoom viewer with overlay toggles: YOLO boxes, UNet masks, anomaly heatmap, shadow regions, low-quality tiles hatched. Click an overlay → opens the contact.
**EXPECTED RESULT:** All 5 toggles work; clicking a box opens the correct contact (e2e test).
**VERIFY:** `web/e2e/viewer.spec.ts`

## T7.4 — Map view
**Depends on:** T7.2
**DO:** MapLibre map: contacts coloured by severity, UNKNOWN distinct marker, known-objects layer toggle, survey track line. Contacts with no nav are listed in a side panel "No location data" (never plotted).
**EXPECTED RESULT:** Geo contacts plotted at correct coordinates (e2e compares API lat/lon vs rendered feature); no-nav contacts absent from map, present in side panel.
**VERIFY:** `web/e2e/map.spec.ts`

## T7.5 — Contact queue + Unknown queue + detail card
**Depends on:** T7.3, T7.4
**DO:** Table sorted by severity (filters: class, decision, chart_status, quality). Detail card: image chip, class/UNKNOWN, all scores with plain-language "why flagged", dimensions + units, height/mass labelled "estimate", lat/lon or "no nav data", chart status, provenance. Separate Unknown queue page.
**EXPECTED RESULT:** Every Contact field from T5.2 is visible on the card (test iterates fields); sort and filters work.
**VERIFY:** `web/e2e/contacts.spec.ts`

## T7.6 — Operator feedback loop
**Depends on:** T7.5
**DO:** Confirm / Reject / Relabel buttons with keyboard shortcuts (C / R / L); status badges; feedback posted to API; export of feedback as a labelled dataset (`scripts/feedback_to_dataset.py`) for the next training round.
**EXPECTED RESULT:** Feedback persists across reload; exported dataset file contains the relabelled items with correct labels.
**VERIFY:** `web/e2e/feedback.spec.ts` + `pytest tests/test_feedback_export.py -q`

## T7.7 — Reports page
**Depends on:** T7.5
**DO:** Survey summary (counts by class/decision/chart status, quality stats) + export buttons (GeoJSON/CSV/KML/PDF).
**EXPECTED RESULT:** Each button downloads a valid file matching T5.3 checks.
**VERIFY:** `web/e2e/reports.spec.ts`

---

# PHASE 8 — INTEGRATION, BENCHMARKS, HONEST METRICS

## T8.1 — End-to-end run on held-out data
**Depends on:** all above
**DO:** Run full system on the test split + one full XTF survey. Save all outputs to `reports/e2e/`.
**EXPECTED RESULT:** Completes without error; contact count, per-stage timings, and per-class results recorded.
**VERIFY:** `python scripts/e2e_run.py --split test`

## T8.2 — Edge benchmark
**Depends on:** T8.1
**DO:** Measure on CPU (record CPU model): total ONNX footprint, per-tile latency per model, end-to-end seconds per km² (or per 1000 pings). Jetson benchmark only if hardware available — otherwise mark "not measured".
**EXPECTED RESULT:** Total model footprint < 100 MB (R11); all latencies recorded as measured numbers.
**VERIFY:** `python scripts/bench/edge_bench.py`

## T8.3 — Metrics report + deck claim audit
**Depends on:** T8.1, T8.2
**TASK:** Every number in the SIH deck must come from this report.
**DO:** Generate `reports/METRICS.md` with tables: YOLO11n per-class (real), YOLOv8n vs YOLO11n, UNet IoU/F1 (real) vs SOTA, synthetic separately, Stage A recall curve, shadow-filter FP change, open-set AUROC per left-out class, quality-gate accuracy, edge benchmark. Then produce `reports/DECK_CLAIM_AUDIT.md`: list every numeric/capability claim in the deck → source row in METRICS.md, or mark "NOT SUPPORTED — remove or label planned".
**EXPECTED RESULT:**
1. Zero deck claims without a source row or a "planned" label.
2. Specifically resolved: "faster detection" (use measured seconds per km² vs manual review time), "fewer false positives" (use T3.3 number), "detects unknown debris" (use T4.2 AUROC), "material weight" (T3.5 or removed).
**VERIFY:** Manual review by team + `python scripts/audit_deck_claims.py` (checks every claim ID has a source).

---

## 4. TASK LEDGER (fill as you go)

| Task | Status | Key result | Date |
|---|---|---|---|
| T0.1 Scaffold | | | |
| T0.2 Datasets | | | |
| T0.3 Colab bridge | | | |
| T1.1 Ingest | | | |
| T1.2 Bottom tracking | | | |
| T1.3 Slant range | | | |
| T1.4 Nadir | | | |
| T1.5 Gain/EGN | | | |
| T1.6 Despeckle | | | |
| T1.7 Quality gate | | | |
| T1.8 Pipeline + tiling | | | |
| T2.1 Autoencoder [COLAB] | | | |
| T2.2 Candidates | | | |
| T2.3 YOLO11n [COLAB] | | | |
| T2.4 Fusion | | | |
| T3.1 UNet [COLAB] | | | |
| T3.2 Routing | | | |
| T3.3 Shadow + height | | | |
| T3.4 Geometry | | | |
| T3.5 Mass (optional) | | | |
| T4.1 Signatures | | | |
| T4.2 Open-set | | | |
| T4.3 Known-objects DB | | | |
| T5.1 Georef | | | |
| T5.2 Contacts + severity | | | |
| T5.3 Exporters | | | |
| T6.1 API | | | |
| T7.1–T7.7 Frontend | | | |
| T8.1 E2E | | | |
| T8.2 Edge bench | | | |
| T8.3 Metrics + deck audit | | | |

## 5. SUGGESTED SCHEDULE (3+ weeks)

| Week | Tasks | Milestone |
|---|---|---|
| 1 | T0.1–T0.3, T1.1–T1.8, start T2.3 on Colab | Correct geometry pipeline; YOLO11n training |
| 2 | T2.1–T2.4, T3.1–T3.4 | Stage A + B working on test split |
| 3 | T4.1–T4.3, T5.1–T5.3, T6.1 | Open-set + reports + API |
| 4 | T7.1–T7.7, T8.1–T8.3 | Dashboard + honest metrics + deck audit |

Critical path: **T1.2 (bottom tracking) → T1.8 → T2.1 → T4.2.** If time runs short, cut T3.5, T4.3 change-detection extras, and T7.7 polish — never cut T4.2 (it is the novelty evidence) or T8.3 (it protects the deck).

## 6. REFERENCES (for the agent's context)
- YOLO variants on SSS (YOLOv8n vs YOLO11n, transfer learning): https://www.mdpi.com/2077-1312/14/6/550
- Class-agnostic objectness in sonar: https://arxiv.org/abs/1907.00734
- Real-time anomaly detection in SSS (salience + rarity, found AF447): https://www2.whoi.edu/staff/jkaeli/wp-content/uploads/sites/141/2019/11/Kaeli2016-AnomalyDetection.pdf
- AI4Shipwrecks benchmark (UNet/SOD IoU baselines): https://arxiv.org/abs/2401.14546
- Shadow/highlight geometric features (MLO classification): https://ieeexplore.ieee.org/document/7760991/
- No-reference sonar image quality: https://doi.org/10.1109/tip.2019.2910666
- sidescantools (bottom detection, slant, BAC/EGN, FFT stripes — GPL-3): https://github.com/sonoware/sidescantools
- PING-Mapper: https://cameronbodine.github.io/PINGMapper/
- EGN method notes: https://chesapeaketech.com/egn-updates/
