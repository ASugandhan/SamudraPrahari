# Task Ledger — Samudra Prahari

| Task | Status | Key result | Date |
|---|---|---|---|
| T0.1 Scaffold | PASS | 6 tests pass, ruff clean, `import samudra` ok | 2026-09-29 |
| T0.2 Datasets | PASS | real v2 indexed: det 1329/seg 300/anom 200; 70/15/15 grouped, 0 leak; R4 separated; training wired | 2026-09-29 |
| T0.3 Colab bridge | PASS | registry validates+rejects ONNX; 4 tests pass; notebook written (Colab run pending human) | 2026-09-29 |
| T1.1 Ingest | PASS | image/XTF/JSF readers → SonarFrame; 4 tests pass (XTF via mock, real XTF = human smoke) | 2026-09-29 |
| T1.2 Bottom tracking | PASS | per-ping altitude, synthetic MAE 4.4%; no NaN; grep clean; 4 tests | 2026-09-29 |
| T1.3 Slant range | PASS | target ±1px at true ground range, per-ping altitude, water col masked; 3 tests | 2026-09-29 |
| T1.4 Nadir | PASS | per-ping nadir mask width ∝ altitude; valid_mask propagated; 3 tests | 2026-09-29 |
| T1.5 Gain/EGN | PASS | CV 0.62→0.00 (100% cut ≥50%); stripe energy →0; 3 tests | 2026-09-29 |
| T1.6 Despeckle | PASS | Lee preserves 1px line (≥3×) where median fails; uint8 out; 4 tests | 2026-09-29 |
| T1.7 Quality gate | PASS (synthetic; real 60 AWAITING) | balanced acc 0.833; threshold calibrated→yaml; 3 tests | 2026-09-29 |
| T1.8 Pipeline + tiling | PASS | round-trip 0px; single pipeline fn + fixed order; 4000×2000 in 3.18s; 8 tests | 2026-09-29 |
| T2.1 Autoencoder [COLAB] | ACCEPTED (gate unmet, by decision) | v3 real-terrain AE reconstructs (confound gone); accepted as anomaly-only/UNKNOWN screen — real validation is T4.2, not tile-AUROC. Don't quote its recall | 2026-09-30 |
| T2.2 Candidates | ACCEPTED (gate unmet, by decision) | code+eval complete; feeds origin="anomaly" candidates to UNKNOWN pool (T2.4→T4.2); standalone recall weak by design. Don't quote it | 2026-09-30 |
| T2.3 YOLO11n [COLAB] | PASS | SCTD test mAP50 0.947 (≥0.85, ~lit); YOLO11n 0.846 > YOLOv8n 0.814; ONNX 10.9MB CPU; ai4 tiny-box 0.131 (honest); synth 0.95–0.995 separate (R4) | 2026-09-30 |
| T2.4 Fusion | PASS | fused recall 0.802 ≥ max(YOLO 0.749, AE 0.671); dup 0.032 ≤0.05; anomaly_only pool 4037→UNKNOWN (weak-AE noise, prune downstream); 5 tests | 2026-09-30 |
