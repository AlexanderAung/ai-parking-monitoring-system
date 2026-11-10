# Five-day MVP Implementation Plan

Status: **Not started**. All task and acceptance checkboxes are intentionally open.
Planning baseline: 2026-11-09, commit `6e5dd82`. Days are focused work sessions, not assigned calendar dates.
Follow the seven-step module workflow in [AGENTS.md](../../AGENTS.md), including architecture agreement, interview, user scaffolding, and independent review handoff. Do not advance to the next day until the user agrees that the current day is complete.

## Evidence and planning assumptions

Approximately eight focused hours per day; central budget **40 hours**, estimated range **34–44 hours**. The range is a planning estimate supplied with the request, not independently established feasibility. No separate earlier audit/feasibility document was located. If actual effort exceeds capacity, discuss sequencing or schedule changes rather than quietly dropping acceptance criteria.

Verified starting points:
- Existing [validation inference](../../scripts/predict_validation.py) depends on COCO annotations; arbitrary-image inference needs adaptation.
- The local checkpoint exists at `runs/baseline-001/checkpoints/checkpoint0019.pth`; loading and hash verification are still required.
- [config.py](../../src/parking_monitoring/config.py) is a docstring-only scaffold. Historical implementation is in commit `0a3031f`; [configuration tests](../../tests/test_config.py) already require `ConfigError` and `load_config`.
- [schemas.py](../../src/parking_monitoring/schemas.py) already supplies four dataclasses; retain these contracts unless an interface change is agreed.
- Local OCR weights and [saved recognition output](../../artifacts/mvp/smoke-test/results/ocr-reading.json) exist. Its recorded `PN02XBX` differs from expected `PNO2 XBX`, even after whitespace normalization. Verify transcription before drawing conclusions; a successful call does not imply recognition accuracy.
- Package inference/storage/visualization modules and app/CLI/evaluation entry points are placeholders. Only configuration tests currently contain substantive tests.
- [labels.jsonl](../../data/mvp/labels.jsonl) is empty. [Image notes](../../data/mvp/images/images.md) are unverified candidates, not held-out labels.
- Existing documentation overlaps: [MVP overview](../mvp.md) describes scope/layout; [architecture](../architecture.md) describes boundaries; [evaluation protocol](../evaluation-protocol.md) defines measurement; [OCR decision](../decisions/003-ocr-selection.md) proposes a crop benchmark. Reuse these; this plan adds scheduling and acceptance gates.
- Local ignored [configuration notes](../../src/parking_monitoring/config.md) explain a historical implementation. Their claims and the existing MVP smoke-test checkbox do not establish current readiness. Local model/evidence links may be unavailable in a fresh clone.

### Baseline collection check (2026-11-09)

Read-only collection command: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest --collect-only -q -p no:cacheprovider`. Result: exit 2, no tests collected, three collection errors. Project tests cannot import `ConfigError` from the scaffold configuration module. Unscoped discovery also reaches vendored tests requiring CUDA-enabled Paddle and the unavailable `torchvision.datapoints` import. These are existing collection blockers, not documentation regressions; no implementation fixes were made.

The scoped command `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests --collect-only -q -p no:cacheprovider` isolates project tests. Result: exit 2, no tests collected, one collection error for missing `ConfigError`. Day 1 should use explicit `tests/` scope for MVP verification and report vendored test incompatibilities separately; passing collection alone is not passing execution.

Each day's final 1.5-hour block reserves approximately 0.75 hour for the short learning/design/review workflow, 0.25 hour for integration review/documentation, and 0.5 hour for contingency. Distribute these across the day at the required workflow steps, not after implementation. Larger learning or debugging needs trigger re-estimation. Dependencies below refer to task IDs; each day also requires the preceding day's user approval.

## Day 1 — Configuration and Detector (8 hours)

Objective: make existing assets usable through an annotation-independent detector and safe cropping.

| ID | Task | Hours | Depends on |
| --- | --- | ---: | --- |
| D1.1 | [ ] Reconcile historical configuration, current YAML, local artifact paths, and existing tests; restore agreed behavior | 1.25 | Architecture/interview/scaffold gates |
| D1.2 | [ ] Get configuration tests collecting and passing; explain any environment/import failures | 0.75 | D1.1 |
| D1.3 | [ ] Adapt EMA loading, preprocessing, inference and postprocessing for arbitrary images without a COCO loader | 2.5 | D1.1, D1.2 |
| D1.4 | [ ] Verify original-image coordinates; implement finite, ordered, clamped crop bounds and original-resolution crops | 2 | D1.3 |
| D1.5 | [ ] Learning workflow, integration review, evidence log, contingency | 1.5 | Distributed across D1.1–D1.4 |

Deliverables: reconciled settings and detector/crop interfaces, inspectable detections/crops, relevant test evidence.

Acceptance criteria:
- [ ] An arbitrary local image produces detections and crops without an annotation file or validation dataset.
- [ ] Real checkpoint loads with expected EMA parameters; model identity and configuration are recorded.
- [ ] Wide/tall image examples confirm original-size mapping; synthetic boundary tests cover invalid/non-finite/reversed/outside boxes and empty results.
- [ ] Crops come from original pixels, have valid nonempty bounds, and can be visually inspected alongside overlays.
- [ ] Configuration tests collect and pass; actual commands and any limitations are recorded.

## Day 2 — OCR and End-to-End Pipeline (8 hours)

Objective: turn real image detections into structured OCR observations and an initial persisted record.

| ID | Task | Hours | Depends on |
| --- | --- | ---: | --- |
| D2.1 | [ ] Implement PaddleOCR wrapper using local weights; preserve raw text, normalization, scores and explicit statuses | 1.75 | D1 accepted; module workflow |
| D2.2 | [ ] Verify development transcriptions and benchmark 20–50 representative crops; compare reading order/recognition-only needs | 1.5 | D2.1; representative development crops |
| D2.3 | [ ] Integrate detection, cropping and OCR with existing dataclasses, per-stage timing and model identity | 2 | D1.4, D2.1, D2.2 |
| D2.4 | [ ] Agree minimal SQLite record contract and implement initial save outside the inference function | 1.25 | D2.3; storage interface agreement |
| D2.5 | [ ] Learning workflow, integration review, evidence log, contingency | 1.5 | Distributed across D2.1–D2.4 |

Deliverables: OCR wrapper, development benchmark, structured image result, initial database save.

Acceptance criteria:
- [ ] Real local weights initialize and run; no implicit model download is required for the demonstrated path.
- [ ] Development benchmark records counts, verified text, exact matches, character errors, latency, versions and failure examples; no held-out tuning.
- [ ] A real image passes detection → original crop → OCR → structured result → initial SQLite save.
- [ ] Raw and normalized text and separate detector/OCR scores are preserved; empty recognition and technical failure are distinct.
- [ ] Pipeline returns results independently of rendering/storage; a caller invokes persistence explicitly.

## Day 3 — Storage and Visualization (8 hours)

Objective: reliably preserve complete and partial results and make them inspectable after restart.

| ID | Task | Hours | Depends on |
| --- | --- | ---: | --- |
| D3.1 | [ ] Complete SQLite storage and retrieval; define observation identity and transaction boundaries | 1.75 | D2 accepted; storage workflow |
| D3.2 | [ ] Persist source images and crops with stable references; handle file/DB write failures consistently | 1.25 | D3.1 |
| D3.3 | [ ] Reuse drawing helpers for annotated copies and plate labels without changing source pixels | 1.25 | D2 structured results; visualization workflow |
| D3.4 | [ ] Integrate multiple plates, empty results, per-plate OCR failures and restart retrieval tests | 2.25 | D3.1–D3.3 |
| D3.5 | [ ] Learning workflow, integration review, evidence log, contingency | 1.5 | Distributed across D3.1–D3.4 |

Deliverables: persistence/retrieval layer, saved source/crop files, annotations, robust repeatable pipeline.

Acceptance criteria:
- [ ] Reopening the database in a new process retrieves equivalent observations and usable source/crop references.
- [ ] Multiple plates retain their separate boxes/text/status; empty detections remain a valid image result.
- [ ] One OCR failure preserves other plate results and is recorded explicitly.
- [ ] Write failures return actionable errors without silently reporting success or presenting broken records as complete.
- [ ] Annotation does not mutate inference inputs or original stored images; tests cover boundaries and persistence integration.

## Day 4 — Streamlit Interface and Integration (8 hours)

Objective: complete the upload/inspect/save/history journey and freeze usable evaluation inputs.

| ID | Task | Hours | Depends on |
| --- | --- | ---: | --- |
| D4.1 | [ ] Implement minimal image upload and model reuse; display annotations, crops, text/status | 1.75 | D3 accepted; UI module workflow |
| D4.2 | [ ] Add explicit save and observation history; prevent duplicate records from UI reruns | 1.25 | D4.1, D3 persistence |
| D4.3 | [ ] Complete integration tests and exercise the actual UI including errors and repeated interactions | 1.5 | D4.1, D4.2 |
| D4.4 | [ ] Verify evaluation image rights, identities, boxes, transcriptions and split isolation; freeze dataset and metric rules | 2 | Development benchmark; evaluation protocol |
| D4.5 | [ ] Learning workflow, integration review, evidence log, contingency | 1.5 | Distributed across D4.1–D4.4 |

Deliverables: minimal usable Streamlit interface, integrated test evidence, verified evaluation dataset/manifest.

Acceptance criteria:
- [ ] User can upload an image, inspect annotation/crops, explicitly save, and revisit history after restart.
- [ ] UI reruns and repeated save interactions do not duplicate the same observation; intentional new-observation behavior is documented.
- [ ] Empty detections, invalid uploads and partial OCR failures are understandable in the UI.
- [ ] Real UI journey and automated integration checks are recorded, including limitations.
- [ ] Held-out set has verified boxes/text, multi/no-plate cases, legibility flags, provenance and recorded counts; related images/capture sessions stay in one split.
- [ ] Define label schema, normalization, IoU matching threshold and metric denominators before evaluation; reserve held-out images from tuning.

## Day 5 — Evaluation and Finalization (8 hours)

Objective: demonstrate a reproducible still-image MVP with measured quality and documented limits.

| ID | Task | Hours | Depends on |
| --- | --- | ---: | --- |
| D5.1 | [ ] Implement/verify evaluation runner and execute held-out manual-crop, predicted-crop and full-pipeline evaluation | 2 | D4 accepted; evaluation module workflow |
| D5.2 | [ ] Analyze exact match, CER, accepted-read accuracy/coverage, startup and warm latency; inspect failures | 1 | D5.1 |
| D5.3 | [ ] Fix critical defects and rerun affected checks, recording any test-set exposure | 1.5 | D5.2 |
| D5.4 | [ ] Verify setup and reproducibility, finish setup/results/limitations documentation and demonstration | 2 | D5.1–D5.3 |
| D5.5 | [ ] Learning workflow, independent-review handoff, evidence log, contingency | 1.5 | Distributed across D5.1–D5.4 |

Deliverables: reproducible demonstration, evaluation evidence, critical fixes, setup instructions and known limitations.

Day-specific acceptance criteria:
- [ ] Held-out evaluation records dataset version/counts, checkpoint hash, OCR/package versions, settings, hardware and actual commands/results.
- [ ] Report exact-match and CER with explicit normalization/denominators; missed readable plates count as failures, unmatched detections separately, and unreadable ground truth separately.
- [ ] Report manual-crop versus predicted-crop OCR, accepted-read quality/coverage, warm latency and startup separately.
- [ ] Fix critical functional defects and rerun relevant checks; do not tune on held-out data and still call it untouched. Disclose reuse or obtain a fresh holdout.
- [ ] Another run can follow documented setup and artifact acquisition requirements; any unavailable external assets or untested portability are explicit.

## Overall Day-five acceptance

- [ ] Complete still-image upload → detection → safe original-resolution crop → OCR → explicit save → history flow works with real models.
- [ ] Observations, source images and crops remain usable after restart; reruns do not produce accidental duplicates.
- [ ] Multi-plate, no-plate, unreadable, and partial-failure cases have verified outcomes.
- [ ] Relevant automated tests and a real UI demonstration have recorded results; implemented/tested/integrated/verified states are distinguished.
- [ ] Held-out performance and latency are measured, failure analysis is available, and limitations are documented.
- [ ] **70% full-plate exact match is a provisional evaluation threshold, not an established prediction or guaranteed result.** Confirm its denominator and suitability before evaluation. Report achieved value and whether the provisional threshold was met; a miss remains an explicit unmet quality goal for user review, never silently marked passed.
- [ ] Setup, model/data prerequisites and permitted examples are documented; unresolved blockers and deviations are disclosed.
- [ ] User accepts Day 5 completion after reviewing evidence; progress and learning records are current.

## Technical risks and responses

| Risk | Response / decision point |
| --- | --- |
| Historical configuration differs from current tests/settings | Compare contracts on Day 1; restore only agreed behavior and verify tests. |
| Detector imports, resize conventions or postprocessor coordinate order differ | Reuse proven loading logic; test non-square images and inspect crops before OCR integration. |
| Paddle package/API compatibility or initialization problems | Verify installed versions and saved local settings early; record failures; use contingency before expanding dependencies. |
| Small/blurred/multiline plates and ambiguous O/0 transcriptions | Benchmark diverse development crops and verify labels; preserve raw output; avoid unsupported substitutions. |
| Labeling effort exceeds two-hour Day 4 allowance | Begin development-label verification on Day 2; agree a feasible held-out size before Day 4 and record coverage limits. No fabricated labels or accuracy claims. |
| Incomplete file writes, DB failures or UI reruns create inconsistent history | Agree transaction/identity policy, test failure recovery, and use explicit save with idempotent behavior. |
| Local ignored assets are unavailable elsewhere | Record asset identities, hashes and acquisition steps; do not confuse local existence with distributability or reproducibility. |
| Interview/scaffolding/debugging exceeds estimates | Use reserved contingency, surface blockers promptly, and agree revised schedule without skipping workflow gates. |

## Excluded features

Live RTSP cameras; video processing; vehicle tracking; vehicle-to-plate association; entry/exit events; parking occupancy/billing; OCR training/fine-tuning; cloud deployment; production-level architecture. Discuss scope additions before work.

## Completion recording

After each module, report files, actual test commands/results, decisions and learning evidence. After each day, verify criteria and update this plan plus [completed work](../progress/completed-work.md), including partial work, blockers and deviations. Do not mark planned work completed. Maintain [Q&A](../learning/q-and-a.md) and the [skill profile](../learning/skill-profile.md) without inventing answers or ability claims.
