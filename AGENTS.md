# Project Operating Instructions

## 1. Project Identity and Objective
Project: AI Parking Monitoring System. Build an AI-powered license-plate monitoring application.
MVP pipeline: Image → License Plate Detection → Cropping → OCR → SQLite Storage → Visualization.
Primary technologies: Python, RT-DETRv2, PaddleOCR, SQLite, and Streamlit.

## 2. Current Project Context
Verified by repository inspection on 2026-11-09 at commit `6e5dd82`; recheck before implementation.
- Initial `git status --short` was empty: no tracked modifications or untracked, non-ignored files.
- Preserve all existing files, ignored local artifacts, and any subsequent uncommitted changes.
- Fine-tuned checkpoint exists locally at `runs/baseline-001/checkpoints/checkpoint0019.pth`; existence does not prove current loadability.
- [Validation inference](scripts/predict_validation.py) loads EMA weights and uses a COCO validation loader and target original sizes. Adapt it for arbitrary images without annotations.
- [schemas.py](src/parking_monitoring/schemas.py) defines `Detection`, `OCRResult`, `PlateResult`, and `ImageResult`. Preserve the filename and agreed contracts; never rename it to `schema.py`.
- [config.py](src/parking_monitoring/config.py) currently contains only a docstring. Commit `0a3031f` contains an implementation; reconcile it with [settings](configs/mvp.yaml) and [tests](tests/test_config.py), not by blind restoration.
- Local ignored [configuration notes](src/parking_monitoring/config.md) describe historical implementation, not current code. They may be absent in a fresh clone.
- Local OCR weights exist under `artifacts/mvp/smoke-test/ocr-models/{det,rec,cls}`; saved [recognition evidence](artifacts/mvp/smoke-test/results/ocr-reading.json) records a successful call but an exact-text mismatch. This is not a benchmark or a fresh verification.
- Reuse drawing helpers in [visualize_coco.py](scripts/visualize_coco.py), [visualize_coco_sizes.py](scripts/visualize_coco_sizes.py), and validation inference; the package visualization module is still a placeholder.
- Detector/OCR/pipeline/storage wrappers, Streamlit app, and prediction/evaluation entry points are placeholders. Crop, pipeline, and storage tests are placeholders; configuration tests contain real checks.
- Baseline test collection fails on missing `ConfigError`; unscoped pytest also finds incompatible vendored tests. Use explicit `tests/` scope for MVP checks; see the plan for commands/results.
- `data/mvp/labels.jsonl` is empty; [image notes](data/mvp/images/images.md) require transcription and identity verification.
- [README](README.md), [MVP overview](docs/mvp.md), [architecture](docs/architecture.md), [evaluation protocol](docs/evaluation-protocol.md), and [OCR decision](docs/decisions/003-ocr-selection.md) remain useful references.
- README's non-bundled checkpoint statement is compatible with a local ignored copy. The MVP overview's checked smoke-test item is historical evidence, not completion of this plan.
- No separate prior repository audit or feasibility assessment was located. User-supplied historical findings were checked where possible; runtime compatibility, artifact integrity, and held-out performance remain unverified.

## 3. MVP Scope and Non-goals
In scope: local still images, Latin registration plates, detection, original-resolution crops, OCR, SQLite persistence, annotated output, a minimal Streamlit UI, and basic evaluation/testing.
Out of scope: live RTSP, video, tracking, vehicle-to-plate association, entry/exit events, occupancy, billing, OCR training/fine-tuning, cloud deployment, and production-level architecture.
Discuss any additional feature with the user before introducing it.

## 4. Seven-step Engineering Workflow
Follow all seven steps for each new module; existing scaffolds do not bypass agreement or learning.

### Step 1 — User + Codex: Design the architecture
Discuss responsibility, boundaries, dependencies, data flow, and proposed public interfaces.
Explain important decisions and trade-offs; obtain agreement before implementation.

### Step 2 — Codex: Adaptive engineering interview
Read the [skill profile](docs/learning/skill-profile.md) and relevant [Q&A history](docs/learning/q-and-a.md).
Ask three module-specific questions by default: conceptual, engineering/architecture, and applied problem-solving.
Keep most approachable and include at least one stretch question; adapt difficulty to evidence and avoid already-mastered questions.
Wait for the user's answers before proceeding. Evaluate them and briefly correct misconceptions.
Record questions, original answers, feedback, and gaps in Q&A; update the profile only with sufficient evidence.
Keep the interview short and relevant to the upcoming implementation.
"Ask questions one at a time when useful. Use my answers to adapt subsequent questions or ask brief follow-ups. Keep the default to three main questions per module; do not turn follow-ups into an unlimited examination."
"Preserve my original answers exactly as given in the historical Q&A record. Never rewrite them to make them appear more technically correct."
"Distinguish between understanding a concept, explaining it independently, and successfully implementing it. These are different levels of evidence."

### Step 3 — User + Codex: Scaffold the core interfaces
Propose the minimum functions/classes and explain their responsibilities and connections.
Show signatures, parameters, return types, and required public-interface docstrings.
Ask the user to write the skeleton; review it and suggest corrections before proceeding.
Focus on interface understanding rather than guessing names. Do not implement internal logic yet.

### Step 4 — Codex: Complete the implementation
After interface agreement and reviewed scaffolding, implement internal logic against those interfaces.
Prioritize a functional MVP, readable code, and appropriate error handling; avoid unnecessary abstractions.
Never silently change an approved interface; explain the need and obtain agreement first.

### Step 5 — User: Review the implementation
Guide the user through responsibility, data flow, important algorithms, decisions, and failure cases.
Support their review with focused explanations rather than line-by-line lectures.

### Step 6 — Codex: Test and integrate
Run relevant tests, fix defects, and integrate with the existing application.
Verify boundaries and actual behavior; report exact commands and real results, including failures and skipped checks.
Never claim unrun tests passed.

### Step 7 — User + ChatGPT: Independent review
Prepare a concise handoff: responsibility/interface, architecture decisions, implementation, tests/results, limitations, unresolved issues, and learning gaps.
The user takes this summary to ChatGPT for independent architecture and engineering review.

## 5. Adaptive Learning and Interview Rules
Engineering execution is primary; learning is integrated into module implementation.
[Q&A history](docs/learning/q-and-a.md) preserves dated, module-specific questions, verbatim original answers, feedback/corrections, and gaps.
Never improve or rewrite historical answers; append follow-up corrections as new evidence.
[Skill profile](docs/learning/skill-profile.md) is a concise evolving assessment of strengths, developing skills, and practice needs.
Separate conceptual understanding from implementation ability; repository artifacts alone do not establish personal authorship or mastery.
Update from actual evidence and use previous records to adapt questions; one correct answer is not proof of mastery.
Avoid excessive questioning, unrelated exercises, and long theoretical lectures.

## 6. Coding and Implementation Principles
- Prefer simple, readable Python and apply SOLID principles pragmatically.
- Give functions clear responsibilities, appropriate type hints, and useful public-interface docstrings.
- Handle expected failure cases and minimize additional dependencies.
- Avoid premature abstractions, overengineering, and production architecture during the MVP.
- Preserve established interfaces; prioritize functional completeness and sound fundamental architecture.
- Inspect Git status before edits; preserve existing work and never restore historical files blindly.

## 7. Testing and Verification Standards
- Run relevant tests after implementation; cover important edge cases, boundaries, and integration points.
- Verify real model integration when applicable; mocked checks alone do not establish inference behavior.
- Report actual commands, outcomes, environment limits, and checks not run.
- Distinguish implemented, tested, integrated, and verified functionality.
- Never fabricate results or mark work complete solely because code exists.

## 8. Five-day Execution Roadmap
Follow the [detailed five-day plan](docs/plans/mvp-5-day-plan.md); approximately eight focused hours/day, 34–44 estimated hours overall.
1. Configuration and detector: arbitrary-image inference and safe original-resolution crops.
2. OCR and end-to-end pipeline: local models, crop benchmark, structured results, initial save.
3. Storage and visualization: durable retrieval, files, overlays, and partial-failure handling.
4. Streamlit and integration: upload, inspect, explicit save/history, rerun safety, evaluation data.
5. Evaluation and finalization: held-out measurements, critical fixes, reproducibility, and documentation.
The 70% exact-match threshold is provisional, not a performance prediction.

## 9. Documentation and Progress Management
After each module, report modified files, implementation and actual test results, and update relevant learning records.
After each day's goals:
1. Verify the day's acceptance criteria with evidence.
2. Mark only verified goals complete in the plan; keep unmet criteria open.
3. Append details to [completed work](docs/progress/completed-work.md).
4. Record unfinished tasks, blockers, deviations, and any revised estimates.
5. Wait for the user's agreement that the day is complete before proceeding to the next day.
Keep partial work explicitly partial. Generated code, planned work, and saved historical evidence do not establish completion.
This initial documentation task does not authorize starting Day 1 implementation.
