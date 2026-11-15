# Day 2 OCR verification and development benchmark

Status: **Proposed execution specification; no fresh inference or benchmark run yet.**
Prepared 2026-11-12. Applies to D2.1/D2.2 in mvp-5-day-plan.md.
The separate baseline detector evaluation plan remains a different workstream.

## Current evidence and scope

- OCR wrapper implemented and reviewed; 134 controlled OCR unit cases and the
  full 374-case project suite pass. Controlled tests do not verify real models.
- Read-only inspection found 54 local MVP images, four existing smoke crops,
  and an empty data/mvp/labels.jsonl. images.md contains historical candidates,
  including conflicting/incomplete identities; do not treat it as ground truth.
- Package metadata currently reports PaddleOCR 2.10.0, PaddlePaddle 3.3.1,
  NumPy 2.5.3, Pillow 12.3.0 and jiwer 4.0.0. Re-record at execution time.
- Use the approved wrapper and crop helper. Application pipeline and SQLite
  implementation remain separate Day 2 modules; this work supplies OCR evidence.
- Preserve source, existing outputs, model files and all uncommitted work.
  Create fresh output folders and record source/config/input/model hashes.

## 1. Prepare a reproducible fresh local-model run

1. Check Git status and hash the source, configuration, lockfile and local model
   files. Compare model hashes against saved provenance where available; record
   mismatches instead of silently replacing files or claiming provenance.
2. Record OS, CPU/chip, architecture, Python and package versions. Resolve all
   model paths through load_config. Record configured version and individual
   detector/recognizer/classifier identities, rather than assuming all three
   component weights share the configured PP-OCRv4 name.
3. Keep the current CPU settings: English, PP-OCRv4, one CPU thread, classifier
   off, MKLDNN off, OCR threshold 0.50 and padding 0. Do not tune during this run.
4. Prepare a small verification driver around PaddleOCREngine, keeping its public
   interface unchanged. Review any needed driver scaffold before implementation.
5. Execute in a fresh subprocess with network access denied for that subprocess
   using the available /usr/bin/sandbox-exec. Fail if offline enforcement cannot
   be applied; do not silently substitute a weaker claim of offline verification.
   This restricts the verification process and changes no system permission.
6. Construct one real engine. No fake backend, model substitution or inference
   mocking. The classifier files are preflighted, but classifier inference/model
   loading is not claimed while use_angle_cls is false.

Use these existing development crops, after checking that they remain readable:

| Input | Purpose |
| --- | --- |
| artifacts/mvp/smoke-test/crops/plate-000.png | Historical single-crop comparison; old transcription is unverified |
| artifacts/mvp/results/crop-check-t1fcbitw/crop.png | Native 106 x 110 crop from the square-image check |
| artifacts/mvp/results/wide-check-4811d7a2/crop.png | Native 1444 x 883 Day 1 crop |
| artifacts/mvp/results/tall-check-d5a0d005/crop.png | Native 67 x 37 Day 1 crop |

Do not infer plate line count from source-image orientation. Inspect each crop.
Run every crop once and then three further calls using the same engine. Save
every result, raw backend region output and latency. A recording adapter may
capture actual backend output while forwarding the real call unchanged; record
the adapter and serialize results after timing, without a second hidden inference.

Verify source image pixels, dimensions, mode and file hashes remain unchanged.
Check outputs are JSON-compatible and scores/statuses follow the approved rules.
Record constructor time (including deferred backend imports), first-call time
and subsequent-call times separately. Calls are timed around recognize(), with
images already open; exclude disk loading, detector work and model startup.

### Runtime gate

Proceed only when real detection/recognition models initialize with networking
denied, the driver completes without technical error results, source/model files
are preserved, and at least one visually legible crop yields usable text. Empty
recognition on an individual crop is a valid unreadable outcome and is recorded.
All-empty output on the selected legible crops needs investigation before a
benchmark. Correct text is not required to prove runtime operation: mismatches
must be retained and investigated as quality evidence, not hidden or corrected.

If initialization/inference fails, preserve logs and diagnose the exact boundary.
Fix demonstrated defects, add focused regression coverage where needed, rerun
relevant checks and repeat the failed real verification. Discuss any required
public-interface change before implementing it.

## 2. Freeze verified development data before benchmark predictions

1. Target 30 distinct readable source plates (within the planned 20-50 range).
   Inspect candidates before viewing their OCR predictions. Include clear,
   small, blurred, angled and multiline examples where available; tags may
   overlap. Report actual category counts and any coverage gaps.
2. Establish source identity, hashes, original-pixel manual plate bounds and
   related-capture grouping. Keep all related material in development, and avoid
   reserved held-out inputs. Disclose known baseline training/validation reuse.
3. Independently transcribe from source/crop pixels. Treat old notes as hints.
   Prepare a labeled contact sheet; refer unresolved ambiguities to the user.
   Do not infer O/0, I/1 or another character from what the recognizer predicts.
4. Keep genuinely ambiguous/unreadable ground truth separate and report its
   count. Do not score it as readable text or exclude difficult readable plates
   because they later produce poor OCR. If the target cannot be met, report the
   available count and resolve the dataset gap before claiming benchmark completion.
5. Agree the label record before populating labels.jsonl. Proposed fields:
   source_id, image_path, image_sha256, plate_id, group_id, split="development",
   bbox_xyxy (original pixels, right/bottom exclusive), transcription_raw,
   legibility, tags, verified_by and verified_at. Use null for unknown grouping
   or unavailable transcription; never invent verification provenance.
6. Generate original-resolution manual reference crops with crop_plate. Save
   their bounds, hashes and paths in a frozen benchmark manifest linked to the
   label rows. Recheck labels.jsonl before writing and preserve existing entries.

Before new benchmark helpers are implemented, complete a short metric-specific
assessment (exact match/CER, denominators and fair latency comparison), then the
agreed user scaffold, implementation, user review and test sequence. Do not
restart the already-completed OCR interview. Trace at least one hand-computed
example and reuse existing libraries for edit-distance counts.

## 3. Run the frozen comparison

Compare these two modes on the same reference crops, local weights and settings:

- A: the actual PaddleOCREngine path, text detection plus recognition.
- B: whole-crop recognition-only through a benchmark adapter to the installed
  backend. Apply the same normalization, optional-score and acceptance rules.
  This adapter adds no mode flag or method to the approved public wrapper.

Run each mode in a separate fresh subprocess under the same offline restriction.
Both may initialize the same full backend; report this and do not claim optimized
recognition-only startup. Warm each mode once on a designated non-benchmark smoke
crop, then take three calls per frozen crop in fixed order. Use the first timed
reading for accuracy, retain all readings and flag disagreements. Never choose
the best repeat or silently use voting. Store all times; report per-crop medians
and overall p50/p95 of those medians (linear-interpolated percentiles).

Keep empty recognition and per-crop technical failures in the records and the
readable-plate metric denominator. Do not silently retry them out of the summary.

After the reference-crop comparison, run a separate paired predicted-crop check
on the same source plates with the existing real EMA detector and crop_plate.
Keep detector score > 0.60 and padding 0. Match original detector boxes to manual
boxes per image at IoU >= 0.50: sort eligible pairs by descending IoU with stable
ID tie-breaking, assigning each prediction/truth at most once. Verify matching
on small examples before using it. Report matched-crop OCR quality separately;
also report source-plate exact matches over all readable source plates, counting
missed detections as failures, plus unmatched predictions separately. Do not
describe this development driver as the completed application pipeline.

## 4. Metrics and checks fixed before execution

Normalize both predictions and reference text by uppercasing and removing
whitespace only; preserve raw text separately. Let N be readable labeled plates
in the specified crop population, A accepted reads and C correct accepted reads.

- Exact-match rate = exact normalized matches / N, including uncertain readings
  when their text is correct. Empty/error results are incorrect, not removed.
- Character error rate = total substitutions + deletions + insertions divided
  by total reference characters across N plates. Empty/error output is the empty
  prediction, so it contributes deletions. Use corpus counts, not mean per-plate CER.
- Accepted-read accuracy = C/A; report N/A when A=0. Coverage = A/N.
- Report all status counts, sample/character denominators, per-category counts,
  mismatches, reading-order failures and representative original crops.
- Report startup/first-call/warm latency and the precise timing boundary. No
  detector/storage/disk time is included in the reference OCR latency measure.

Verify metrics using exact, substitution, insertion, deletion and empty-prediction
examples, including an uncertain correct text and an accepted incorrect text.
Independently recalculate aggregate counts from saved per-crop rows and check
totals. Do not change thresholds to improve these baseline numbers. Any later
development tuning must be a separately identified run; held-out data stays unused.

## 5. Outputs, proposed commands and closeout

Each run gets a new output directory under artifacts/mvp/results. Planned outputs:

- Verification: environment.json, inputs.jsonl, raw-output.jsonl,
  readings.jsonl, timings.json, preservation.json and stdout/stderr logs.
- Benchmark: frozen manifest.jsonl, reference/predicted crops and contact sheets,
  raw-output.jsonl, per-crop-results.jsonl, timing rows, matching rows,
  metrics.json, failures.json and environment/config/model identities.
- Human-readable report: docs/reports/mvp-001/ocr-development.md with actual
  commands, counts, measurements, inspected failures and reproducibility limits.

These are proposed commands for drivers to be prepared/reviewed; the scripts
do not exist yet. They are not claims of runnable or already-executed commands.

```bash
OCR_VERIFY_DIR="$(mktemp -d artifacts/mvp/results/ocr-local-XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/sandbox-exec \
  -p '(version 1)(allow default)(deny network*)' \
  .venv/bin/python scripts/verify_ocr_local.py \
  --config configs/mvp.yaml --output-dir "$OCR_VERIFY_DIR"

# After reference labels/crops are verified and the benchmark manifest is frozen:
OCR_BENCH_DIR="$(mktemp -d artifacts/mvp/results/ocr-dev-XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/sandbox-exec \
  -p '(version 1)(allow default)(deny network*)' \
  .venv/bin/python scripts/benchmark_ocr_crops.py \
  --config configs/mvp.yaml --manifest "<frozen-reference-manifest.jsonl>" \
  --modes detect-recognize recognize-only --warmup 1 --repeats 3 \
  --paired-predicted-crops --iou-threshold 0.50 --output-dir "$OCR_BENCH_DIR"
```

Execute the important commands together with the user and inspect one actual
reading, metric example and failure together. The user attempts interpretation
before Codex supplies it. Fix evidence-backed problems, rerun checks appropriate
to actual changes, update the OCR decision/progress/learning records and prepare
the independent-review handoff. Mark only evidenced criteria complete.

The 70% exact-match figure remains provisional, not a guaranteed result or a
silent pass/fail gate. A poor but honest development result may establish the
need to revise the approach; it does not establish held-out performance. Day 2
completion still requires its pipeline/save work and the user's acceptance.
Label verification and any new evaluation tooling can exceed the original D2.2
estimate; report actual effort and discuss re-estimation rather than silently
dropping data verification, representative coverage or acceptance criteria.
