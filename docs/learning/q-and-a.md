# Engineering Q&A History

The first configuration interview is recorded below. Answers preserve the user's original wording; explanations supplied by Codex are separate evidence.

Before each module interview, review related entries and the [skill profile](skill-profile.md). Ask three relevant questions by default: conceptual, engineering/architecture, and applied problem-solving. Keep most approachable with at least one stretch question, then wait for answers. Avoid repeating demonstrated mastery; use gaps to guide future questions.

Append dated module entries using the template below. Preserve the user's original answers verbatim, including uncertainty or mistakes. Never rewrite them to appear better. Add later corrections as explicitly dated follow-ups. Distinguish Codex feedback from the user's own “What I learned” reflection; leave unanswered fields pending rather than inventing responses.

## [Date] — [Module]

Template only; copy for future interviews.

### Question 1 — [Conceptual]

**Question:**

**My original answer:**

**Feedback:**

**What I learned:**

### Question 2 — [Engineering/architecture]

**Question:**

**My original answer:**

**Feedback:**

**What I learned:**

### Question 3 — [Applied problem-solving]

**Question:**

**My original answer:**

**Feedback:**

**What I learned:**

### Learning gaps identified

- [Evidence-based gap and next practice opportunity]

### Concepts demonstrated

- [Concept and supporting answer or observed implementation]

### Follow-up evidence

- [Date, clarification or scaffold/review observation; link any profile update]

## 2026-11-10 — Configuration

### Architecture agreement before the interview

The user stated:

> I deleted the tests/test_config.py, you will need to rewrite it after we finished implemented our config.py. 
>
> and I agree with your proposal. Yes, A component should validate the resources it needs, rather than forcing unrelated components to depend on each other's availability.

Agreed boundary: configuration reads YAML, validates structure and values, resolves paths, and returns structured settings. The app/CLI supplies relevant settings to components. Each component validates its own resources. Loading configuration does not initialize/download models or create outputs. Dictionary versus dataclass representation has been discussed; dataclasses are Codex's recommendation, pending interface agreement.

### Question 1 — Conceptual

**Question:**

A YAML file can be valid YAML but still contain invalid application settings. What's the difference between parsing configuration and validating it? For example, consider `detector.score_threshold: 1.5`. What would parsing establish, and what should validation check?

**My original answer:**

> Parsing would establish whether if its a valid yaml or not, and validation is that if that parsed result are within our specified values range. 

**Feedback:**

Correct distinction between valid YAML and application value constraints. Parsing also constructs Python values; validation additionally covers required fields, types and supported choices. The threshold must be within the inclusive range [0, 1]. This is limited evidence of conceptual understanding, not implementation ability.

**What I learned:** Not supplied by the user.

### Question 2 — Engineering/architecture

**Question:**

After validation, we could return either a dictionary (`config["detector"]["score_threshold"]`) or a typed settings object (`config.detector.score_threshold`). Which would you choose for this project, and why? Consider how other components would use it and how easily we could catch mistakes. It's fine if typed settings objects are unfamiliar.

**My original answer:**

> I dont' know why, never thought about it, should I choose a dictionary? 

**Feedback:**

A dictionary is a reasonable simple option. Codex recommended dataclasses to make fields explicit, support editor/static type checks, and pass component-specific settings. Explained that `frozen=True` helps prevent accidental reassignment and dataclasses do not automatically validate values. This explanation was supplied with guidance; independent understanding and implementation remain unassessed.

**What I learned:** Not supplied by the user.

### Question 3 — Applied problem-solving

**Question:**

Suppose our project is at `/project/parking`, and the YAML contains `detector.checkpoint_path: runs/model.pth`. We launch the app while the terminal's working directory is `/tmp`. Where should that checkpoint path resolve, and what could go wrong if we simply used `Path("runs/model.pth")`?

**My original answer:**

> i think it should be resolved in project root? and i don't know how to anchor the project root and where? 

**Feedback:**

Correct intended base: the project root, producing `/project/parking/runs/model.pth`. The user explicitly identified a gap in implementing that anchor. Codex explained that a relative Path uses the process working directory when accessed, so the example could look under `/tmp/runs/model.pth`. For this source-checkout layout, `Path(__file__).resolve().parents[2]` in `src/parking_monitoring/config.py` identifies the root. An explicit root override was proposed for tests or another layout; no user implementation has been observed.

**What I learned:** Not supplied by the user.

### Learning gaps identified

- Practice defining typed settings containers and distinguish type annotations from runtime validation.
- Practice locating the source-checkout root with pathlib and resolving relative settings independently of the working directory.
- Interface scaffolding and implementation ability remain unassessed.

### Concepts demonstrated

- Distinguished YAML syntax from application value-range validation (Question 1).
- Identified the project root as the intended path base, with uncertainty and an explicit implementation gap (Question 3).
- Agreed with the proposed component-owned resource validation after the architecture explanation; independent architecture design was not assessed.

### Follow-up evidence

- Interview complete; interface agreement and user scaffolding are pending. No implementation or tests run during this interview.
- 2026-11-10: Following the dataclass and path-anchor explanation, the user replied: "Ok, I agree. Let's go". Frozen dataclass containers are accepted. This agreement does not establish independent explanation or implementation ability. Proceeding to the public-interface contract and user-written skeleton; no internal implementation yet.
- 2026-11-10: The user reported completing the skeleton. Review found all four containers' fields, the exception, and the loader signature/stub matched the supplied contract. The first two decorators used `fronzen=True`; Codex corrected this to `frozen=True` and supplied the missing OCRConfig, StorageConfig and AppConfig docstrings. The scaffold was written against a detailed supplied contract, so this is supported scaffolding evidence, not independent interface design or validation implementation.
- Scaffold check: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -c 'import parking_monitoring.config; print("Configuration skeleton imports successfully")'` exited 0 and printed the success message. This establishes importability of the corrected skeleton only; `load_config` still raises `NotImplementedError`, and no configuration tests were run.
- 2026-11-10: The user said "ok, go" after scaffold review. Codex implemented internal YAML reading, structure/value checks, path resolution and grouped settings construction without changing the public interface. Step 5 user review is pending; test rewriting/execution and integration follow that review. This implementation is Codex-authored and provides no additional evidence of the user's implementation mastery.
- 2026-11-10: After the implementation review guide, the user instructed: "Ok, rewrite the tests and run." Codex rewrote tests/test_config.py, then ran `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider`: exit 0, 80 passed in 0.33s. No loader defects or test failures occurred in this run. This authorization is not evidence of independently explaining the implementation; tests are Codex-authored. An independent-review handoff is prepared at ../reviews/configuration-handoff.md; the external review is not yet recorded.

## 2026-11-10 — Detector

### Architecture agreement

The user approved the detector design with: "Ok, agree."

Agreed interface: RTDETRDetector(DetectorConfig) initializes reusable resources;
predict(Pillow image) returns list[Detection] in original-image pixels.
The CLI opens image paths and handles drawing/saving. The detector owns
preprocessing, inference and postprocessing. No checkpoint copy is needed;
the configured epoch-19 checkpoint remains the source. See architecture.md.
No detector implementation or real-model verification has occurred.

### Question 1 — Conceptual

**Question:**

Our detector settings contain two files: config_path, the RT-DETR model
configuration YAML, and checkpoint_path, the trained .pth checkpoint. What do
you think each file provides, and why might the checkpoint alone be insufficient
to initialize the detector?

**My original answer:**

> config_path is model architecture and checkpoint is the weight, we need both to make inference.

**Feedback:**

Correct distinction for this state-dictionary loading path. The configuration
constructs the model and checkpoint weights must match its structure/shapes.
Codex additionally explained that the checkpoint contains training state and
that this baseline restores its EMA weights. The user's answer does not
independently establish understanding of EMA selection or checkpoint loading.

**What I learned:** Not supplied by the user.

### Question 2 — Engineering/architecture

**Question:**

Suppose we process 100 images using one RTDETRDetector instance. Should
checkpoint loading happen in __init__() or inside every predict() call? Why,
and which work still needs to happen separately for each image?

**My original answer:**

```text
it should happen in **\_\_ init \_\_(), because it should only initialized once, instead of reloading everytime it predict. Loading once, predict multiple time.&#x20;**
```

**Feedback:**

Correct model-lifetime reasoning: initialize once per instance and reuse it
across predictions. The user did not enumerate per-image work; Codex supplied
preprocessing, the forward pass, original-size postprocessing and score filtering.
This is conceptual evidence, not observed implementation.

**What I learned:** Not supplied by the user.

### Question 3 — Applied problem-solving (stretch)

**Question:**

An original image is 1280 pixels wide x 720 pixels high. We resize it directly
to 640 x 640, without padding. A box measured on the resized image is
(x_min, y_min, x_max, y_max) = (160, 160, 480, 480). What would its coordinates
be on the original image? Can we use the same scaling factor for both axes?

**My original answer:**

```text
I m totally lost on this one. But i ll try, here is my approach 
original = 1280 x 720, 
resize = 640 x 640, 
So i need to calculate 160, 160, 480, 480, of 640 x 640 is XXXX 
i think if we can get the scaling factor of original to resize ratio, we can just multiply that with coordinates and get the original scale back. 
Does it answer the question? 
```

**Feedback:**

The original/resized ratio and multiplication approach is correct. The answer
does not yet calculate coordinates or distinguish the two axis ratios. Codex
worked through x scale = 1280/640 = 2 and y scale = 720/640 = 1.125, yielding
(320, 180, 960, 540). One shared factor would be incorrect for this direct
non-square-to-square resize. RT-DETR's postprocessor performs this mapping
when supplied the original dimensions; do not scale its pixel output again.
The worked arithmetic and axis-specific explanation are guided evidence,
not an independently completed solution.

**What I learned:** Not supplied by the user.

### Learning gaps identified

- Practice independent width/height scaling for directly resized images and
  distinguish resized-image coordinates from already postprocessed original pixels.
- EMA weight selection, original-size argument ordering, and preprocessing
  implementation remain unassessed.
- Detector interface scaffolding and actual model integration have not been observed.

### Concepts demonstrated

- Distinguished model architecture configuration from trained weights.
- Explained why checkpoint loading belongs in initialization and the instance is reused.
- Proposed the correct ratio-based mapping approach, without completing axis-specific arithmetic.

### Follow-up evidence

- Three-question interview complete. User-written detector scaffolding is next;
  exact exception/threshold-boundary contracts still need specification.

### Follow-up — Skeleton responsibilities

**Original message:**

```text
Ok, let's go to detector skeleton. Here is my guess for the&#x20;

1. i need to load the two config, one is model architecture and second is weight&#x20;
2. Do i also run the inference by just running this module?&#x20;
3. i wont need to touch the preprocessing and postprocessing, since it is already done in the model architecture, right?&#x20;



what do you think?  
```

**Codex clarification:**

- There are two model-loading inputs: the architecture YAML and the weights checkpoint; the checkpoint is not a second configuration file.
- detector.py defines a reusable class; the CLI script constructs it and calls predict. Running the definitions alone does not execute image inference.
- The YAML declares relevant components/transforms, but constructing the model does not automatically run image preprocessing or the separate postprocessor. Our predict method must invoke the sequence explicitly, reusing the existing postprocessor and matching validation transforms.
- This is a clarification within the completed interview/scaffolding discussion, not a fourth main interview question. The user's uncertainty identifies a practice need around the distinction between declared configuration, neural-network forward execution, and caller orchestration.
- A two-method skeleton with docstrings and NotImplementedError stubs is being supplied for the user to write. Internal loading, transforms and inference remain pending reviewed scaffolding.

### Follow-up — Detector skeleton review

- The user reported: "Ok, check my skeleton, i think there is somehting wrong."
- Observed skeleton: method signatures, type hints, docstrings and stubs followed
  the supplied contract. The class was named RTDETRdetector instead of the agreed
  RTDETRDetector. The predict docstring began at nine spaces while its raise
  statement used eight, producing an IndentationError. Codex corrected these
  issues and minor class-docstring spelling/import spacing.
- Before correction, `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import ast; from pathlib import Path; ast.parse(Path("src/parking_monitoring/detector.py").read_text(encoding="utf-8"))'`
  exited 1 with `IndentationError: unindent does not match any outer indentation level`
  at line 42. The line reported was the raise statement, although the inconsistent
  block indentation started at the opening docstring.
- After correction, `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -c 'from parking_monitoring.detector import RTDETRDetector; print("Detector skeleton imports successfully")'`
  exited 0 and printed the success message. This checks corrected skeleton
  importability only; initialization and prediction still raise NotImplementedError.
- This is supported scaffolding against a supplied contract, not independent
  detector design or implementation. Practice consistent indentation and exact
  public symbol names. Internal implementation and real-model tests are pending.

### Follow-up — Detector internal implementation

- The user authorized implementation with: "ok, now implement it."
- Codex implemented the reviewed constructor/predict signatures, restoring EMA
  weights and constructing the configured validation transforms/postprocessor
  without creating a dataset or loader. The per-image call preserves original
  pixels and supplies original [width, height] for one coordinate conversion.
- Private backend loading and input/output checks support the agreed error
  contract; no additional public methods or result types were introduced.
- The implemented module imports successfully. The checkpoint SHA-256 matches
  the saved manifest, but no checkpoint loading or image inference was executed.
  Step 5 user review comes next, followed by Step 6 tests/real-model verification.
- Internal code is Codex-authored. The import/hash checks are not evidence of
  the user's independent implementation or review understanding; no personal
  skill assessment is promoted on that basis.

### Follow-up — Detector unit tests

- After the implementation guide, the user stated: "Ok, i now understand it, what is our next step. report me first."
  This is a self-report of understanding, not an independent review explanation.
- The user then instructed: "ok, write and run detector tests. after that we will do step 2 together."
  Scope: write/run unit tests now; reserve real checkpoint verification (item 2
  in the preceding verification plan) for a joint next activity. The completed
  module interview is not being restarted.
- Codex added tests/test_detector.py with controlled backend components, mocked
  checkpoint deserialization and a tiny real PyTorch module for strict synthetic
  EMA-state matching. The targeted run passed 69 cases; the final scoped project
  run passed 149 cases in 1.99s. The initial Ruff run found four SIM117 context-
  manager style findings; they were corrected and the final Ruff run passed.
  No detector implementation fixes were required. Exact commands/results are
  recorded in ../progress/completed-work.md.
- No real checkpoint, vendored model or actual image inference was run. Model
  compatibility/accuracy and real transform/postprocessor behavior remain unverified.
  Tests are Codex-authored; passing them supplies no new evidence of the user's
  independent test-design or implementation mastery.

### Follow-up — Guided real-checkpoint initialization

- Following the supplied Python-session sequence (load_config, then
  RTDETRDetector(config.detector)), the user reported exactly:
  "Detector initialized; EMA checkpoint loaded successfully."
- This is user-run success evidence for the constructor and strict EMA restoration
  against the configured model. The agent did not rerun initialization; the full
  startup log/device print and timing were not supplied. The current default YAML
  specifies CPU and the existing epoch-19 baseline checkpoint.
- Guided command execution is practical usage evidence, distinct from independently
  implementing the loader or explaining its algorithms. No skill-mastery promotion
  follows from the success message alone.
- Next: guide the first real image prediction in the same session. The available
  image291_jpg.rf.fbd83e8ba4fbf26dfb8e92d3e43d2c21.jpg candidate is 480 x 480 RGB,
  visually contains a plate, and is selected for a development smoke test.
  Prediction has not yet been run. Two historical filenames in images.md were
  absent when checked; they were not supplied as runnable inputs.

### Follow-up — Guided first real-image prediction

**User's original output:**

```text
Original image size: (480, 480)
[Detection(bbox_xyxy=(287.6219787597656, 223.14540100097656, 392.65081787109375, 332.2291564941406), score=0.8843116760253906)]
```

- This followed the supplied image291 development-image commands using the
  existing detector instance. It provides user-run inference evidence with one
  returned box. The agent did not execute or rerun the prediction.
- Coordinates are finite, ordered and inside the reported dimensions. Visual
  alignment and wide/tall image mapping still need inspection; the score is a
  detector output and does not establish calibrated accuracy or OCR correctness.
- Next, guide the user to reuse the existing validation drawing helper on the
  returned detections and inspect a saved overlay. No new main interview question
  is introduced; guided runtime execution is not independent implementation mastery.

### Follow-up — First overlay inspection

**User's original message:**

> I hve checked it and it is great! It is here, you can check if you want "/Users/boy/Desktop/ai-parking-monitoring-system/artifacts/mvp/results/detector-check-de6537eb/overlay.png"

- The agent opened that saved overlay and checked its metadata (480 x 480 RGB).
  The visible rectangle encloses the plate and the drawn 0.884 score matches the
  reported detection rounded to three decimals. This is direct artifact inspection
  in addition to the user's successful guided initialization/prediction reports.
- D1.3's arbitrary-image smoke path is verified. Wide/tall mapping and safe crops
  remain D1.4 work; one overlay is not an accuracy benchmark. Detector's independent-
  review handoff is prepared; no external review findings have been supplied.
- The user completed guided runtime commands and reported visual satisfaction.
  This does not establish independent implementation or formal evaluation skill;
  original interview answers and evidence distinctions remain preserved.

## 2026-11-11 — Cropping

### Architecture agreement

The user accepted the proposed crop contract with: "Agree, let's go".

Use crop_plate(original Pillow image, floating-point xyxy box, padding_pixels=0)
in pipeline.py, returning (crop image, exact integer xyxy bounds). Validate
finite/ordered coordinates, nonnegative integer padding and original-image
overlap before padding. Expand, round outward, clamp and ensure nonempty bounds.
Raise ValueError for invalid/unusable inputs; preserve original pixels/source;
caller owns configuration loading, OCR, display and persistence. See architecture.md.

The three-question interview begins now, one question at a time. No answers or
feedback are inferred from agreement; scaffolding and implementation are pending.

### Question 1 — Conceptual

**Question:**

The detector resizes images to 640 x 640 internally. Why should crop_plate()
extract the plate from the original image instead of that resized version?
Consider both the returned box coordinates and the detail OCR needs to read characters.

**My original answer:**

> I guess it is To get the full resolution? we first need to convert the bbox coordinates back to the original ratio than we crop it.

**Feedback:**

Correct reason to retain original character detail. The proposed extra coordinate
conversion is unnecessary here: the detector's postprocessor already returns
original-image coordinates. Codex clarified that crop_plate receives them directly
and performs validation, padding, rounding and clamping, without scaling again.
This is partial conceptual understanding with a responsibility-boundary gap;
independent correct coordinate handling has not been observed.

**What I learned:** Not supplied by the user.

### Question 2 — Engineering/architecture

**Question:**

Why should crop_plate() return the exact integer bounds it used, alongside the
cropped image, when we already have the detector's original box? Consider what
padding, rounding, and clamping can change.

**My original answer:**

> I don't know, why do we need both? my guess is bec the detector's original box are in floats? we need discrete number to select the pixels? I think i m wrong.

**Feedback:**

Correctly identified discrete pixel selection. Codex explained the additional
reason for returning bounds: preserve the original model prediction separately
from the exact region actually extracted after padding/rounding/clamping. The
bounds support crop reproducibility and later PlateResult.crop_bbox_xyxy use.
The user did not independently explain that result-contract rationale.

**What I learned:** Not supplied by the user.

### Question 3 — Applied problem-solving (stretch)

**Question:**

An image is 100 pixels wide x 60 pixels high. With zero padding, the detector
gives (x_min, y_min, x_max, y_max) = (-3.2, 10.8, 30.1, 65.4). Using our rules
(round left/top down, right/bottom up, then clamp), what integer bounds should
we use? Should this partially outside box be rejected, or can we extract a valid crop?

**My original answer:**

> i don't know this one.

**Feedback:**

No solution was supplied; Codex provided a worked example. Round to
(-4, 10, 31, 66), then clamp x boundaries to [0, 100] and y boundaries to [0, 60],
giving (0, 10, 31, 60). The box overlaps the image, so it produces a valid
31 x 50 crop. Bottom 60 is a valid exclusive bound; the last included row is 59.
The arithmetic and overlap explanation are guided instruction, not a solution
demonstrated by the user. No additional main interview questions are added.

**What I learned:** Not supplied by the user.

### Learning gaps identified

- Reinforce that postprocessing already produces original-pixel boxes; crop code
  must not scale them again.
- Practice outward rounding (including negative values), clipping each coordinate
  against its own image dimension, and exclusive right/bottom bounds.
- Distinguish partial overlap (valid after clamping) from no overlap (reject),
  and understand why the returned crop bounds accompany original detections.
- User crop scaffolding and implementation ability remain unassessed.

### Concepts demonstrated

- Identified original-resolution detail as a reason for cropping original pixels.
- Identified discrete pixel selection as a reason to use integer crop bounds.
- These observations do not establish independent geometry or implementation mastery.

### Follow-up evidence

- Three-question interview complete. Codex is supplying the agreed function
  signature/docstring and NotImplementedError stub for user-written scaffolding.
- No crop implementation or tests have been written or run during this interview.

### Follow-up — Crop skeleton syntax review

- User request: "check there is any syntax error in my implementation."
- The user's function signature, return type, docstring contract and stub matched
  the supplied interface. The opening docstring had five leading spaces and
  raise NotImplementedError had four. AST parsing exited 1 with IndentationError
  at the raise on original line 27; the inconsistent block began at the docstring.
- Codex aligned both body statements to four spaces and normalized signature/import
  spacing. The signature's original continuation indentation and omitted trailing
  comma were legal Python, not the cause of the syntax failure.
- After correction, `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import ast; from pathlib import Path; path = Path("src/parking_monitoring/pipeline.py"); ast.parse(path.read_text(encoding="utf-8"), filename=str(path)); print("Syntax check passed")'`
  exited 0 and printed the success message. No crop function execution or tests
  occurred; the body still raises NotImplementedError.
- This repeats the docstring/body indentation pattern seen in the detector
  skeleton. Supported scaffolding is observed; independent diagnosis or internal
  implementation is not inferred. The reviewed crop skeleton is ready for internals.

### Follow-up — Approved crop implementation

- After requesting an exact implementation list, the user replied: "approve".
- Codex implemented only crop_plate internals (plus the standard-library math
  import), preserving the approved public signature and return contract.
  Validation, unpadded overlap checks, padding/outward rounding/clamping and
  original-pixel extraction are present. No full inference orchestration is added.
- Integer padding commutes with floor/ceil; the code uses integer arithmetic for
  that equivalent calculation to avoid float overflow/precision loss. This is an
  implementation explanation supplied by Codex, not independently demonstrated
  user reasoning or an added interview question.
- The module imports and Ruff/whitespace checks pass. No crop invocation or
  behavior tests were run; Step 5 user review precedes Step 6 testing/integration.
- Internal logic is Codex-authored. No personal mastery assessment is promoted
  based on the generated implementation or static checks.

### Follow-up — Requirement annotations, crop tests and real output

**User's original request:**

> ok, before writing test and verify real output, take each lines from your implementation list and annotate above its respective codes. after that you can carry out your tests

- Codex added numbered comments mapping all eight approved implementation items
  to their respective code blocks, then wrote 91 crop tests. No production logic
  changes were needed. The initial targeted run passed with one pytest parameter-
  iterator deprecation warning; converting that new-test parameter input to a
  list resolved it. The final scoped suite passed 240 cases in 2.17s without warnings.
- Real crop verification used the user's earlier detection box on the actual
  source image, not a new detector run. Bounds (287, 223, 393, 333), size 106 x 110,
  pixel identity, mode/source preservation and lossless saved PNG checks passed.
  The agent opened the saved crop and confirmed it contains the visible plate.
  Evidence: artifacts/mvp/results/crop-check-t1fcbitw/.
- Real detector wide/tall mapping remains pending. Tests and verification scripts
  are Codex-authored; requirement annotations support review but do not demonstrate
  the user's independent understanding or implementation mastery. No new questions
  or inferred user reflections are added.

### Follow-up — Development wide/tall image selection

- User request: "ok, can you find, select and choose?"
- Agent scanned local metadata and visually inspected candidates, selecting
  pic_286 from data/raw/valid (2048 x 1536) and image037 from data/mvp/images
  (361 x 480) before running predictions. Both have fully visible plates.
- Selection and file hashes are recorded in the progress log and
  artifacts/mvp/results/non-square-selection-9t8m101u/selection.json.
  These are development inputs; one comes from the already-used baseline
  validation pool. No new held-out evaluation, inference results or user mastery
  evidence is claimed. Real wide/tall verification is still the next activity.

### Follow-up — Guided manual driver and wide-image result

- The user requested guess-first feedback instead of a complete recipe. Through
  iterative snippets, the user corrected the coordinate field selection, package-
  qualified imports, execution guard/setup placement, and an early return for
  empty detections after focused prompts. This is guided applied evidence; it
  does not establish independent mastery of those patterns. User-authored source
  edits are preserved, including the current manual main in pipeline.py.
- The user later explicitly requested display code; Codex supplied the Pillow
  image-viewer method. No independent implementation authorship is attributed
  for that supplied refinement.
- The user's original wide-image output was:

```text
(374, 330, 1818, 1213)
(1444, 883)
```

- The user then confirmed: "i ve checked it and all fine. what is next?"
  The reported size agrees with the bounds and selected 2048 x 1536 source.
  Visual coverage is user-observed; the agent has not viewed a saved wide result.
- Next is the tall image037 at 361 x 480. No new main interview questions are
  introduced. Current source edits require fresh tests before final acceptance;
  no new runtime checks were performed in this feedback turn.

### Follow-up — Tall result and delegated verification

**User's original tall-image output:**

```text
Original image size: (361, 480)
(157, 248, 224, 285)
(67, 37)
```

**User's original visual-confirmation message:**

```text
For step 1.
i ve check the pic and it is great. 
where do i save? 
```

- Visual satisfaction is user-reported. Codex supplied a fresh-folder save snippet.
- The user then delegated the remaining run/save work: "i ve added it to the main, you finish that job, and report me after it finished."
- Codex refined the user's manual runner to process both preselected examples
  using one detector instance, close files, use configured padding, save PNG and
  result metadata, and preserve the guarded entry point. No approved public
  config/detector/crop signature changed. A redundant rethrow was removed.
- The agent ran actual CPU main/model inference with only the desktop viewer
  suppressed, verified exact native crop pixels/source hashes, and opened both
  saved overlays and crops. Both cases reproduced the user's bounds. No model
  or prediction was mocked. Final scoped suite: 240 passed in 1.74s; Ruff passes.
- These real execution results verify implementation behavior. The user's
  earlier guided fixes demonstrate developing application of imports, function
  calls/return values and control flow after prompts. Resource handling, looped
  runner refinements, metadata saving and automated verification here are
  Codex-authored; independent mastery is not inferred. External review and
  Day 1 completion agreement remain pending.

### Follow-up — Day 1 review completion reported

**User's original message:**

> I ve finished the review. So that's all for Day1?

- Review completion is recorded as reported by the user. No detailed review
  findings, independent explanation or additional implementation evidence was
  supplied; none is inferred and no mastery assessment is changed.
- All Day 1 planned tasks/technical criteria are checked. Final day-completion
  agreement remains to be recorded before moving to Day 2. No source changes,
  model execution or test rerun occurred in this closeout feedback turn.
