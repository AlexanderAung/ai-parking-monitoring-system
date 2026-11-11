# Engineering Skill Profile

Initial evidence review: 2026-11-09, repository commit `6e5dd82`. This is a working assessment, not a permanent judgment of ability.

Repository artifacts show project experience, but their existence and Git authorship do not establish how independently the user designed, implemented or understood them. No original interview answers or observed user scaffolding were available for this assessment. Consequently all personal assessments start as **Not assessed**, rather than inferring mastery or lack of ability. Conceptual understanding and practical implementation ability both remain unassessed unless stated otherwise.

| Category | Current assessment | Evidence | Next development area |
| --- | --- | --- | --- |
| Python fundamentals | Conceptual: Developing in imports, function calls and control flow; independent implementation: Not assessed | [2026-11-11 crop/driver Q&A](q-and-a.md): corrected package imports, coordinate-field access, main guard and early return through iterative guided snippets; repeated indentation corrections needed. | Practice argument/return contracts, resource handling and purposeful exception handling; explain choices independently. |
| Object-oriented programming | Not assessed | Four result dataclasses and historical configuration dataclasses (`0a3031f`) provide discussion material, not personal competency evidence. | Explain data containers versus behavior-bearing classes and when composition helps. |
| Data structures and algorithms | Not assessed | [Validation inference](../../scripts/predict_validation.py) uses lists, dictionaries and tensors; personal reasoning about them has not been observed. | Choose result containers and reason through filtering and crop-bound operations. |
| Software architecture and interface design | Not assessed | [Architecture](../architecture.md) separates inference, UI and persistence; personal design reasoning is not recorded. | Explain module boundaries and scaffold agreed public interfaces. |
| ML inference and model integration | Not assessed | [Baseline report](../reports/baseline-001/report.md), checkpoint and inference code establish project-level work, not independent implementation ability. | Explain EMA loading, preprocessing/postprocessing and model lifetime; adapt arbitrary-image inference. |
| Computer vision | Not assessed | [Visualization helpers](../../scripts/visualize_coco_sizes.py) and baseline evidence concern original-image coordinates and size effects. | Reason through non-square resize mapping, clamping, rounding and original-resolution cropping. |
| OCR | Not assessed | Local [saved recognition evidence](../../artifacts/mvp/smoke-test/results/ocr-reading.json) shows a call and mismatch; no user explanation or benchmark is available. | Explain raw versus normalized text, score/status semantics, reading order and label ambiguity. |
| Data and database management | Not assessed | [Storage module](../../src/parking_monitoring/storage.py) is a placeholder; absence of project implementation is not evidence of inability. | Design observation identity, file references, transactions and restart retrieval. |
| Testing and debugging | Not assessed | [Configuration tests](../../tests/test_config.py) contain real checks; other MVP test files are placeholders. No direct evidence of the user's test design/debugging reasoning. | Explain a failing import, design boundary cases, and interpret actual test results. |
| Engineering problem-solving | Not assessed | [Technical appendix](../reports/baseline-001/technical-appendix.md) records project troubleshooting; individual reasoning and degree of assistance are unknown. | Explain a diagnosis with alternatives, then verify a chosen fix against evidence. |

## Assessment labels

- **Not assessed:** insufficient attributable evidence; not a negative assessment.
- **Developing:** observed partial understanding or implementation with gaps/support needed.
- **Demonstrated:** repeated relevant evidence of correct reasoning or implementation in the stated scope.
- **Strongly demonstrated:** consistent independent reasoning and application across varied cases, including failures and trade-offs.

## Updating the profile

1. Read this profile and relevant [Q&A history](q-and-a.md) before each module interview; use uncertainty to calibrate questions rather than assume beginner or expert status.
2. Base changes on original answers, observed user scaffolding, review explanations and debugging behavior. Cite the dated module entry and relevant file/task.
3. Record conceptual and practical assessments separately in a row when evidence differs; note assistance and the limited scope demonstrated.
4. Do not treat a single correct answer, generated code, a commit author or successful project artifact as proof of mastery. Seek corroborating evidence before promotion.
5. Track strengths, developing skills and practice needs concisely. Avoid mastered questions and target evidenced gaps with approachable questions plus a stretch question.
6. Append a brief dated change note explaining evidence and any revised assessment. Preserve original answers in Q&A; correct mistaken assessments transparently.
7. Treat missing or mixed evidence as uncertainty. Revisit assessments as new evidence arrives; do not invent answers, grades or skill levels.

## Assessment change history

### 2026-11-10 — Initial configuration interview

Limited conceptual evidence in the [configuration Q&A](q-and-a.md): the user
distinguished parsing from value-range validation and identified the project
root as the intended path base. The user requested guidance on dictionaries
versus typed settings and on implementing the path anchor. Codex supplied those
explanations. These observations do not yet justify promoting the broad category
assessments above; independent explanation and implementation remain unassessed.
Next practice: typed settings scaffolding and pathlib root/path resolution.

Repository-context correction: `tests/test_config.py` is now empty, and the user
confirmed deleting its earlier contents. The baseline table's reference to real
checks is historical, not a statement about current tests. Rewrite tests after
configuration implementation as agreed.

2026-11-10 follow-up: the user wrote the settings/loader skeleton against a
detailed supplied contract; Codex corrected two decorator typos and supplied
three missing class docstrings. Codex subsequently implemented and tested the
loader (80 passing cases). This is supported scaffolding evidence; independent
implementation, test design and review explanation remain unassessed. The test
file is now rewritten; the empty-file note above describes the earlier state.

### 2026-11-10 — Detector interview

The [detector Q&A](q-and-a.md) contains direct explanations distinguishing
architecture configuration from weights and choosing one-time model loading.
The user proposed original/resized ratio multiplication for coordinates but
did not calculate separate axis factors or the final box; Codex supplied that
worked example. These are limited conceptual observations, not evidence of
independent inference implementation or coordinate-mapping mastery. Broad
category assessments remain conservative pending scaffolding and further evidence.

Next focused practice: apply width and height ratios separately, recognize
that postprocessing already returns original-image pixels, and scaffold the
agreed reusable detector interface. Avoid treating the guided numerical answer
as an independently solved problem or repeating already answered basic
architecture-versus-weights/model-reuse questions without a new reason.

Detector scaffolding follow-up: the user asked whether running the module starts
inference and whether architecture configuration automatically handles pre/post
processing. Codex clarified class definitions versus a CLI caller and explicit
orchestration of transforms, forward pass and postprocessor. This is a concrete
practice need; no independent implementation evidence or additional mastery
assessment was inferred from the explanation.

Detector skeleton review: the user supplied the two-method scaffold and noticed
something was wrong. Codex diagnosed/fixed class-name capitalization and a
docstring/body indentation mismatch; the corrected public class imports.
This adds supported scaffolding evidence. Independent diagnosis, internal
implementation and runtime model integration remain unassessed. Reinforce Python
block indentation and exact public names during practical review.

### 2026-11-11 — Cropping interview

The [cropping Q&A](q-and-a.md) preserves all three original answers. The user
identified original image detail and discrete pixel selection, but proposed
another coordinate conversion after the detector; Codex clarified that
postprocessing already returns original pixels. Returning exact crop bounds
separately from model predictions required an explanation. The user supplied no
solution to the outward-rounding/clamping exercise; Codex provided the full
worked example. This is guided learning, not an independently completed solution.

Practice priorities are original-pixel responsibility boundaries, floor/ceil,
partial overlap versus no overlap, and exclusive bounds. Independent crop
geometry and implementation remain unassessed; broad mastery labels are not
promoted from the guided arithmetic. Next evidence should come from reviewing
the user's scaffold and discussing concrete implementation behavior.

Crop skeleton follow-up: the user supplied the agreed interface but repeated a
docstring/body indentation mismatch (five spaces versus four). Codex diagnosed
and corrected it; AST parsing then passed. This reinforces a concrete Python
practice need: consistent function-body indentation, including docstrings.
The scaffold follows a supplied contract and contains no internal crop logic;
independent implementation and numerical crop handling remain unassessed.

2026-11-11 crop verification follow-up: the user requested numbered requirement
annotations above their code blocks before tests. Codex added them and authored
91 passing crop cases plus a verified real crop. This records a useful review
preference; passing generated tests and the annotated implementation do not
establish independent user mastery. Original-pixel ownership and bounds behavior
remain appropriate subjects for a focused implementation review, without restarting
the completed three-question interview.

2026-11-11 manual-driver follow-up: repeated user-authored guesses and revisions
provide enough direct evidence for a narrowly scoped Developing conceptual
assessment in Python imports/calls/control flow. Corrections followed focused
prompts, and the complete runner was refined by Codex, so independent
implementation remains unassessed. The real wide/tall runs and passing tests
establish software behavior, not personal mastery. Do not repeat the completed
module interviews; carry these specific practice needs into the next module.
