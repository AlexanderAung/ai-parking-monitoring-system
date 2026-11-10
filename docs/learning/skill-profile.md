# Engineering Skill Profile

Initial evidence review: 2026-11-09, repository commit `6e5dd82`. This is a working assessment, not a permanent judgment of ability.

Repository artifacts show project experience, but their existence and Git authorship do not establish how independently the user designed, implemented or understood them. No original interview answers or observed user scaffolding were available for this assessment. Consequently all personal assessments start as **Not assessed**, rather than inferring mastery or lack of ability. Conceptual understanding and practical implementation ability both remain unassessed unless stated otherwise.

| Category | Current assessment | Evidence | Next development area |
| --- | --- | --- | --- |
| Python fundamentals | Not assessed | [Result dataclasses](../../src/parking_monitoring/schemas.py) and [data scripts](../../scripts/data/README.md) exist; individual understanding/implementation independence is unverified. | Explain type hints, validation and exceptions; write a small typed module skeleton. |
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

No assessment changes yet. Initial baseline is unassessed pending direct evidence.
