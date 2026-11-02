# Detection error-analysis template

This is a planning template, not an experiment report. Replace placeholders only with observed results and link their supporting evidence. Leave unperformed analyses explicitly marked “Not run.” The completed baseline evidence is in the [baseline report](../reports/baseline-001/report.md).

## Experiment identity

- Author/date: [fill in]
- Status: [planned / in progress / complete]
- Model, checkpoint hash, and code revision: [fill in]
- Dataset version, split, image/annotation counts, and hashes: [fill in]
- Evaluation config and environment: [link]

## Objective

[State the question, intended application, and decisions this analysis will inform.]

## Baseline performance

| Metric | Overall | Small | Medium | Large |
|---|---|---|---|---|
| AP50 | [measured] | [measured] | [measured] | [measured] |
| AP75 | [measured] | [measured] | [measured] | [measured] |
| AP50:95 | [measured] | [measured] | [measured] | [measured] |

Recall: [value, IoU range, detection limit, and source]. Define size groups using the actual evaluator settings; do not mix annotation area with plate-height or relative-area groups.

## Evaluation sanity checks

- [ ] Checkpoint and class mapping verified.
- [ ] Preprocessing, box coordinates, and original-image scaling verified.
- [ ] Annotation quality inspected; scope and limitations recorded.
- [ ] IoU/area ranges, detection limits, and confidence filtering recorded.

[Link evidence for each completed check and describe any fixes.]

## Quantitative error analysis

Status: Not run.

[If performed, record the tool/version, command, inputs, error definitions, measured results, and output artifacts. Do not claim TIDE results without running it. Oracle improvements are counterfactual and must not be added together or presented as guaranteed gains.]

## Qualitative review

Status: Not run.

[Record sample selection, reviewed image IDs, sample size, visualization threshold, and observations. Distinguish model errors, annotation defects, and application-domain differences. Do not infer prevalence from hand-picked examples.]

| Image | Observation | Evidence | Interpretation/uncertainty |
|---|---|---|---|
| [filename] | [observed behavior] | [comparison link] | [supported interpretation] |

## Hypotheses and controlled experiments

| Hypothesis | Evidence so far | Proposed test | Status |
|---|---|---|---|
| [hypothesis] | [evidence or unknown] | [controlled comparison] | Not run |

For each executed experiment record fixed inputs, changed variables, commands, metrics, hardware/timing method if relevant, and artifact links. Separate proposed tests from measured outcomes.

## Findings, limitations, and next action

- Supported findings: [fill in]
- Unresolved questions and sampling limitations: [fill in]
- Recommended next action and supporting evidence: [fill in]

## Supporting artifacts

[Link only artifacts that exist. Mark external inputs and access requirements explicitly.]
