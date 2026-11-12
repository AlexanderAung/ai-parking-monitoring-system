# Configuration — independent-review handoff

Created 2026-11-10; updated 2026-11-11. Configuration is implemented, tested and
used by the real detector/crop manual runner. Independent review and OCR/app
integration remain pending. See day-1-handoff.md for current final evidence.

## Responsibility and public interface

`src/parking_monitoring/config.py` reads explicit YAML, validates structure and
values, resolves paths, and returns frozen dataclasses grouped by component.

```python
def load_config(
    path: str | Path = "configs/mvp.yaml",
    *,
    project_root: str | Path | None = None,
) -> AppConfig: ...
```

AppConfig contains DetectorConfig, OCRConfig, StorageConfig and crop padding.
ConfigError(ValueError) reports invalid input/read failures. Fields match the
checked-in `configs/mvp.yaml`; the approved interface was preserved.

## Architecture and implementation decisions

- The app/CLI loads configuration and supplies each component its own settings.
  Components check their resource availability and runtime compatibility.
- The source-checkout root is the default path anchor, derived from config.py.
  An explicit project_root override supports tests/other layouts. Both relative
  YAML paths and resource paths use this root, even for externally located YAML.
- Missing/unknown fields are rejected. YAML is parsed with safe_load; scalar
  values are checked before constructing containers. Thresholds are finite in
  [0, 1], threads >= 1, padding >= 0. Booleans are not accepted as integers,
  and quoted numbers/booleans are not silently coerced.
- OCR engine is paddleocr. Device, language and model version require nonempty
  strings; their backend compatibility is the owning component's responsibility.
- Model initialization/downloads, output creation and database access are outside
  the loader. Historical existence checks were intentionally not restored.
- The user supplied the skeleton; Codex corrected decorator spelling/docstrings
  and implemented private validation/path helpers and loader orchestration.

## Tests and actual results

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider
80 passed in 0.33s; exit 0
```

Environment: Python 3.12.9, pytest 9.2.0, PyYAML 6.0.3. `git diff --check` passed.
The result above is the initial configuration-only run. Current final suite:
240 passed in 1.74s (80 configuration, 69 detector, 91 crop). Focused Ruff passes
on all three modules and their tests. `tests/test_config.py` contains 80 cases;
full-pipeline/storage tests remain placeholders. Coverage includes real MVP YAML from another working
directory, explicit roots/external YAML, all resource-path conversions, missing
assets/no project output creation, frozen settings, structure/types/ranges and
contextual errors for invalid YAML/encoding/file input. No test failures occurred.

## Limits and open work

- Successful configuration loading does not prove a checkpoint, OCR models,
  hardware device or backend option can initialize. A separate current real-model
  run verified the selected CPU configuration; it does not establish other options.
- Manual detector/crop integration uses load_config; the standalone prediction
  CLI and OCR/Streamlit/storage integration are still pending.
- Direct dataclass construction bypasses loader validation. The default root
  assumes a source checkout; other layouts must supply project_root.
- Unscoped vendored tests were not run. Current crop tests are separate evidence;
  full detection-to-OCR pipeline/storage behavior remains unverified.
- Technical Day 1 criteria are verified; external review and user sign-off remain.

## Learning evidence and review focus

See `docs/learning/q-and-a.md` for verbatim answers. The user distinguished
parsing from value-range validation and identified the intended root; typed
containers and path anchoring required guidance. Supported skeleton writing was
observed; independent implementation/test-design mastery is not established.

Please review responsibility boundaries, path semantics, validation/error
behavior, and whether the tests establish the stated contract. Return actionable
defects or trade-offs without expanding MVP scope. Independent-review findings
have not yet been received.
