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
