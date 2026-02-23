# AGENT.md

Guidelines for LLM/code agents working in this repository.

## Principles

1. Keep changes minimal, clear, and correct.
2. Prefer deletion over addition when complexity can be reduced.
3. Every line should justify its existence.
4. Avoid speculative abstractions.
5. Preserve existing behavior unless the task explicitly changes it.

## Workflow

1. Read nearby code before editing.
2. Make the smallest patch that solves the request.
3. Run targeted checks/tests for changed paths.
4. Update docs when user-facing behavior changes.
5. Keep commit messages short and explicit.

## Style

- Favor simple functions, explicit names, and direct control flow.
- Avoid dead code, broad comments, and noisy logging.
- Match existing conventions in touched files.
- Do not add dependencies unless clearly necessary.

## Validation

- For Python edits, run at least one fast sanity check (for example `python -m compileall src`).
- For behavior changes, run the closest relevant test(s).
- If a check cannot run, state why.

## Documentation

- README should stay concise and example-driven.
- Prefer practical command examples over long prose.
- Keep tables and sections scannable.
