# AGENTS.md

## Scope

These instructions apply to the entire `abdulbasit742/prostate-cadx` repository. More specific AGENTS.md or AGENTS.override.md files in subdirectories may refine them.

Project: **prostate-cadx**.

Detected root stack: **Python project configured by pyproject.toml**.

## Working method

1. Read README.md, the relevant manifests, and nearby tests before editing.
2. Check the current diff and preserve unrelated user changes.
3. Make the smallest coherent change that solves the task; follow existing names, patterns, and directory boundaries.
4. Do not hand-edit generated, vendored, dependency, build-output, model-weight, or dataset files unless the task explicitly targets them.
5. Update tests and documentation when behavior, configuration, public APIs, or setup steps change.

## Commands

- Create an isolated environment: `python -m venv .venv`.
- Install the project with `python -m pip install -e .` unless pyproject.toml documents a different workflow.
- Run tests with `python -m pytest`.
- Treat pyproject.toml and the README as the source of truth for tool-specific commands.

## Verification

- Run the narrowest relevant test first, then the repository's available lint, type-check, test, and build commands.
- Never report a check as passed unless it was actually run. State skipped checks and the concrete reason.
- For UI changes, verify loading, empty, error, and success states plus keyboard access and responsive layout.
- For API or persistence changes, verify validation, authorization, failure behavior, and backward compatibility.

## Security and side effects

- Never commit secrets, tokens, passwords, private keys, production data, or populated environment files. Use documented environment variables and sanitized examples.
- Treat migrations, deployments, billing, live network calls, account changes, destructive Git operations, and external messages as side effects. Do not perform them without explicit task authorization.
- Validate untrusted input at trust boundaries and avoid logging credentials, personal data, prompts containing secrets, or raw third-party payloads.
- Health/medical scope: preserve research-only and non-diagnostic wording, avoid unsupported clinical claims, protect patient data, and require domain-expert review for changes affecting predictions or advice.
- Scientific/data work: preserve reproducibility. Record seeds and parameters, avoid committing large generated datasets/models, and distinguish measured results from assumptions.

## Completion checklist

- The requested behavior is implemented with a focused diff.
- Relevant automated checks pass, or any unavailable checks are clearly identified.
- No secrets, generated artifacts, or unrelated formatting churn were introduced.
- The final handoff summarizes changed files, verification evidence, risks, and any follow-up work.
