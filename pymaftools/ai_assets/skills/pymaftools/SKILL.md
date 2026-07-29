---
name: pymaftools
description: Use when analyzing MAF or harmonized VCF data with pymaftools, building mutation or multi-omics tables, creating oncoplots and cancer-genomics figures, reading TCGA/GDC exports, conducting example bioinformatics studies, or contributing code and documentation to the pymaftools repository.
---

# pymaftools

Use the installed package and current source as the API authority. Do not rely on
remembered signatures when source or generated API documentation is available.

## Route the task

- For package architecture or contributions, read
  [references/architecture.md](references/architecture.md).
- For analysis APIs and common workflows, read
  [references/api-workflows.md](references/api-workflows.md).
- For an example study or scientific interpretation, also read
  [references/research-validation.md](references/research-validation.md).

## Work safely

1. Identify the input type, sample identifiers, reference/annotation version,
   cohort definition, and desired output.
2. Inspect the relevant class or function before writing code.
3. Preserve alignment between the feature-by-sample matrix,
   `feature_metadata`, and `sample_metadata`.
4. Keep statistical computation separate from visualization.
5. Use the smallest public workflow that answers the question.
6. Save reusable tables and figures, and record versions and filtering choices.
7. Validate behavior with focused tests or a rerunnable example.

## Repository commands

Run project commands from the repository root:

```bash
uv run pytest -q
uv run ruff check pymaftools/
uv run ruff format --check pymaftools/
```

Integration tests are deselected by default. Do not require live network access
for unit tests.

## Boundaries

- Treat external VCF normalization, left alignment, reference checking, and
  multi-allelic splitting as upstream pipeline responsibilities.
- Prefer HDF5 for new high-dimensional `PivotTable` and `Cohort` persistence.
- Render plots before saving them.
- Do not interpret a visually plausible plot as evidence that its scientific
  meaning is correct.
- Keep package changes separate from example-study outputs.
