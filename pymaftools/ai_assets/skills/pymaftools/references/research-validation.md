# Research and validation

## Before analysis

Record:

- research question and falsifiable hypothesis;
- unit of analysis and whether samples are independent;
- inclusion, exclusion, and missing-data rules;
- genome build and annotation/database versions;
- cohort source, access date, and preprocessing provenance;
- primary outcome, covariates, confounders, and planned comparisons;
- exploratory versus confirmatory status.

Do not silently combine incompatible identifiers, callers, genome builds, or
measurement scales.

Build the analysis universe from eligibility records, not from samples that
happen to have mutation rows. Keep eligible zero-event samples. Treat missing
matrix cells as unobserved unless assay/callability evidence supports observed
absence; encode that evidence with `SampleManifest` and `ObservationMask`.

## Statistical checks

- Report effect sizes and uncertainty, not only p-values.
- Correct multiple comparisons with an explicit method.
- Validate assumptions for the chosen test.
- Keep biological absence separate from missing measurement.
- Prevent train/test and patient-level leakage in model workflows.
- Use deterministic seeds where randomness is involved.
- For TMB, record event filters, duplicate policy, genome build, and
  sample-specific callable territory; inspect the exclusion ledger.
- For feature-wise tests, declare the tested-family rule before BH correction
  and keep unadjusted Fisher results separate from covariate-adjusted claims.

## Plot checks

Render and inspect the actual artifact. Test cohort-size regimes that can change
label density, legend layout, auto-hide behavior, performance, or statistical
meaning. A bundled small fixture is not sufficient evidence for a large-cohort
figure.

## Study outputs

Produce:

- a rerunnable analysis script or notebook;
- machine-readable result tables;
- figures with explicit encodings and captions;
- software and data-version records;
- Methods, Results, and Limitations;
- a list of observations, interpretations, and unresolved hypotheses.

Have a separate validator review statistical and biological claims. Have a
separate reproducibility pass rerun the study in a clean environment.
