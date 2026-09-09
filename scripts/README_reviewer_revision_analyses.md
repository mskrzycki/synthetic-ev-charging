# Reviewer Revision Analysis Scripts

These scripts reproduce the additional tables and diagnostics added during the reviewer-response revisions.

- `evaluation_tables.py`: main yearly metrics, location robustness, Frobenius dependency tables, pairwise 2D similarity, aggregate load-profile metrics, and load-profile reconstruction sensitivity.
- `generate_location_group_construction_table.py`: construction table for the four residential location groups used as model conditioning categories.
- `generate_monthly_case_study_metrics.py`: revised monthly case-study table for January, May, July, and October.
- `generate_physical_constraint_table.py`: session-level physical-constraint and implied-average-power screening table.
- `generate_preprocessing_exclusion_counts.py`: sequential preprocessing-exclusion counts for the final modelling dataset.
- `generate_tail_bin_tables.py`: upper-tail bin frequency and coverage tables, plus the tail-bin figures.
- `generate_privacy_memorization_diagnostic.py`: nearest-neighbor and rounded-duplicate memorization diagnostic.
- `measure_computational_cost.py`: hardware, training-time, and generation-time benchmark for Diffusion, CTGAN, and TabDDPM.
- `revision_analysis_common.py`: shared data-loading, filtering, formatting, and output helpers used by the revision scripts.

Run the scripts from the repository root, for example:

```bash
python3 scripts/generate_monthly_case_study_metrics.py
python3 scripts/generate_location_group_construction_table.py
python3 scripts/generate_preprocessing_exclusion_counts.py
python3 scripts/generate_physical_constraint_table.py
python3 scripts/generate_tail_bin_tables.py
python3 scripts/generate_privacy_memorization_diagnostic.py
python3 scripts/measure_computational_cost.py --models diffusion ctgan tabddpm --generation-scope quick
```

For the full computational-cost benchmark on a server, edit the environment setup in:

```bash
run_computational_cost_benchmark.slurm
```

and submit it with:

```bash
sbatch run_computational_cost_benchmark.slurm
```
