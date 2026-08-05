You package the selected model so someone who has never seen this project can
score new data with it.

## What to do

1. Call `export_bundle`. It copies the selected model, the fitted preprocessor
   state and the exact preprocessing code, then generates `predict.py`, a
   `pyproject.toml` with only the runtime dependencies, and a README.
2. Read `bundle/README.md` to confirm it says what you expect.

## What to report back

- the exact command the user runs to score a new file
- the raw input columns their file must contain
- what `predict.py` returns (a `prediction` column, plus one `proba_<class>`
  column per class for classification)
- the reminder that these metrics come from a held-out split of the *same*
  dataset, so the model should be re-scored on labelled rows before being reused
  on a different data source

Keep it practical. This is the last thing the user reads before trying to use the
model.
