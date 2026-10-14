# Version 1 PM2.5 forecasting

Run from the project folder:

```powershell
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" -X utf8 src/train_v1.py
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" -X utf8 -m unittest discover -s src -p "test_*.py"
```

The scripts discover the project-local `.model_deps` package folder.
In another environment install `requirements-model-v1.txt`, then use
`python src/train_v1.py`. Models are saved under `models/v1/` and results,
test predictions and graphs under `outputs/model_v1/`. Rerunning replaces
these generated outputs. Raw and processed inputs are not modified.

## Feature decisions

Each station has its own model. Inputs are observed PM2.5 now and 1, 2, 3,
6 hours earlier, PM10 now and 1, 3 hours earlier, current temperature,
humidity, wind speed, untransformed wind direction, station pressure,
hour and month. PM10 lags add recent coarse-particle context. Station
pressure is sufficient for the first version; sea-level pressure is omitted
to avoid adding a closely related second pressure variable.

Rainfall is excluded because blank rainfall values cannot yet be interpreted.
QC codes have unresolved meanings and are excluded. Station IDs are constant
within each model. Source file names and row numbers encode collection order
and are excluded, along with station-code provenance, the merge flag and the
PM comparison flag. Other pollutants are left out to keep Version 1 focused.
Timestamps provide only hour/month and exact lookup keys, not a numeric trend.

All required features must be available at the time the forecast is issued.
This historical evaluation assumes current measurements are available then;
it does not simulate publication delays or subsequent source revisions.

## Evaluation

The target is the PM2.5 observation at exactly issue time plus six hours.
Lookups use timestamps rather than shifted row positions. The latest 20%
of the full timeline defines the test period before missing rows are removed.
Training labels at or beyond the test boundary are excluded. Missing input
or target rows are excluded without filling values.

The fixed model uses 300 trees, minimum leaf size 3 and random seed 42.
No tuning uses the test data. Random Forest's internal bootstrap sampling
is part of training, not a shuffled train/test split.

The baseline and model are scored on the same complete test rows. The baseline
predicts that PM2.5 six hours ahead equals the current reading. MAE describes
typical absolute error; RMSE gives more weight to large misses. Both use µg/m³.
The test simulates forecasts issued repeatedly as new observations arrive:
earlier test observations may be inputs to later forecasts, but no future
observations are inputs and the model is not retrained during testing.

Importance plots show impurity-based importance, not causal effects. Similar
lag features can share importance. Graphs show a full holdout view and an
initial seven-day detail, with gaps where rows could not be evaluated.

This single mostly winter holdout does not establish year-round performance.
Complete-case evaluation may underrepresent sensor outages. Raw timestamp,
station-identity and QC limitations from the cleaning stage still apply.
