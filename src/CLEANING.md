# Cleaning and exact hourly merging

Run from the project folder:

```powershell
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" -X utf8 src/clean_merge_data.py
```

Alternatively use `python src/clean_merge_data.py` with Python 3 and the
dependency in `requirements-validation.txt`. No additional package is needed.

Outputs are `data_processed/airkorea_cleaned.csv`, `kma_cleaned.csv`,
`merged_gwanggyo.csv`, and `merged_homaesil.csv`. Rerunning replaces these
generated outputs. Raw files in `DataSet/` are read only.

The full decisions, source-to-output column mapping, duplicate audit, invalid
value audit, unmatched timestamps, per-column missing counts, output hashes
and raw-file integrity check are recorded in
`outputs/cleaning/cleaning_merge_report.md` and `.json`.

The merged files use an exact timestamp left join: one row per air station
and hour, preserving air observations without matching weather. Check
`weather_matched` before using weather-dependent rows. Blank CSV fields mean
missing values. Station codes are supplied by project context and explicitly
marked unverified in `air_station_code_source`. The `pm25_gt_pm10_flag` is
blank when either measurement is missing.

Timestamp assumptions and QC limitations are listed in the report. No
imputation, feature engineering, train/test splitting or training is performed.

Run the edge-case checks with the same Python executable:

```powershell
python -m unittest discover -s src -p test_clean_merge_data.py
```
