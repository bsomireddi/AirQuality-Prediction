# Data inspection only

Run from the project folder with Python 3 and `xlrd` installed:

```powershell
python src/validate_data.py
```

In this workspace, Python is available at the following path and `xlrd` was
installed into `.validation_deps/` (the script finds that folder automatically):

```powershell
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" -X utf8 src/validate_data.py
```

For a separate Python environment, install the single dependency first:

```powershell
python -m pip install -r requirements-validation.txt
```

The script reads all files recursively inside `DataSet/` and writes
`outputs/validation/validation_report.md` and `validation_report.json`.
The Markdown report includes the file inventory, per-file schemas, source
types, inferred numeric types, missing cells, numeric ranges, suspicious
values, station identity evidence, timestamps and overlap counts.
The JSON report also records source hashes before and after inspection.

Blank physical lines in the KMA text export and AirKorea column/unit headers
are structural metadata, not observation rows. All nonempty workbook sheets
are inspected, and empty sheets are inventoried. Unexpected formats or read
errors are reported and cause a nonzero exit status. Data-quality findings
are reported without altering or removing observations.

AirKorea station names and years come from filenames. The raw tables do not
verify station codes. Diagnostic parsing interprets hour `24` as the next
day's midnight. Timestamp overlap is conditional on this interpretation and
the sources sharing a timezone and compatible hourly measurement semantics.
There is no source timezone metadata or QC codebook in these tables.

No cleaning, imputation, merging, feature generation or training is performed.
No data is read from experimental datasets outside `DataSet/`.
