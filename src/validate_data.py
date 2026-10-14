"""Read-only validation of DataSet inputs. No cleaning, joining, or training.

Run: python src/validate_data.py
Requires xlrd==2.0.2 (also looks in the project-local .validation_deps folder).
Outputs only validation summaries under outputs/validation/.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".validation_deps"))
import xlrd

DATA = ROOT / "DataSet"
OUT = ROOT / "outputs" / "validation"
EXPECTED_CODES = {"광교동": "131113", "호매실": "131119"}
NOTES = [
    "Validation only: no source edits, cleaned data, merged observations, or model training.",
    "AirKorea year and station label are inferred from filenames, not observation columns. Station codes are expectations from VibeCodingPrompt.md, not verified in the raw files.",
    "For diagnostic timestamp comparison only, AirKorea hour 24 is interpreted as next-day 00:00; hours 01-23 retain their displayed hour. No one-hour shift is applied. Source timezone and interval semantics are not documented in these files.",
    "Missing counts mean empty or whitespace-only cells. Literal nonnumeric tokens and candidate numeric sentinels are reported separately; no replacements are made.",
    "Numeric parsing and timestamp parsing create diagnostic values only. Source cell types and raw timestamp examples are retained in the report.",
    "Rainfall blanks remain missing. QC codes are counted verbatim; their meanings are not inferred without a codebook.",
    "Duplicate counts mean excess occurrences beyond the first. Gap counts cover only each observed minimum-to-maximum range, not an assumed requested period.",
    "Overlap is a comparison of timestamp sets per station, not a merge. Matching timestamps alone do not establish valid complete model inputs.",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def profile(values):
    nonblank = [v for v in values if not blank(v)]
    numbers = [number(v) for v in nonblank]
    numeric = [v for v in numbers if v is not None]
    tokens = Counter(str(v) for v, n in zip(nonblank, numbers) if n is None)
    return {
        "raw_python_types": dict(Counter(type(v).__name__ for v in values)),
        "observed_type": "empty" if not nonblank else "numeric-compatible" if not tokens else "text/mixed",
        "blank": len(values) - len(nonblank),
        "numeric_count": len(numeric),
        "nonnumeric_tokens": dict(tokens),
        "minimum": min(numeric) if numeric else None,
        "maximum": max(numeric) if numeric else None,
        "negative_count": sum(v < 0 for v in numeric),
        "candidate_sentinels": dict(Counter(str(v) for v in numeric if v in (-999, -9999, -99999))),
    }


def time_profile(times):
    valid = [t for t in times if t is not None]
    counts = Counter(valid)
    unique = sorted(counts)
    gaps = []
    for left, right in zip(unique, unique[1:]):
        hours = (right - left).total_seconds() / 3600
        if hours > 1:
            gaps.append({"after": str(left), "before": str(right), "missing_hours": int(hours) - 1})
    return {
        "start": str(unique[0]) if unique else None,
        "end": str(unique[-1]) if unique else None,
        "invalid_or_missing": len(times) - len(valid),
        "unique_timestamps": len(unique),
        "duplicate_excess": sum(n - 1 for n in counts.values()),
        "duplicate_examples": {str(t): n for t, n in counts.items() if n > 1},
        "off_hour": sum(t.minute != 0 or t.second != 0 for t in valid),
        "descending_steps": sum(b < a for a, b in zip(valid, valid[1:])),
        "missing_hours_within_range": sum(g["missing_hours"] for g in gaps),
        "gaps": gaps,
    }


def load_file(path):
    raw = path.read_bytes()
    if raw.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        book = xlrd.open_workbook(file_contents=raw)
        sheets = []
        for sheet in book.sheets():
            matrix = [sheet.row_values(i) for i in range(sheet.nrows)]
            sheets.append({"name": sheet.name, "matrix": matrix, "nrows": sheet.nrows, "ncols": sheet.ncols})
        return "Excel BIFF/OLE binary workbook", "workbook encoding", sheets
    # Detect and decode strictly; never replace undecodable characters.
    for encoding in ("utf-8-sig", "cp949"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Unsupported text encoding")
    if "\t" not in decoded.splitlines()[0]:
        raise ValueError("Unrecognized file format; expected inspected tab-separated format")
    # The export uses CR CR LF, which produces empty physical lines.
    parsed = list(csv.reader(io.StringIO(decoded, newline=None), delimiter="\t"))
    matrix = [row for row in parsed if row]
    return "Tab-separated text (not Excel)", encoding, [{
        "name": "text table", "matrix": matrix, "nrows": len(matrix),
        "ncols": len(matrix[0]), "empty_physical_rows": len(parsed) - len(matrix),
    }]


def inspect_table(path, sheet):
    matrix = sheet["matrix"]
    headers = matrix[0]
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate column headers")
    air = "날짜" in headers and "PM2.5" in headers
    kma = "일시" in headers and "지점" in headers
    if not (air or kma):
        raise ValueError(f"Unrecognized nonempty table: {headers}")
    offset = 2 if air else 1
    if air and matrix[1][0] != "(월-일-시)":
        raise ValueError("Unexpected AirKorea units row")
    rows = matrix[offset:]
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Inconsistent row widths")
    columns = {name: [row[i] for row in rows] for i, name in enumerate(headers)}
    result = {
        "sheet": sheet["name"], "kind": "AirKorea" if air else "KMA",
        "rows": len(rows), "columns": headers,
        "units_row": matrix[1] if air else None,
        "column_profiles": {name: profile(values) for name, values in columns.items()},
        "exact_duplicate_rows_excess": len(rows) - len(set(tuple(row) for row in rows)),
    }
    if air:
        match = re.fullmatch(r"Airkorea_(.+)_(\d{4})(\d{2})", path.stem)
        if match is None:
            raise ValueError("Cannot infer year from filename")
        station, year, month = match.groups()
        result["station"] = {"filename_label": station, "raw_station_code": None,
                             "raw_station_name": None, "expected_unverified_code": EXPECTED_CODES.get(station)}
        result["field_mapping"] = {"timestamp": "날짜", "pm25": "PM2.5", "pm10": "PM10"}
        raw_times = columns["날짜"]
        times, bad, mismatch = [], [], []
        for index, value in enumerate(raw_times):
            try:
                m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})", str(value))
                if not m:
                    raise ValueError("Expected MM-DD-HH")
                mm, dd, hh = map(int, m.groups())
                if not 1 <= hh <= 24:
                    raise ValueError("Hour outside observed 01-24 convention")
                t = datetime(int(year), mm, dd) + timedelta(hours=hh)
                if mm != int(month):
                    mismatch.append({"source_row": index + offset + 1, "raw": value})
                times.append(t)
            except (ValueError, TypeError) as error:
                times.append(None)
                bad.append({"source_row": index + offset + 1, "raw": value, "error": str(error)})
        result["filename_month_mismatches"] = mismatch
        result["hour_24_count"] = sum(str(v).endswith("-24") for v in raw_times)
        numeric_fields = headers[1:]
        group = "AirKorea / " + station
    else:
        result["station"] = {name: dict(Counter(str(v) for v in columns[name])) for name in ("지점", "지점명")}
        result["suwon_119_confirmed"] = set(columns["지점"]) == {"119"} and set(columns["지점명"]) == {"수원"}
        result["field_mapping"] = {"station_number": "지점", "timestamp": "일시", "temperature": "기온(°C)",
                                   "humidity": "습도(%)", "wind_speed": "풍속(m/s)", "wind_direction": "풍향(16방위)",
                                   "station_pressure": "현지기압(hPa)", "sea_level_pressure": "해면기압(hPa)", "rainfall": "강수량(mm)"}
        raw_times = columns["일시"]
        times, bad = [], []
        for index, value in enumerate(raw_times):
            try:
                times.append(datetime.strptime(value, "%Y-%m-%d %H:%M"))
            except (ValueError, TypeError) as error:
                times.append(None)
                bad.append({"source_row": index + offset + 1, "raw": value, "error": str(error)})
        numeric_fields = [v for k, v in result["field_mapping"].items() if k not in ("station_number", "timestamp")]
        result["qc_values"] = {name: dict(Counter(str(v) for v in values)) for name, values in columns.items() if "QC" in name}
        group = "KMA / " + ",".join(sorted(set(columns["지점"])))
    result["raw_timestamp_first_last"] = [raw_times[0], raw_times[-1]] if raw_times else []
    result["timestamps"] = time_profile(times)
    result["timestamp_errors"] = bad
    result["numeric_field_issues"] = {}
    for field in numeric_fields:
        values = columns[field]
        p = result["column_profiles"][field]
        nums = [number(v) for v in values]
        bounds = (0, None) if air or field in ("풍속(m/s)", "강수량(mm)") else {
            "습도(%)": (0, 100), "풍향(16방위)": (0, 360),
            "현지기압(hPa)": (0, None), "해면기압(hPa)": (0, None),
        }.get(field)
        invalid = [i for i, n in enumerate(nums) if n is not None and bounds and
                   (n < bounds[0] or (bounds[1] is not None and n > bounds[1]) or ("기압" in field and n == 0))]
        result["numeric_field_issues"][field] = {
            "blank": p["blank"], "nonnumeric": p["nonnumeric_tokens"],
            "candidate_sentinels": p["candidate_sentinels"], "bounds": bounds,
            "outside_basic_bounds": len(invalid),
            "examples": [{"source_row": i + offset + 1, "raw": values[i]} for i in invalid[:10]],
        }
    if air:
        result["pm25_greater_than_pm10"] = sum(
            number(a) is not None and number(b) is not None and number(a) > number(b)
            for a, b in zip(columns["PM2.5"], columns["PM10"]))
    return result, group, times


def markdown(report):
    lines = ["# Raw data validation", "", *[f"- {note}" for note in NOTES], "",
             f"Files: {len(report['files'])}. Raw file SHA-256 hashes unchanged: {report['raw_files_unchanged']}.", "",
             "## Station coverage", "", "| Source | Rows | Start | End | Duplicate excess | Missing hours |", "|---|---:|---|---|---:|---:|"]
    for name, item in report["groups"].items():
        t = item["timestamps"]
        lines.append(f"| {name} | {item['rows']} | {t['start']} | {t['end']} | {t['duplicate_excess']} | {t['missing_hours_within_range']} |")
    lines += ["", "## Missing measurements and review flags", "",
              "| Source | Field | Blank cells | Nonnumeric tokens | Outside basic bounds |",
              "|---|---|---:|---|---:|"]
    for name, item in report["groups"].items():
        for field, stats in item["numeric_fields"].items():
            lines.append(f"| {name} | {field} | {stats['blank']} | {stats['nonnumeric']} | {stats['outside_basic_bounds']} |")
    lines += ["", "PM2.5 greater than PM10 is a review flag, not an automatic invalidation:"]
    for name, item in report["groups"].items():
        if name.startswith("AirKorea"):
            lines.append(f"- {name}: {item['pm25_greater_than_pm10']} observations.")
    lines += ["", "KMA wind direction is labeled 풍향(16방위), but observed numeric values span 0–360. Do not assume categories 1–16. Rainfall and QC conventions need review before cleaning.",
              "", "## File inventory", "", "| File | Table | Data rows | Start | End | Duplicate excess |", "|---|---|---:|---|---|---:|"]
    for file in report["files"]:
        for table in file.get("tables", []):
            t = table["timestamps"]
            lines.append(f"| {file['file']} | {table['sheet']} | {table['rows']} | {t['start']} | {t['end']} | {t['duplicate_excess']} |")
    lines += ["", "## Potential timestamp overlap (no merge performed)", "", "```json", json.dumps(report["overlap"], ensure_ascii=False, indent=2), "```", "", "## Per-file findings"]
    for file in report["files"]:
        lines += ["", f"### {file['file']}", "", f"Format: {file.get('format')}; encoding: {file.get('encoding')}; sheets: {file.get('sheet_inventory')}"]
        if "error" in file:
            lines.append(f"ERROR: {file['error']}")
        for table in file.get("tables", []):
            lines += ["", f"Sheet: {table['sheet']}; data rows: {table['rows']}; station: {json.dumps(table['station'], ensure_ascii=False)}",
                      "", f"Field mapping: {json.dumps(table['field_mapping'], ensure_ascii=False)}",
                      "", f"Raw timestamp endpoints: {table['raw_timestamp_first_last']}",
                      "", f"Diagnostic timestamp results: {json.dumps(table['timestamps'], ensure_ascii=False)}",
                      "", "| Column | Raw types / inferred type | Blank | Numeric min | Numeric max | Negative | Candidate sentinels | Nonnumeric tokens (except timestamp/name) |",
                      "|---|---|---:|---:|---:|---:|---|---|"]
            for col, p in table["column_profiles"].items():
                tokens = p["nonnumeric_tokens"] if col not in ("날짜", "일시", "지점명") else "see field mapping"
                lines.append(f"| {col} | {p['raw_python_types']} / {p['observed_type']} | {p['blank']} | {p['minimum']} | {p['maximum']} | {p['negative_count']} | {p['candidate_sentinels']} | {tokens} |")
            lines += ["", "Numeric validation:", "```json", json.dumps(table["numeric_field_issues"], ensure_ascii=False, indent=2), "```"]
            for key in ("units_row", "qc_values", "pm25_greater_than_pm10", "hour_24_count", "filename_month_mismatches", "timestamp_errors", "exact_duplicate_rows_excess", "suwon_119_confirmed"):
                if key in table:
                    lines += ["", f"{key}: {json.dumps(table[key], ensure_ascii=False)}"]
    return "\n".join(lines) + "\n"


def main():
    paths = sorted(p for p in DATA.rglob("*") if p.is_file())
    if not paths:
        raise SystemExit(f"No raw files in {DATA}")
    before = {str(p.relative_to(DATA)): digest(p) for p in paths}
    report = {"notes": NOTES, "files": [], "groups": {}, "overlap": [], "hashes_before": before}
    group_times = defaultdict(list)
    group_tables = defaultdict(list)
    for path in paths:
        file = {"file": str(path.relative_to(DATA)), "extension": path.suffix, "bytes": path.stat().st_size, "tables": []}
        report["files"].append(file)
        try:
            kind, encoding, sheets = load_file(path)
            file.update(format=kind, encoding=encoding, sheet_inventory=[{k: v for k, v in s.items() if k != "matrix"} for s in sheets])
            for sheet in sheets:
                if not sheet["matrix"]:
                    continue
                result, group, times = inspect_table(path, sheet)
                file["tables"].append(result)
                group_times[group].extend(times)
                group_tables[group].append(result)
        except Exception as error:
            file["error"] = f"{type(error).__name__}: {error}"
    for group, times in group_times.items():
        tables = group_tables[group]
        fields = tables[0]["numeric_field_issues"]
        report["groups"][group] = {
            "rows": len(times), "timestamps": time_profile(times),
            "numeric_fields": {field: {
                "blank": sum(t["numeric_field_issues"][field]["blank"] for t in tables),
                "nonnumeric": dict(sum((Counter(t["numeric_field_issues"][field]["nonnumeric"]) for t in tables), Counter())),
                "outside_basic_bounds": sum(t["numeric_field_issues"][field]["outside_basic_bounds"] for t in tables),
            } for field in fields},
            "pm25_greater_than_pm10": sum(t.get("pm25_greater_than_pm10", 0) for t in tables),
        }
    for air, air_times in group_times.items():
        if not air.startswith("AirKorea"):
            continue
        a = set(air_times) - {None}
        for weather, weather_times in group_times.items():
            if not weather.startswith("KMA"):
                continue
            w = set(weather_times) - {None}
            report["overlap"].append({"air_station": air, "weather_station": weather,
                "matching_unique_hours": len(a & w), "air_only_hours": len(a - w), "weather_only_hours": len(w - a),
                "air_only_examples": [str(t) for t in sorted(a - w)[:10]],
                "weather_only_examples": [str(t) for t in sorted(w - a)[:10]],
                "unique_join_keys": len(a) == len(air_times) and len(w) == len(weather_times)})
    after = {str(p.relative_to(DATA)): digest(p) for p in sorted(DATA.rglob("*")) if p.is_file()}
    report["hashes_after"] = after
    report["raw_files_unchanged"] = before == after
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "validation_report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("raw_files_unchanged", "groups", "overlap")}, ensure_ascii=False, indent=2))
    errors = [f for f in report["files"] if "error" in f]
    print(f"Inspected {len(paths)} files; {len(errors)} file errors. Reports: {OUT}")
    if errors or not report["raw_files_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
