"""Clean raw DataSet observations and join weather; never train or impute.

Run with Python 3 and xlrd: python src/clean_merge_data.py
Reuses the inspected file readers from validate_data.py.
"""
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import csv
import json
import re
import sys

from validate_data import DATA, ROOT, EXPECTED_CODES, blank, digest, load_file, number

PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "outputs" / "cleaning"
AIR_FIELDS = {"PM2.5": "pm25", "PM10": "pm10", "오 존": "ozone_ppm",
              "이산화질소": "no2_ppm", "일산화탄소": "co_ppm", "아황산가스": "so2_ppm"}
WEATHER_FIELDS = {"기온(°C)": "temperature_c", "습도(%)": "humidity_pct",
                  "풍속(m/s)": "wind_speed_m_s", "풍향(16방위)": "wind_direction",
                  "현지기압(hPa)": "station_pressure_hpa", "해면기압(hPa)": "sea_level_pressure_hpa",
                  "강수량(mm)": "rainfall_mm"}
QC_FIELDS = {"기온 QC플래그": "temperature_qc", "습도 QC플래그": "humidity_qc",
             "풍속 QC플래그": "wind_speed_qc", "풍향 QC플래그": "wind_direction_qc",
             "현지기압 QC플래그": "station_pressure_qc", "해면기압 QC플래그": "sea_level_pressure_qc",
             "강수량 QC플래그": "rainfall_qc"}
POLICIES = [
    "Only raw inputs inside DataSet/ are used. SHA-256 hashes are checked before and after the run.",
    "Station name and year come from AirKorea filenames. Codes 131113 and 131119 are project-supplied mappings, not verified raw-file identifiers; code provenance is retained.",
    "AirKorea hours 01-23 retain their hour; 24 becomes next-day 00:00. No flooring, rounding, interpolation or one-hour shift. Invalid/off-hour timestamps stop the run.",
    "Timestamps are timezone-naive local wall times. Exact matching assumes compatible source timezone and hourly semantics; the raw tables do not document these.",
    "Blank numeric cells become missing. Nonnumeric/nonfinite values, -999/-9999/-99999 sentinels and negative pollutant concentrations become missing with a cell-level audit.",
    "Weather checks reject negative rainfall/wind speed, humidity outside 0-100, direction outside 0-360, and nonpositive pressure. Negative temperatures remain valid.",
    "No PM2.5, PM10 or rainfall imputation. PM2.5 greater than PM10 is retained with a nullable review flag, not automatically corrected.",
    "Duplicate station/timestamp keys are checked against ALL raw data columns before numeric cleaning. Identical raw records keep the first deterministic source record; conflicting records stop the run and are reported. No averaging.",
    "KMA must identify station 119 and name 수원. Required weather fields, both pressure measures, and their raw QC flags are retained. Other weather fields remain available in the original file.",
    "QC codes are retained verbatim, including 9; no undocumented QC meanings are assumed. Blank QC cells mean unavailable flags, not proof of good quality.",
    "Wind direction retains source values and the original 16방위 label in the mapping; no categorical or degree conversion is applied.",
    "Exact timestamp left join per AirKorea station preserves unmatched air rows with missing weather and weather_matched=False. No nearest-time matching or cross-station averaging.",
    "CSV missing values are empty fields; timestamps use YYYY-MM-DD HH:MM:SS. No features, training or model outputs are produced.",
]


def manifest():
    return {str(p.relative_to(DATA)): digest(p) for p in sorted(DATA.rglob("*")) if p.is_file()}


def air_time(raw, year, month):
    match = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})", str(raw))
    if not match:
        raise ValueError(f"Invalid AirKorea timestamp: {raw!r}")
    mm, dd, hh = map(int, match.groups())
    if mm != month or not 1 <= hh <= 24:
        raise ValueError(f"Timestamp disagrees with filename/month or 01-24 hour convention: {raw}")
    return datetime(year, mm, dd) + timedelta(hours=hh)


def clean_number(raw, field, source, audit):
    if blank(raw):
        return None
    value = number(raw)
    reason = None
    if value is None:
        reason = "nonnumeric_or_nonfinite"
    elif value in (-999, -9999, -99999):
        reason = "numeric_sentinel"
    elif field in AIR_FIELDS.values() and value < 0:
        reason = "negative_pollutant"
    elif field in ("wind_speed_m_s", "rainfall_mm") and value < 0:
        reason = "negative_weather_measurement"
    elif field == "humidity_pct" and not 0 <= value <= 100:
        reason = "humidity_outside_0_100"
    elif field == "wind_direction" and not 0 <= value <= 360:
        reason = "direction_outside_0_360"
    elif field.endswith("pressure_hpa") and value <= 0:
        reason = "nonpositive_pressure"
    if reason:
        audit.append({**source, "field": field, "raw_value": raw, "reason": reason})
        return None
    return value


def deduplicate(records, label, report):
    """Do not let conversion to missing disguise conflicting raw readings."""
    seen, kept = {}, []
    details = {"input_rows": len(records), "identical_removed": [], "conflicts": []}
    report[label] = details
    for record in records:
        key = (record["station"], record["timestamp"])
        prior = seen.get(key)
        if prior is None:
            seen[key] = record
            kept.append(record)
        else:
            event = {"station": key[0], "timestamp": str(key[1]),
                     "kept_source": prior["source"], "other_source": record["source"]}
            if prior["raw"] == record["raw"]:
                details["identical_removed"].append(event)
            else:
                event["differing_columns"] = [c for c in set(prior["raw"]) | set(record["raw"])
                                               if prior["raw"].get(c) != record["raw"].get(c)]
                details["conflicts"].append(event)
    details["output_rows"] = len(kept)
    if details["conflicts"]:
        raise ValueError(f"Conflicting duplicate {label} keys; see report. No datasets written.")
    return kept


def read_records():
    air, weather = [], []
    for path in sorted(p for p in DATA.rglob("*") if p.is_file()):
        _, _, sheets = load_file(path)
        for sheet in sheets:
            matrix = sheet["matrix"]
            if not matrix:
                continue
            headers = matrix[0]
            if len(set(headers)) != len(headers):
                raise ValueError(f"Duplicate headers: {path}")
            is_air = "날짜" in headers and "PM2.5" in headers
            needed = {"날짜", *AIR_FIELDS} if is_air else {"지점", "지점명", "일시", *WEATHER_FIELDS, *QC_FIELDS}
            if not needed.issubset(headers):
                raise ValueError(f"Unexpected schema: {path}: missing {needed - set(headers)}")
            if is_air:
                match = re.fullmatch(r"Airkorea_(.+)_(\d{4})(\d{2})", path.stem)
                if not match or match[1] not in EXPECTED_CODES or matrix[1][0] != "(월-일-시)":
                    raise ValueError(f"Unexpected filename or units: {path}")
                station, year, month = match.groups()
            offset = 2 if is_air else 1
            for rownum, row in enumerate(matrix[offset:], offset + 1):
                if len(row) != len(headers):
                    raise ValueError(f"Row width mismatch: {path}:{rownum}")
                raw = dict(zip(headers, row))
                source = {"file": str(path.relative_to(DATA)), "sheet": sheet["name"], "row": rownum}
                if is_air:
                    timestamp = air_time(raw["날짜"], int(year), int(month))
                else:
                    station = str(raw["지점"])
                    if station != "119" or raw["지점명"] != "수원":
                        raise ValueError(f"Unexpected weather station: {source}")
                    timestamp = datetime.strptime(raw["일시"], "%Y-%m-%d %H:%M")
                    if timestamp.minute != 0:
                        raise ValueError(f"Non-hourly weather timestamp: {source}")
                record = {"station": station, "timestamp": timestamp, "raw": raw, "source": source}
                (air if is_air else weather).append(record)
    if not air or not weather:
        raise ValueError("Both AirKorea and KMA inputs are required")
    return air, weather


def clean_records(records, is_air, audit):
    output = []
    mapping = AIR_FIELDS if is_air else WEATHER_FIELDS
    for r in records:
        raw, source = r["raw"], r["source"]
        result = {"timestamp": r["timestamp"]}
        if is_air:
            result.update(air_station_name=r["station"], air_station_code=EXPECTED_CODES[r["station"]],
                          air_station_code_source="project_mapping_unverified")
        else:
            result.update(kma_station_number="119", kma_station_name=raw["지점명"])
        for original, field in mapping.items():
            result[field] = clean_number(raw[original], field, source, audit)
        if is_air:
            a, b = result["pm25"], result["pm10"]
            result["pm25_gt_pm10_flag"] = a > b if a is not None and b is not None else None
        else:
            for original, field in QC_FIELDS.items():
                result[field] = None if blank(raw[original]) else str(raw[original])
        prefix = "air" if is_air else "kma"
        for field, value in source.items():
            result[f"{prefix}_source_{field}"] = value
        output.append(result)
    return sorted(output, key=lambda r: (r.get("air_station_name", ""), r["timestamp"]))


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # Verify saved row counts, column order, and missing counts, not just memory.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == columns
        saved = list(reader)
    assert len(saved) == len(rows)
    for col in columns:
        assert sum(r[col] == "" for r in saved) == sum(r[col] is None for r in rows), col


def run(report):
    air_raw, weather_raw = read_records()
    air_raw = deduplicate(air_raw, "AirKorea", report["duplicates"])
    weather_raw = deduplicate(weather_raw, "KMA", report["duplicates"])
    air = clean_records(air_raw, True, report["invalid_value_changes"])
    weather = clean_records(weather_raw, False, report["invalid_value_changes"])
    lookup = {r["timestamp"]: r for r in weather}
    assert len(lookup) == len(weather)
    weather_columns = [c for c in weather[0] if c != "timestamp"]
    columns = list(air[0]) + weather_columns + ["weather_matched"]
    report["final_columns"] = columns
    datasets = [("airkorea_cleaned.csv", air, list(air[0])), ("kma_cleaned.csv", weather, list(weather[0]))]
    for station, slug in (("광교동", "gwanggyo"), ("호매실", "homaesil")):
        source_rows = [r for r in air if r["air_station_name"] == station]
        if not source_rows:
            raise ValueError(f"Missing expected station: {station}")
        merged = []
        for row in source_rows:
            matched = lookup.get(row["timestamp"])
            merged.append({**row, **{c: matched[c] if matched else None for c in weather_columns},
                           "weather_matched": matched is not None})
        times = {r["timestamp"] for r in source_rows}
        assert len(times) == len(merged)
        unmatched_air = sorted(times - lookup.keys())
        unmatched_weather = sorted(lookup.keys() - times)
        report["stations"][station] = {
            "merged_rows": len(merged), "matched_rows": sum(r["weather_matched"] for r in merged),
            "start": str(min(times)), "end": str(max(times)),
            "unmatched_air_timestamps": list(map(str, unmatched_air)),
            "unmatched_weather_timestamps": list(map(str, unmatched_weather)),
            "missing_after_merge": {c: sum(r[c] is None for r in merged) for c in columns},
            "pm25_gt_pm10_flag_true": sum(r["pm25_gt_pm10_flag"] is True for r in merged),
        }
        datasets.append((f"merged_{slug}.csv", merged, columns))
    PROCESSED.mkdir(exist_ok=True)
    report["outputs"] = []
    for name, rows, cols in datasets:
        write_csv(PROCESSED / name, rows, cols)
        report["outputs"].append({"file": f"data_processed/{name}", "rows": len(rows), "sha256": digest(PROCESSED / name)})


def main():
    report = {"status": "running", "policies": POLICIES,
              "field_mapping": {"air": AIR_FIELDS, "weather": WEATHER_FIELDS, "qc": QC_FIELDS},
              "duplicates": {}, "invalid_value_changes": [], "stations": {}, "hashes_before": manifest()}
    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        run(report)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["hashes_after"] = manifest()
        report["raw_files_unchanged"] = report["hashes_before"] == report["hashes_after"]
        if not report["raw_files_unchanged"]:
            report["status"] = "failed"
            report["error"] = "Raw input hashes changed during processing"
        (REPORTS / "cleaning_merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# Cleaning and merging report", "", f"Status: {report['status']}. Raw files unchanged: {report['raw_files_unchanged']}.",
                 "", "## Decisions", "", *[f"- {p}" for p in POLICIES], "",
                 "## Results", "", "```json", json.dumps({k: v for k, v in report.items() if k not in ("policies", "hashes_before", "hashes_after")}, ensure_ascii=False, indent=2), "```", ""]
        (REPORTS / "cleaning_merge_report.md").write_text("\n".join(lines), encoding="utf-8")
    if report["status"] != "complete":
        raise SystemExit(1)
    print(json.dumps({"status": report["status"], "raw_files_unchanged": report["raw_files_unchanged"],
                      "invalid_values_changed": len(report["invalid_value_changes"]), "duplicates": report["duplicates"],
                      "stations": report["stations"], "final_columns": report["final_columns"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
