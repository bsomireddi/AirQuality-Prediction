"""Checks for cleaning edge cases absent from the current raw data."""
from datetime import datetime
import unittest

from clean_merge_data import air_time, clean_number, deduplicate


class CleaningChecks(unittest.TestCase):
    def test_midnight_rollover_and_invalid_time(self):
        self.assertEqual(air_time("12-31-24", 2025, 12), datetime(2026, 1, 1))
        for raw in ("02-01-00", "02-01-25", "03-01-01", "02-30-01"):
            with self.assertRaises(ValueError):
                air_time(raw, 2025, 2)

    def test_missing_sentinels_and_valid_zero(self):
        audit = []
        for value in ("-999", "-9999", "-1", "bad", "inf"):
            self.assertIsNone(clean_number(value, "pm25", {}, audit))
        self.assertEqual(len(audit), 5)
        self.assertIsNone(clean_number("", "pm10", {}, audit))
        self.assertEqual(clean_number("0", "pm10", {}, audit), 0)
        self.assertEqual(clean_number("-12.4", "temperature_c", {}, audit), -12.4)
        self.assertIsNone(clean_number("", "rainfall_mm", {}, audit))
        self.assertIsNone(clean_number("101", "humidity_pct", {}, audit))

    def test_duplicate_identity_and_conflicts(self):
        a = {"station": "A", "timestamp": datetime(2025, 1, 1),
             "raw": {"pm25": "10"}, "source": {"file": "first"}}
        b = {**a, "source": {"file": "second"}}
        report = {}
        self.assertEqual(len(deduplicate([a, b], "air", report)), 1)
        self.assertEqual(len(report["air"]["identical_removed"]), 1)
        # Same hour at a separate station is not a duplicate.
        self.assertEqual(len(deduplicate([a, {**b, "station": "B"}], "air", {})), 2)
        with self.assertRaises(ValueError):
            deduplicate([a, {**b, "raw": {"pm25": "20"}}], "air", {})


if __name__ == "__main__":
    unittest.main()
