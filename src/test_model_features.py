from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import unittest
from model_features import pd, np, FEATURES, TARGET, build_features, chronological_split


class ForecastChecks(unittest.TestCase):
    def sample(self):
        times = pd.date_range('2025-01-01', periods=100, freq='h')
        d = pd.DataFrame({'timestamp': times, 'air_station_name': 'A'})
        for c in ['pm25', 'pm10', 'temperature_c', 'humidity_pct', 'wind_speed_m_s', 'wind_direction', 'station_pressure_hpa']:
            d[c] = np.arange(100, dtype=float)
        return d

    def test_exact_times_across_gap(self):
        d = self.sample().drop(index=10)
        f = build_features(d)
        self.assertTrue(pd.isna(f.loc['2025-01-01 11:00', 'pm25_lag_1h']))
        self.assertEqual(f.loc['2025-01-01 09:00', TARGET], 15)
        self.assertTrue(pd.isna(f.loc['2025-01-01 04:00', TARGET]))

    def test_future_changes_do_not_affect_features(self):
        d = self.sample()
        before = build_features(d)
        d.loc[d.index > 40, 'pm25'] = 9999
        after = build_features(d)
        pd.testing.assert_frame_equal(before[FEATURES].iloc[:41], after[FEATURES].iloc[:41])
        self.assertNotEqual(before[TARGET].iloc[40], after[TARGET].iloc[40])

    def test_split_purges_future_labels_and_preserves_missing(self):
        d = self.sample()
        d.loc[20, 'pm25'] = np.nan
        f = build_features(d)
        train, test, cutoff = chronological_split(f)
        self.assertLess(train.target_timestamp.max(), cutoff)
        self.assertGreaterEqual(test.index.min(), cutoff)
        self.assertTrue(pd.isna(f.iloc[20].pm25))
        self.assertNotIn(f.index[20], train.index)
        self.assertEqual(train.index.max(), f.index[73])

    def test_station_mixing_and_duplicates_rejected(self):
        d = self.sample()
        with self.assertRaises(ValueError):
            build_features(pd.concat([d, d.iloc[:1]]))
        d.loc[1, 'air_station_name'] = 'B'
        with self.assertRaises(ValueError):
            build_features(d)


if __name__ == '__main__':
    unittest.main()
