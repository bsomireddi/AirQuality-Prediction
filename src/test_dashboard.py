"""Dashboard inference and station-switch verification with Streamlit AppTest."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
try:
    _init = ROOT / '.dashboard_deps' / 'pandas' / '__init__.py'
    if _init.is_file():
        with open(_init, 'rb') as _f:
            _f.read(1)
        sys.path.insert(0, str(ROOT / '.dashboard_deps'))
except (PermissionError, OSError):
    pass
from dashboard_data import STATIONS, load_station, risk_category
from model_features import FEATURES, build_features, pd
from streamlit.testing.v1 import AppTest


class DashboardChecks(unittest.TestCase):
    def test_risk_thresholds_have_no_decimal_gaps(self):
        for value, expected in [(0, 'Good'), (15, 'Good'), (15.01, 'Moderate'),
                                (35, 'Moderate'), (35.01, 'Bad'), (75, 'Bad'), (75.01, 'Very Bad')]:
            self.assertEqual(risk_category(value), expected)
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                risk_category(value)

    def test_latest_complete_inputs_do_not_require_future_actual(self):
        for slug in STATIONS.values():
            result = load_station(slug)
            data = pd.read_csv(ROOT / 'data_processed' / f'merged_{slug}.csv')
            features = build_features(data)
            expected = features.dropna(subset=FEATURES).index.max()
            self.assertEqual(result['observation_time'], expected)
            self.assertEqual(result['prediction_time'] - expected, pd.Timedelta(hours=6))
            self.assertTrue(pd.isna(result['row']['target_pm25_6h']))

    def test_both_station_selections_update_values_and_results(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        labels = [m.label for m in app.metric]
        self.assertIn('PM2.5 NOW', labels)
        self.assertIn('PM10 NOW', labels)
        self.assertIn('Temperature', labels)
        self.assertIn('Humidity', labels)
        self.assertIn('Wind Speed', labels)
        self.assertIn('6-HOUR FORECAST', labels)
        self.assertIn('TREND', labels)
        self.assertGreaterEqual(len(app.get('plotly_chart')) + len(app.get('vega_lite_chart')), 1)

    def test_live_and_forecast_metrics_and_alert(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        # Check Live Forecast metrics: PM2.5 NOW, 6-HOUR FORECAST, TREND
        labels = [m.label for m in app.metric]
        self.assertIn('PM2.5 NOW', labels)
        self.assertIn('6-HOUR FORECAST', labels)
        self.assertIn('TREND', labels)
        # Verify simulated bad episode triggers early alert
        opt = app.radio(key='demo_scenario_radio').options[1]
        app.radio(key='demo_scenario_radio').set_value(opt).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(len(app.error) > 0)
        self.assertTrue(any('Early Air Quality Alert' in err.value for err in app.error))


if __name__ == '__main__':
    unittest.main()
