"""Read existing artifacts and predict once; no training or data writes."""
import json
import math
from model_features import ROOT, FEATURES, build_features
import joblib
import pandas as pd

STATIONS = {'Gwanggyo-dong (광교동)': 'gwanggyo', 'Homaesil (호매실)': 'homaesil'}


def risk_category(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('Prediction must be a finite nonnegative concentration')
    # Continuous thresholds extend the supplied integer display bands without gaps.
    if value <= 15:
        return 'Good'
    if value <= 35:
        return 'Moderate'
    if value <= 75:
        return 'Bad'
    return 'Very Bad'


ALERTS = {
    'Good': 'Air quality is predicted to remain good.',
    'Moderate': 'PM2.5 is predicted to reach a moderate level.',
    'Bad': 'PM2.5 is predicted to reach a bad level within the next 6 hours. Consider reducing prolonged outdoor activity.',
    'Very Bad': 'Very high PM2.5 is predicted within the next 6 hours. Consider avoiding prolonged outdoor activity.',
}


def load_station(slug):
    if slug not in STATIONS.values():
        raise ValueError('Unknown station')
    data = pd.read_csv(ROOT / 'data_processed' / f'merged_{slug}.csv')
    features = build_features(data)
    # Inference requires inputs only: a future actual target need not exist.
    complete = features.dropna(subset=FEATURES)
    if complete.empty:
        raise ValueError('No observation has all required model inputs')
    latest = complete.iloc[[-1]]
    bundle = joblib.load(ROOT / 'models' / 'v1' / f'{slug}_random_forest.joblib')
    if bundle['features'] != FEATURES or bundle['horizon_hours'] != 6:
        raise ValueError('Saved model feature schema or horizon differs from Version 1')
    expected_name = {'gwanggyo': 'Gwanggyo-dong', 'homaesil': 'Homaesil'}[slug]
    if bundle['station'] != expected_name:
        raise ValueError('Saved model belongs to another station')
    predicted = float(bundle['model'].predict(latest[FEATURES])[0])
    results = json.loads((ROOT / 'outputs' / 'model_v1' / 'results.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(ROOT / 'outputs' / 'model_v1' / f'{slug}_test_predictions.csv',
                              parse_dates=['prediction_timestamp', 'target_timestamp'])
    return {'row': latest.iloc[0], 'observation_time': latest.index[0],
            'prediction_time': latest.index[0] + pd.Timedelta(hours=6),
            'latest_raw_time': features.index.max(), 'prediction': predicted,
            'category': risk_category(predicted), 'results': results['stations'][slug],
            'test_predictions': predictions,
            'importance_image': ROOT / 'outputs' / 'model_v1' / f'{slug}_feature_importance.png'}
