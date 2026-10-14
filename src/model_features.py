"""Version 1 feature construction with exact timestamp lookups."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
try:
    _init = ROOT / '.model_deps' / 'joblib' / '__init__.py'
    if _init.is_file():
        with open(_init, 'rb') as _f:
            _f.read(1)
        sys.path.insert(0, str(ROOT / '.model_deps'))
except (PermissionError, OSError):
    pass
import numpy as np
import pandas as pd

FEATURES = ['pm25', 'pm25_lag_1h', 'pm25_lag_2h', 'pm25_lag_3h',
            'pm25_lag_6h', 'pm10', 'pm10_lag_1h', 'pm10_lag_3h',
            'temperature_c', 'humidity_pct', 'wind_speed_m_s', 'wind_direction',
            'station_pressure_hpa', 'hour', 'month']
TARGET = 'target_pm25_6h'


def build_features(data):
    if data.air_station_name.nunique() != 1:
        raise ValueError('Build features separately for each station')
    data = data.copy()
    data['timestamp'] = pd.to_datetime(data.timestamp, errors='raise')
    if data.timestamp.isna().any() or data.timestamp.duplicated().any():
        raise ValueError('Missing or duplicate timestamp')
    if not data.timestamp.eq(data.timestamp.dt.floor('h')).all():
        raise ValueError('Non-hourly timestamps')
    data = data.sort_values('timestamp').set_index('timestamp')
    frame = data[['pm25', 'pm10', 'temperature_c', 'humidity_pct',
                  'wind_speed_m_s', 'wind_direction', 'station_pressure_hpa']].copy()
    # Reindex by clock time, never by row position: safe even with gaps.
    for col, lags in [('pm25', [1, 2, 3, 6]), ('pm10', [1, 3])]:
        for lag in lags:
            frame[f'{col}_lag_{lag}h'] = data[col].reindex(
                frame.index - pd.Timedelta(hours=lag)).to_numpy()
    frame[TARGET] = data.pm25.reindex(frame.index + pd.Timedelta(hours=6)).to_numpy()
    frame['target_timestamp'] = frame.index + pd.Timedelta(hours=6)
    frame['hour'] = frame.index.hour
    frame['month'] = frame.index.month
    if np.isinf(frame[FEATURES + [TARGET]].to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature or target values')
    return frame


def chronological_split(frame):
    # Choose cutoff on the full timeline before excluding incomplete rows.
    cutoff = frame.index[int(len(frame) * 0.8)]
    usable = frame.dropna(subset=FEATURES + [TARGET])
    # A training label must be observed strictly before test prediction starts.
    train = usable[(usable.index < cutoff) & (usable.target_timestamp < cutoff)]
    test = usable[usable.index >= cutoff]
    if train.empty or test.empty:
        raise ValueError('Insufficient complete training or test observations')
    assert train.target_timestamp.max() < test.index.min()
    return train, test, cutoff
