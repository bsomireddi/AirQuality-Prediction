"""Train and evaluate one fixed Random Forest per station. No dashboard."""
import hashlib
import json
import os
from model_features import ROOT, FEATURES, TARGET, build_features, chronological_split
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.model_cache' / 'matplotlib'))
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = ROOT / 'outputs' / 'model_v1'
MODELS = ROOT / 'models' / 'v1'
PARAMETERS = dict(n_estimators=300, min_samples_leaf=3, max_features=1.0,
                  random_state=42, n_jobs=-1)


def hashes():
    paths = list((ROOT / 'DataSet').rglob('*')) + list((ROOT / 'data_processed').glob('*.csv'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def score(y, prediction):
    return {'MAE': float(mean_absolute_error(y, prediction)),
            'RMSE': float(np.sqrt(mean_squared_error(y, prediction)))}


def span(frame):
    return {'rows': len(frame), 'prediction_start': str(frame.index.min()),
            'prediction_end': str(frame.index.max()),
            'target_start': str(frame.target_timestamp.min()),
            'target_end': str(frame.target_timestamp.max())}


def main():
    before = hashes()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    report = {'features': FEATURES, 'target': TARGET, 'horizon_hours': 6,
              'parameters': PARAMETERS, 'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                  'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__, 'joblib': joblib.__version__},
              'decisions': [
                  'Independent models; no station mixing. Fixed 80/20 chronological cutoff on full input timeline.',
                  'Training labels must occur strictly before test cutoff (six-hour boundary purge).',
                  'Lags/targets use exact hourly timestamp lookups. No imputation or interpolation.',
                  'Test is rolling six-hour forecasting: current observed readings and earlier test-period observations can supply later test inputs. Models are not refitted.',
                  'Rainfall excluded: missing meaning unresolved. QC codes, provenance, station IDs, weather_matched and PM comparison flag excluded.',
                  'Other pollutants excluded to keep Version 1 focused. Station pressure selected; redundant sea-level pressure excluded.',
                  'Raw numeric wind direction retained without undocumented category conversion; circular representation deferred.',
                  'Features require observations to be available at prediction time; publication delays and revised historical readings are not represented.',
                  'Complete-case evaluation only; baseline and forest use identical test rows.',
                  'No hyperparameter search or test-driven selection. Feature importances are impurity-based, not causal, and correlated lags share importance.',
                  'Local timestamp interpretation and station-code provenance inherit cleaning-stage assumptions. PM2.5 > PM10 review flags do not remove observations.',
              ], 'stations': {}}
    for slug, name in [('gwanggyo', 'Gwanggyo-dong'), ('homaesil', 'Homaesil')]:
        data = pd.read_csv(ROOT / 'data_processed' / f'merged_{slug}.csv')
        frame = build_features(data)
        train, test, cutoff = chronological_split(frame)
        inspection = {c: {'dtype': str(data[c].dtype), 'missing': int(data[c].isna().sum()),
                          'use': 'feature' if c in FEATURES else 'excluded; see decisions'} for c in data.columns}
        inspection['timestamp']['use'] = 'exact lookup key and current hour/month only'
        (OUT / f'{slug}_feature_inspection.json').write_text(
            json.dumps({'features': FEATURES, 'input_columns': inspection,
                        'decisions': report['decisions']}, ensure_ascii=False, indent=2), encoding='utf-8')
        model = RandomForestRegressor(**PARAMETERS)
        model.fit(train[FEATURES], train[TARGET])
        predictions = model.predict(test[FEATURES])
        baseline = test.pm25.to_numpy()
        metrics = {'persistence': score(test[TARGET], baseline), 'random_forest': score(test[TARGET], predictions)}
        result = {'name': name, 'input_rows': len(data), 'inspection': inspection,
                  'cutoff': str(cutoff), 'train': span(train), 'test': span(test),
                  'missing_feature_or_target_rows': int(frame[FEATURES + [TARGET]].isna().any(axis=1).sum()),
                  'complete_rows_purged_at_boundary': int(((frame.index < cutoff) & (frame.target_timestamp >= cutoff) & frame[FEATURES + [TARGET]].notna().all(axis=1)).sum()),
                  'missing_per_feature_and_target': {c: int(frame[c].isna().sum()) for c in FEATURES + [TARGET]},
                  'metrics_ug_m3': metrics}
        report['stations'][slug] = result
        joblib.dump({'model': model, 'features': FEATURES, 'horizon_hours': 6,
                     'station': name, 'train': span(train), 'versions': report['versions']}, MODELS / f'{slug}_random_forest.joblib')
        prediction_table = pd.DataFrame({'prediction_timestamp': test.index,
            'target_timestamp': test.target_timestamp.to_numpy(), 'actual_pm25': test[TARGET].to_numpy(),
            'persistence_pm25': baseline, 'random_forest_pm25': predictions})
        prediction_table.to_csv(OUT / f'{slug}_test_predictions.csv', index=False)
        importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        importance.rename('importance').rename_axis('feature').to_csv(OUT / f'{slug}_feature_importance.csv')
        # Reindex to the full hourly target axis so missing evaluated hours show gaps.
        plot = prediction_table.set_index('target_timestamp').drop(columns='prediction_timestamp')
        plot = plot.reindex(pd.date_range(plot.index.min(), plot.index.max(), freq='h'))
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
        for ax, part, label in [(axes[0], plot, 'Full test period'),
                                 (axes[1], plot.iloc[:168], 'First seven test days')]:
            for column, display, color in [('actual_pm25', 'Actual', 'black'), ('persistence_pm25', 'Persistence', '#d18d23'),
                                            ('random_forest_pm25', 'Random Forest', '#2474b5')]:
                ax.plot(part.index, part[column], label=display, color=color, linewidth=1, alpha=0.8)
            ax.set(title=f'{name}: {label}', ylabel='PM2.5 (µg/m³)', xlabel='Target time (forecast issued 6 hours earlier)')
            ax.legend(ncol=3)
            ax.grid(alpha=.2)
        fig.savefig(OUT / f'{slug}_actual_vs_predicted.png', dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        importance.plot.barh(ax=ax, color='#2474b5')
        ax.set(title=f'{name}: Random Forest feature importance', xlabel='Impurity-based importance', ylabel='Feature')
        fig.savefig(OUT / f'{slug}_feature_importance.png', dpi=150)
        plt.close(fig)
        # Verify the saved model and predictions, not just the in-memory results.
        saved = joblib.load(MODELS / f'{slug}_random_forest.joblib')
        np.testing.assert_allclose(saved['model'].predict(test[FEATURES]), predictions)
        reread = pd.read_csv(OUT / f'{slug}_test_predictions.csv')
        np.testing.assert_allclose(reread.random_forest_pm25, predictions)
        print(name, json.dumps({k: result[k] for k in ['train', 'test', 'metrics_ug_m3']}, indent=2), flush=True)
    after = hashes()
    report['raw_and_processed_inputs_unchanged'] = before == after
    report['input_hashes_before'] = before
    report['input_hashes_after'] = after
    assert before == after, 'Input files changed'
    (OUT / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Version 1: six-hour PM2.5 forecast', '', *['- ' + x for x in report['decisions']], '',
             'All dates below refer to prediction issue times; target dates are six hours later.', '',
             '| Station | Train rows | Test rows | Training dates | Test dates | Baseline MAE | Baseline RMSE | Forest MAE | Forest RMSE |',
             '|---|---:|---:|---|---|---:|---:|---:|---:|']
    for r in report['stations'].values():
        a, b = r['train'], r['test']; p, f = r['metrics_ug_m3']['persistence'], r['metrics_ug_m3']['random_forest']
        lines.append(f"| {r['name']} | {a['rows']} | {b['rows']} | {a['prediction_start']} to {a['prediction_end']} | {b['prediction_start']} to {b['prediction_end']} | {p['MAE']:.3f} | {p['RMSE']:.3f} | {f['MAE']:.3f} | {f['RMSE']:.3f} |")
    lines += ['', '## Interpretation', '']
    for r in report['stations'].values():
        p, f = r['metrics_ug_m3']['persistence'], r['metrics_ug_m3']['random_forest']
        improvement = 100 * (p['MAE'] - f['MAE']) / p['MAE']
        lines.append(f"{r['name']}: Random Forest MAE is {f['MAE']:.2f} µg/m³. Its relative MAE improvement over persistence is {improvement:.1f}% (negative means worse).")
    lines += ['', 'MAE is the average absolute prediction error. RMSE penalizes larger errors more. Both are in µg/m³; lower is better.',
              '', 'This is one chronological holdout, not proof of year-round or operational performance. Missing observations are excluded, which can bias evaluation toward well-observed periods.',
              '', '## Input features', '', ', '.join(FEATURES), '',
              'Full column inspection, exclusions, row accounting, target date ranges, parameters and source hashes are in results.json.',
              '', f'Raw and processed inputs unchanged: {before == after}. No dashboard created.']
    (OUT / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
