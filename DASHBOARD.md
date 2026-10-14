# Professor demonstration

From the project folder, run in PowerShell:

```powershell
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" run_dashboard.py
```

Open http://127.0.0.1:8501. Keep the terminal running. Ctrl+C stops a
foreground launch. The launcher uses the project-local `.dashboard_deps`
and `.model_deps` folders and binds only to the local computer.

For another environment, install `requirements-dashboard.txt`, then use
`python run_dashboard.py`. Streamlit requires pandas below version 3, so
the dashboard uses a local pandas 2.3.3 dependency without replacing the
shared training runtime. Existing models and evaluation files are reused.

## What to demonstrate

1. Select Gwanggyo-dong. Point to the historical observation and forecast times.
2. Compare current PM2.5, its six-hour prediction, the category and alert.
3. Switch to Homaesil. The observation, prediction, chart and performance change.
4. Show the recent test graph and the baseline-versus-model error table.
5. Expand Feature importance, then explain the pipeline and AI-assisted process.
6. Switch to the **⚡ Live & 6-Hour Forecast** tab:
   - Check the **PM2.5 NOW** card, **6-HOUR FORECAST**, and **TREND** indicator (`↑ Worsening` / `↓ Improving`).
   - Observe the **Predicted PM2.5 Trend** line chart (`Now`, `+1h`, `+2h`, `+3h`, `+4h`, `+5h`, `+6h`) with the 36 µg/m³ Bad threshold line.
   - Open the "Scenario & Demonstration Controls" to demonstrate the **Early Air Quality Alert** banner triggering when PM2.5 reaches or exceeds Bad levels (≥ 36 µg/m³).

The model uses the latest row with all required current and lag inputs.
It does not require a known future target. Gwanggyo uses 2026-02-28 21:00;
Homaesil uses 2026-02-28 23:00. Later incomplete rows are not filled.
Predictions refer to 2026-03-01 03:00 and 05:00 respectively.
The graph and error table use saved historical test results, not these latest
inferences. Risk categories use the original unrounded prediction with
continuous thresholds at 15, 35 and 75 to avoid gaps for decimals.

The app reads processed CSVs, saved models, saved predictions, saved feature
importance PNGs and `outputs/model_v1/results.json`. It never trains models
or writes source data. It caches each station's display data for responsiveness.

## One-minute explanation

“This prototype predicts PM2.5 six hours ahead for two monitoring stations
in Suwon. We combine AirKorea pollution measurements with KMA weather data.
After validation, cleaning and hourly merging, we create features from
current and past pollution, weather, hour and month. Each station has its
own Random Forest model.

The dashboard shows the latest complete historical observation, the predicted
PM2.5 and a simple risk message. It is not live data or an official government
warning. Switching stations changes the values and results. The graph compares
saved predictions with actual observations, and the table compares Random
Forest with a baseline that assumes pollution stays unchanged. The model
improved errors modestly, but sharp pollution peaks remain difficult.

AI-assisted coding helped us implement and test each stage. Human instructions,
review and iterative corrections guided the development.”
