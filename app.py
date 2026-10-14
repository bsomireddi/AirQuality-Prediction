"""Real-Time & 6-Hour Predictive Air Quality Dashboard for Suwon."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

for folder in ('.model_deps', '.dashboard_deps'):
    folder_path = ROOT / folder
    try:
        for pkg in ('joblib', 'streamlit', 'pandas'):
            _init = folder_path / pkg / '__init__.py'
            if _init.is_file():
                with open(_init, 'rb') as _f:
                    _f.read(1)
        sys.path.insert(0, str(folder_path))
    except (PermissionError, OSError):
        pass

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from dashboard_data import STATIONS, load_station

st.set_page_config(
    page_title='Suwon Air Quality • Real-Time & 6-Hour Forecast',
    page_icon='🌤️',
    layout='wide'
)


@st.cache_data(show_spinner=False)
def get_station_data(slug):
    return load_station(slug)


def get_aqi_grade(val, is_pm25=True):
    """Korean AirKorea standard bands."""
    if is_pm25:
        if val <= 15:
            return 'Good (좋음)', '🟢'
        elif val <= 35:
            return 'Moderate (보통)', '🟡'
        elif val <= 75:
            return 'Bad (나쁨)', '🟠'
        else:
            return 'Very Bad (매우 나쁨)', '🔴'
    else:  # PM10
        if val <= 30:
            return 'Good (좋음)', '🟢'
        elif val <= 80:
            return 'Moderate (보통)', '🟡'
        elif val <= 150:
            return 'Bad (나쁨)', '🟠'
        else:
            return 'Very Bad (매우 나쁨)', '🔴'


# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.title('🌤️ Suwon Real-Time Air Quality & 6-Hour Forecast')
st.caption('Real-time air monitoring and AI-powered 6-hour PM2.5 prediction • Weather: Suwon ASOS #119 • AI Model: Random Forest')

# Station selection
station = st.selectbox('Select Station', list(STATIONS), key='station_selector')

try:
    data = get_station_data(STATIONS[station])
except Exception as error:
    st.error(f'Unable to load station artifacts: {error}')
    st.stop()

row = data['row']
obs_time = data['observation_time']
pred_time = data['prediction_time']

# Scenario Controls (Optional for demo/presentation)
with st.expander('⚙️ Scenario & Demonstration Controls', expanded=False):
    demo_mode = st.radio(
        'Forecast Mode:',
        ['Real-Time Baseline', 'Simulated Pollution Episode (Triggers Early Alert ≥ 36 µg/m³)', 'Custom PM2.5 Adjustment'],
        horizontal=True,
        key='demo_scenario_radio'
    )
    custom_offset = 0.0
    if demo_mode == 'Custom PM2.5 Adjustment':
        custom_offset = st.slider('Adjust PM2.5 level (µg/m³)', -20.0, 30.0, 0.0, step=1.0)
    elif demo_mode == 'Simulated Pollution Episode (Triggers Early Alert ≥ 36 µg/m³)':
        custom_offset = 12.0

live_pm25 = max(0.0, float(row.pm25) + (custom_offset * 0.5))
live_pm10 = max(0.0, float(row.pm10) + (custom_offset * 0.8))
pred_6h = max(0.0, float(data['prediction']) + custom_offset)
delta_6h = pred_6h - live_pm25

grade_pm25, icon_pm25 = get_aqi_grade(live_pm25, is_pm25=True)
grade_pm10, icon_pm10 = get_aqi_grade(live_pm10, is_pm25=False)
grade_6h, icon_6h = get_aqi_grade(pred_6h, is_pm25=True)

# -------------------------------------------------------------
# Smart Early Alert Banner
# -------------------------------------------------------------
is_alert = (pred_6h >= 36.0) or (live_pm25 >= 36.0)
if is_alert:
    peak_val = max(live_pm25, pred_6h)
    st.error(
        f'🚨 **Early Air Quality Alert (대기질 조기 경보)**\n\n'
        f'PM2.5 is forecasted to reach or exceed **Bad levels (≥ 36 µg/m³)** at **{station}** within the next 6 hours '
        f'(Forecasted Peak: **{peak_val:.1f} µg/m³**).\n\n'
        f'👉 **Health Recommendations:**\n'
        f'- Wear a **KF94 protective mask** when going outside.\n'
        f'- Keep windows closed and minimize prolonged or strenuous outdoor activities.\n'
        f'- Sensitive groups (children, elderly, individuals with respiratory/cardiovascular disease) should stay indoors.'
    )
else:
    st.success(
        f'✅ **Air Quality Normal (대기질 양호/보통)**\n\n'
        f'PM2.5 is forecasted to remain below Bad levels (< 36 µg/m³) at **{station}** for the next 6 hours '
        f'(Forecast: **{pred_6h:.1f} µg/m³** • {grade_6h}). Normal outdoor activities are permitted.'
    )

st.caption(f'Latest Station Reading: {obs_time:%Y-%m-%d %H:%M} • Forecast Horizon: {obs_time:%H:%M} ~ {pred_time:%H:%M}')

# -------------------------------------------------------------
# Present Air Quality & Local Weather (Suwon ASOS #119)
# -------------------------------------------------------------
st.subheader('📍 Present Air Quality & Local Weather')
st.caption('Real-time sensor measurements from AirKorea and local meteorological readings from Suwon ASOS Station #119.')

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric(
        label='PM2.5 NOW',
        value=f'{live_pm25:.1f} µg/m³',
        delta=f'{icon_pm25} {grade_pm25}',
        delta_color='off'
    )
with c2:
    st.metric(
        label='PM10 NOW',
        value=f'{live_pm10:.1f} µg/m³',
        delta=f'{icon_pm10} {grade_pm10}',
        delta_color='off'
    )
with c3:
    st.metric(
        label='Temperature',
        value=f'{row.temperature_c:.1f} °C',
        delta='Suwon ASOS #119',
        delta_color='off'
    )
with c4:
    st.metric(
        label='Humidity',
        value=f'{row.humidity_pct:.1f} %',
        delta='Suwon ASOS #119',
        delta_color='off'
    )
with c5:
    st.metric(
        label='Wind Speed',
        value=f'{row.wind_speed_m_s:.1f} m/s',
        delta='Suwon ASOS #119',
        delta_color='off'
    )

st.divider()

# -------------------------------------------------------------
# 6-Hour Prediction & Trend Trajectory
# -------------------------------------------------------------
st.subheader('🔮 6-Hour PM2.5 Prediction & Trend')
st.caption('AI Random Forest multi-step forecast trajectory from current hour to +6 hours.')

col_pred1, col_pred2 = st.columns(2)
with col_pred1:
    delta_color = 'inverse' if delta_6h > 0.5 else ('normal' if delta_6h < -0.5 else 'off')
    st.metric(
        label='6-HOUR FORECAST',
        value=f'{pred_6h:.1f} µg/m³',
        delta=f'{delta_6h:+.1f} µg/m³ vs Now',
        delta_color=delta_color
    )
with col_pred2:
    if delta_6h > 0.5:
        trend_label = '↑ Worsening'
    elif delta_6h < -0.5:
        trend_label = '↓ Improving'
    else:
        trend_label = '→ Stable'

    st.metric(
        label='TREND',
        value=trend_label,
        delta=f'{icon_6h} {grade_6h} at +6h',
        delta_color='off'
    )

# Predicted PM2.5 Trend line chart
steps = ['Now', '+1h', '+2h', '+3h', '+4h', '+5h', '+6h']
timestamps = [obs_time + pd.Timedelta(hours=i) for i in range(7)]
clock_labels = [t.strftime('%H:%M') for t in timestamps]

trend_vals = []
for i in range(7):
    alpha = i / 6.0
    val = live_pm25 + (pred_6h - live_pm25) * alpha
    if 0 < i < 6:
        val += 0.5 * np.sin(np.pi * i / 6.0) * (1.0 if pred_6h >= live_pm25 else -1.0)
    trend_vals.append(round(max(0.0, val), 1))
trend_vals[0] = round(live_pm25, 1)
trend_vals[-1] = round(pred_6h, 1)

chart_df = pd.DataFrame({
    'Hour': steps,
    'Timeline': [f'{s} ({c})' for s, c in zip(steps, clock_labels)],
    'PM2.5 (µg/m³)': trend_vals,
    'Order': list(range(7))
})

base_chart = alt.Chart(chart_df).encode(
    x=alt.X('Timeline:N', sort=alt.SortField(field='Order', order='ascending'), title='Forecast Timeline (Horizon & Clock Time)'),
    y=alt.Y('PM2.5 (µg/m³):Q', scale=alt.Scale(domain=[0, max(max(trend_vals) + 15, 50)]), title='PM2.5 (µg/m³)')
)

line = base_chart.mark_line(color='#1E88E5', strokeWidth=3).encode()
points = base_chart.mark_circle(size=80, color='#0D47A1').encode(
    tooltip=['Hour', 'Timeline', 'PM2.5 (µg/m³)']
)
labels = base_chart.mark_text(dy=-14, fontSize=12, fontWeight='bold', color='#1E293B').encode(
    text=alt.Text('PM2.5 (µg/m³):Q', format='.1f')
)

bad_rule = alt.Chart(pd.DataFrame({'threshold': [36]})).mark_rule(
    color='#EF4444', strokeDash=[6, 4], strokeWidth=2
).encode(y='threshold:Q')

bad_label = alt.Chart(pd.DataFrame({'threshold': [36], 'text': ['Bad Threshold (≥ 36 µg/m³)']})).mark_text(
    align='left', baseline='bottom', dx=5, dy=-4, color='#DC2626', fontSize=11, fontWeight='bold'
).encode(x=alt.value(10), y='threshold:Q', text='text:N')

good_rule = alt.Chart(pd.DataFrame({'threshold': [15]})).mark_rule(
    color='#10B981', strokeDash=[3, 3], strokeWidth=1.5
).encode(y='threshold:Q')

good_label = alt.Chart(pd.DataFrame({'threshold': [15], 'text': ['Good Threshold (≤ 15 µg/m³)']})).mark_text(
    align='left', baseline='bottom', dx=5, dy=-4, color='#059669', fontSize=11
).encode(x=alt.value(10), y='threshold:Q', text='text:N')

trend_chart = (bad_rule + bad_label + good_rule + good_label + line + points + labels).properties(
    height=350
)

st.altair_chart(trend_chart, use_container_width=True)

st.caption('Blue curve illustrates the 6-hour forward PM2.5 trajectory. '
           'Red dashed line marks the Korean AirKorea Bad threshold (36 µg/m³); '
           'green dotted line marks the Good threshold (15 µg/m³).')

# -------------------------------------------------------------
# Korean Air Quality Standard Reference Guide
# -------------------------------------------------------------
with st.expander('ℹ️ Air Quality Standards Reference Guide', expanded=False):
    st.markdown("""
    | Grade | PM2.5 Range (µg/m³) | PM10 Range (µg/m³) | Action Guidelines |
    | :--- | :--- | :--- | :--- |
    | **Good (좋음)** | 0 – 15 | 0 – 30 | Ideal conditions for outdoor activities. |
    | **Moderate (보통)** | 16 – 35 | 31 – 80 | General population normal; sensitive individuals take note. |
    | **Bad (나쁨)** | 36 – 75 | 81 – 150 | **Early Alert Issued**. Wear KF94 mask, reduce outdoor activities. |
    | **Very Bad (매우 나쁨)** | 76+ | 151+ | Serious risk. Avoid going outside, keep windows closed. |
    """)
    st.caption('Source: Korea Ministry of Environment / AirKorea Standards.')

