
import glob
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "dataset" / "processed"
ML_DIR = ROOT / "dataset" / "ml_dataset"

st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.08);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.18);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_parquet_folder(folder):
    files = glob.glob(str(folder / "part-*.parquet"))
    if not files:
        return pd.DataFrame()

    frames = [pd.read_parquet(file) for file in files]
    df = pd.concat(frames, ignore_index=True)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df


processed = load_parquet_folder(PROCESSED_DIR)
ml_data = load_parquet_folder(ML_DIR)

st.title("🔋 EV Battery Intelligence")
st.caption(
    "Electric vehicle telemetry analytics and predictive operating-risk prototype"
)

if processed.empty:
    st.error("No processed Parquet data found in dataset/processed.")
    st.stop()

processed = processed.dropna(subset=["time"])
processed = processed.sort_values("time")

if not ml_data.empty:
    ml_data = ml_data.dropna(subset=["time"]).sort_values("time")

st.sidebar.title("Dashboard Controls")

vehicles = sorted(processed["vehicle_id"].dropna().unique().tolist())

selected_vehicle = st.sidebar.selectbox(
    "Select vehicle",
    vehicles
)

vehicle_data = processed[
    processed["vehicle_id"] == selected_vehicle
].copy()

if vehicle_data.empty:
    st.warning("No records available for the selected vehicle.")
    st.stop()

if not ml_data.empty and "vehicle_id" in ml_data.columns:
    vehicle_ml = ml_data[
        ml_data["vehicle_id"] == selected_vehicle
    ].copy()
else:
    vehicle_ml = pd.DataFrame()

minimum_date = vehicle_data["time"].min().date()
maximum_date = vehicle_data["time"].max().date()

date_range = st.sidebar.date_input(
    "Telemetry date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    vehicle_data = vehicle_data[
        (vehicle_data["time"].dt.date >= start_date)
        & (vehicle_data["time"].dt.date <= end_date)
    ]

    if not vehicle_ml.empty:
        vehicle_ml = vehicle_ml[
            (vehicle_ml["time"].dt.date >= start_date)
            & (vehicle_ml["time"].dt.date <= end_date)
        ]

st.sidebar.divider()
st.sidebar.caption("Data source: processed TUM EV telemetry")
st.sidebar.caption("Window size: 10 seconds")

if vehicle_data.empty:
    st.warning("No records match the selected date range.")
    st.stop()

latest = vehicle_data.iloc[-1]

voltage = pd.to_numeric(
    vehicle_data["hv_battery_voltage"], errors="coerce"
).dropna()

soc = pd.to_numeric(
    vehicle_data["hv_soc"], errors="coerce"
).dropna()

speed = pd.to_numeric(
    vehicle_data["vehicle_speed"], errors="coerce"
).dropna()

temperature = pd.to_numeric(
    vehicle_data["ambient_air_temp"], errors="coerce"
).dropna()

risk_count = None
risk_percentage = None

if not vehicle_ml.empty and "maintenance_risk" in vehicle_ml.columns:
    labels = pd.to_numeric(
        vehicle_ml["maintenance_risk"], errors="coerce"
    ).dropna()

    if not labels.empty:
        risk_count = int((labels == 1).sum())
        risk_percentage = 100 * risk_count / len(labels)

st.subheader(f"Vehicle Overview · {selected_vehicle}")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Latest Battery Voltage",
    f"{latest['hv_battery_voltage']:.2f} V"
    if pd.notna(latest["hv_battery_voltage"]) else "N/A"
)

c2.metric(
    "Latest SOC",
    f"{latest['hv_soc']:.1f}%"
    if pd.notna(latest["hv_soc"]) else "N/A"
)

c3.metric(
    "Average Speed",
    f"{speed.mean():.1f} km/h" if not speed.empty else "N/A"
)

c4.metric(
    "Average Ambient Temp",
    f"{temperature.mean():.1f} °C"
    if not temperature.empty else "N/A"
)

c5.metric(
    "Telemetry Windows",
    f"{len(vehicle_data):,}"
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Battery Voltage Trend")

    fig = px.line(
        vehicle_data,
        x="time",
        y="hv_battery_voltage",
        title="High-Voltage Battery",
        labels={
            "time": "Time",
            "hv_battery_voltage": "Voltage (V)"
        }
    )

    fig.update_traces(line_width=2)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("State of Charge")

    fig = px.line(
        vehicle_data,
        x="time",
        y="hv_soc",
        title="Battery State of Charge",
        labels={
            "time": "Time",
            "hv_soc": "SOC (%)"
        }
    )

    fig.update_traces(line_width=2)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Vehicle Speed")

    fig = px.area(
        vehicle_data,
        x="time",
        y="vehicle_speed",
        labels={
            "time": "Time",
            "vehicle_speed": "Speed (km/h)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Ambient Temperature")

    fig = px.line(
        vehicle_data,
        x="time",
        y="ambient_air_temp",
        labels={
            "time": "Time",
            "ambient_air_temp": "Temperature (°C)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Battery Temperature Signals")

temperature_columns = [
    column for column in [
        "hv_temp_min",
        "hv_temp_max",
        "battery_temp_avg"
    ]
    if column in vehicle_data.columns
    and vehicle_data[column].notna().any()
]

if temperature_columns:
    temp_data = vehicle_data[["time"] + temperature_columns].melt(
        id_vars="time",
        var_name="Temperature Signal",
        value_name="Temperature (°C)"
    )

    fig = px.line(
        temp_data,
        x="time",
        y="Temperature (°C)",
        color="Temperature Signal"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "Battery temperature measurements are unavailable in the selected "
        "processed records. Ambient temperature is displayed separately."
    )

st.divider()
st.subheader("Historical Operating-Risk Labels")

if not vehicle_ml.empty and "maintenance_risk" in vehicle_ml.columns:
    labels = pd.to_numeric(
        vehicle_ml["maintenance_risk"], errors="coerce"
    ).dropna()

    if not labels.empty:
        risk_left, risk_right = st.columns([1, 2])

        with risk_left:
            st.metric("Windows labelled elevated risk", f"{risk_count:,}")
            st.metric("Share of labelled windows", f"{risk_percentage:.2f}%")

            st.caption(
                "These are proxy labels based on future abnormal voltage "
                "change, not verified maintenance events or battery failures."
            )

        with risk_right:
            distribution = (
                labels.map({0: "Normal proxy label", 1: "Elevated-risk proxy label"})
                .value_counts()
                .rename_axis("Label")
                .reset_index(name="Windows")
            )

            fig = px.bar(
                distribution,
                x="Label",
                y="Windows",
                color="Label",
                title="Distribution of Historical Proxy Labels"
            )

            st.plotly_chart(fig, use_container_width=True)

        if "voltage_change" in vehicle_ml.columns:
            st.subheader("Voltage Change vs SOC Change")

            fig = px.scatter(
                vehicle_ml,
                x="soc_change",
                y="voltage_change",
                color=vehicle_ml["maintenance_risk"].astype(str),
                opacity=0.65,
                labels={
                    "soc_change": "SOC Change (%)",
                    "voltage_change": "Voltage Change (V)",
                    "color": "Proxy Risk Label"
                }
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid proxy labels are available for this vehicle.")
else:
    st.info("The ML dataset does not contain risk labels for this selection.")

st.divider()
st.subheader("Model Evaluation · Held-Out Test Set")

metrics_path = ROOT / "report" / "results" / "test_threshold_metrics.csv"

if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path)

    threshold_options = metrics_df["threshold"].tolist()
    selected_threshold = st.select_slider(
        "Risk alert threshold",
        options=threshold_options,
        value=0.70,
        format_func=lambda value: f"{value:.2f}"
    )

    selected_metrics = metrics_df[
        metrics_df["threshold"] == selected_threshold
    ].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Precision", f"{selected_metrics['precision']:.2%}")
    m2.metric("Risk Recall", f"{selected_metrics['recall']:.2%}")
    m3.metric("Risk F1", f"{selected_metrics['f1']:.4f}")
    m4.metric("False Alarms", f"{int(selected_metrics['false_positives']):,}")

    chart_df = metrics_df.melt(
        id_vars="threshold",
        value_vars=["precision", "recall", "f1"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.line(
        chart_df,
        x="threshold",
        y="Score",
        color="Metric",
        markers=True,
        title="Risk Detection Metrics by Classification Threshold"
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title="Classification Threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        metrics_df.rename(columns={
            "threshold": "Threshold",
            "precision": "Risk Precision",
            "recall": "Risk Recall",
            "f1": "Risk F1",
            "true_positives": "Detected Risk Cases",
            "false_positives": "False Alarms",
            "false_negatives": "Missed Risk Cases"
        }).style.format({
            "Threshold": "{:.2f}",
            "Risk Precision": "{:.2%}",
            "Risk Recall": "{:.2%}",
            "Risk F1": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Metrics use the held-out test set. The target is a project-defined "
        "future voltage-change risk proxy, not confirmed battery failures. "
        "Changing the threshold changes the precision-recall trade-off; "
        "it does not retrain the model."
    )
else:
    st.warning("Model evaluation CSV not found in report/results.")

st.divider()
st.subheader("Recent Telemetry Records")

display_columns = [
    column for column in [
        "time",
        "vehicle_id",
        "vehicle_speed",
        "ambient_air_temp",
        "hv_soc",
        "hv_battery_voltage",
        "hv_temp_min",
        "hv_temp_max"
    ]
    if column in vehicle_data.columns
]

st.dataframe(
    vehicle_data[display_columns]
    .sort_values("time", ascending=False)
    .head(100),
    use_container_width=True,
    hide_index=True
)

csv = vehicle_data.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered telemetry CSV",
    data=csv,
    file_name=f"{selected_vehicle}_telemetry.csv",
    mime="text/csv"
)

st.caption(
    "Research prototype · Historical dataset visualization · "
    "Not a certified battery safety or maintenance system"
)
