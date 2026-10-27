
import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Equipment Failure Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM DESIGN
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1530 55%, #10172d 100%);
    color: #edf3ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.85);
}

section[data-testid="stSidebar"] {
    min-width: 330px !important;
    width: 330px !important;
    background: linear-gradient(180deg, #0b1427, #101c35);
    border-right: 1px solid #304463;
}

section[data-testid="stSidebar"] > div {
    width: 330px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #d0dbee;
    font-size: 16px !important;
    line-height: 1.65;
}

[data-testid="stSidebar"] h3 {
    font-size: 23px !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 8px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px 12px !important;
    min-height: 45px;
    border-radius: 10px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 17px !important;
}

h1, h2, h3 {
    color: #eaf2ff !important;
    letter-spacing: -0.4px;
}

p, label {
    color: #bdc9df;
}

.hero {
    padding: 34px;
    border: 1px solid #30476d;
    border-radius: 22px;
    background: linear-gradient(120deg, #142747, #172044 60%, #211d48);
    margin-bottom: 30px;
}

.hero-title {
    font-size: clamp(27px, 3vw, 39px);
    font-weight: 800;
    line-height: 1.3;
    color: #f5f8ff;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 17px;
    line-height: 1.7;
    color: #c4d2eb;
}

.section-label {
    color: #8db8ff;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #142440, #17213c);
    border: 1px solid #30496f;
    padding: 26px 24px;
    border-radius: 18px;
    min-height: 155px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
}

div[data-testid="stMetric"] label {
    color: #b7c8e4 !important;
    font-size: 16px !important;
}

div[data-testid="stMetricValue"] {
    color: #f2f6ff !important;
    font-size: 34px !important;
    font-weight: 800 !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 14px !important;
}

div[data-testid="stForm"] {
    background: #111e35;
    border: 1px solid #304565;
    border-radius: 20px;
    padding: 28px;
}

div[data-testid="stPlotlyChart"] {
    background: #101c31;
    border: 1px solid #2b4263;
    border-radius: 17px;
    padding: 12px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #3478f6, #7955e8);
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    font-size: 16px;
    padding: 12px 20px;
    min-height: 48px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid #9bbcff;
    color: white;
    filter: brightness(1.12);
}

.stDownloadButton > button {
    background: #1a2d4c;
    color: #e9f1ff;
    border: 1px solid #38577e;
    border-radius: 10px;
    min-height: 44px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #344966;
    margin-top: 24px;
    margin-bottom: 24px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #293d5d;
    border-radius: 12px;
    overflow: hidden;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #101d33;
    border-color: #344b70;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #3478f6, #a27aff);
}

footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        min-width: 280px !important;
        width: 280px !important;
    }

    .hero {
        padding: 22px;
    }

    .hero-title {
        font-size: 27px;
    }

    div[data-testid="stMetric"] {
        padding: 18px;
        min-height: 120px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL FILES
# Keep all model files in the GitHub repository root.
# =========================================================
NN_FILE = "ai4i_equipment_failure_nn.keras"
RF_FILE = "ai4i_random_forest.joblib"
PREPROCESSOR_FILE = "ai4i_preprocessor.joblib"
THRESHOLDS_FILE = "ai4i_thresholds.json"
PREDICTIONS_FILE = "ai4i_test_predictions.csv"
HISTORY_FILE = "prediction_history.csv"


# =========================================================
# CHART THEME
# =========================================================
CHART_LAYOUT = dict(
    paper_bgcolor="#101c31",
    plot_bgcolor="#101c31",
    font=dict(color="#dce8ff", family="Inter", size=13),
    margin=dict(l=30, r=30, t=65, b=35),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


# =========================================================
# LOAD TRAINED MODELS
# =========================================================
@st.cache_resource
def load_models():
    nn = tf.keras.models.load_model(NN_FILE)
    rf = joblib.load(RF_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    with open(THRESHOLDS_FILE, "r") as file:
        thresholds = json.load(file)

    return nn, rf, preprocessor, thresholds


# =========================================================
# DATASET STATISTICS
# =========================================================
TOTAL_MACHINES = 10000
NORMAL_MACHINES = 9661
FAILURE_MACHINES = 339
FAILURE_RATE = FAILURE_MACHINES / TOTAL_MACHINES * 100


# =========================================================
# PREDICTION HISTORY
# =========================================================
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(HISTORY_FILE)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def save_prediction(record):
    new_record = pd.DataFrame([record])
    file_exists = (
        os.path.exists(HISTORY_FILE)
        and os.path.getsize(HISTORY_FILE) > 0
    )

    new_record.to_csv(
        HISTORY_FILE,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False
    )


# =========================================================
# MAIN SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0 24px;">
        <div style="font-size:52px; margin-bottom:12px;">⚙️</div>
        <div style="font-size:22px; font-weight:800; color:#f1f6ff; line-height:1.4;">
            EQUIPMENT FAILURE
        </div>
        <div style="font-size:22px; font-weight:800; color:#f1f6ff; line-height:1.4;">
            PREDICTION
        </div>
        <div style="font-size:13px; letter-spacing:1.5px; color:#86a9df; margin-top:10px;">
            FROM TIME SERIES SENSORS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Choose a page",
        [
            "Overview",
            "Predict Failure",
            "Prediction History",
            "Model Performance",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Project Status")

    st.markdown("""
    <div style="background:#142640; border:1px solid #38577e;
                padding:18px; border-radius:14px;">
        <div style="color:#83b8ff; font-size:14px; font-weight:700;">
            ● MODEL DEMO
        </div>
        <div style="color:#d0dbee; font-size:14px; margin-top:10px; line-height:1.8;">
            AI4I 2020 Dataset<br>
            Neural Network + Random Forest
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.caption("Educational Deep Learning PBL Project")


# =========================================================
# MAIN HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="section-label">INTELLIGENT PREDICTIVE MAINTENANCE</div>
    <div class="hero-title">
        ⚙️ Equipment Failure Prediction
    </div>
    <div class="hero-subtitle">
        From Time Series Sensors<br>
        Neural Network + Random Forest | AI4I 2020 Dataset
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# OVERVIEW PAGE
# =========================================================
if page == "Overview":

    st.markdown("## Dashboard Overview")
    st.caption("Dataset summary based on the AI4I 2020 dataset.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{TOTAL_MACHINES:,}")
    c2.metric("Normal Records", f"{NORMAL_MACHINES:,}")
    c3.metric("Failure Records", f"{FAILURE_MACHINES:,}")
    c4.metric("Failure Rate", f"{FAILURE_RATE:.2f}%")

    st.markdown("")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Machine Status Distribution")

        distribution = pd.DataFrame({
            "Machine Status": ["Normal", "Failure"],
            "Records": [NORMAL_MACHINES, FAILURE_MACHINES]
        })

        fig = px.pie(
            distribution,
            names="Machine Status",
            values="Records",
            hole=0.58,
            color="Machine Status",
            color_discrete_map={
                "Normal": "#3987ff",
                "Failure": "#ff5d78"
            }
        )

        fig.update_traces(
            textinfo="percent+label",
            textfont_size=14,
            marker=dict(line=dict(color="#101c31", width=3))
        )

        fig.update_layout(
            **CHART_LAYOUT,
            height=440,
            showlegend=False,
            annotations=[dict(
                text="10,000<br><sup>Records</sup>",
                x=0.5,
                y=0.5,
                font_size=22,
                showarrow=False,
                font_color="#f0f5ff"
            )]
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Dataset Class Comparison")

        bar_data = pd.DataFrame({
            "Status": ["Normal", "Failure"],
            "Records": [NORMAL_MACHINES, FAILURE_MACHINES]
        })

        fig2 = px.bar(
            bar_data,
            x="Status",
            y="Records",
            color="Status",
            text="Records",
            color_discrete_map={
                "Normal": "#3987ff",
                "Failure": "#ff5d78"
            }
        )

        fig2.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont_size=14
        )

        fig2.update_layout(
            **CHART_LAYOUT,
            height=440,
            showlegend=False,
            yaxis_title="Number of records",
            xaxis_title="",
            yaxis=dict(gridcolor="#263954")
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## Project Highlights")

    info1, info2 = st.columns(2, gap="large")

    with info1:
        st.markdown("### 🧠 Neural Network")
        st.write(
            "Learns patterns from machine operating conditions "
            "and estimates a potential failure probability."
        )

    with info2:
        st.markdown("### 🌳 Random Forest")
        st.write(
            "Combines multiple decision trees to estimate potential "
            "machine failure risk."
        )

    st.info(
        "The AI4I dataset has significantly more normal records than "
        "failure records. High accuracy alone does not mean every failure "
        "will be detected."
    )


# =========================================================
# PREDICT FAILURE PAGE
# =========================================================
elif page == "Predict Failure":

    st.markdown("## Machine Failure Prediction")
    st.caption(
        "Enter the operating conditions to compare both trained models."
    )

    with st.form("prediction_form"):

        st.markdown("### 1. Machine Information")

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            help="Machine type category from the training dataset."
        )

        st.markdown("### 2. Operating Conditions")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1
            )

            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0,
                max_value=10000,
                value=1500,
                step=10
            )

        with col2:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
                step=0.1
            )

            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=500,
                value=100,
                step=1
            )

        submitted = st.form_submit_button(
            "⚡ Predict with Both Models",
            use_container_width=True
        )

    if submitted:
        try:
            nn, rf, preprocessor, thresholds = load_models()

            input_data = pd.DataFrame([{
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear
            }])

            processed = preprocessor.transform(input_data)

            nn_probability = float(
                np.asarray(
                    nn.predict(processed, verbose=0)
                ).reshape(-1)[0]
            )

            rf_probability = float(
                rf.predict_proba(processed)[0][1]
            )

            nn_threshold = float(
                thresholds["neural_network_threshold"]
            )
            rf_threshold = float(
                thresholds["random_forest_threshold"]
            )

            nn_failure = nn_probability >= nn_threshold
            rf_failure = rf_probability >= rf_threshold

            timestamp = datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )

            record = {
                "Date and Time": timestamp,
                "Machine Type": machine_type,
                "Air Temperature (K)": air_temp,
                "Process Temperature (K)": process_temp,
                "Rotational Speed (rpm)": rotational_speed,
                "Torque (Nm)": torque,
                "Tool Wear (min)": tool_wear,
                "NN Failure Probability (%)": round(
                    nn_probability * 100, 2
                ),
                "NN Result": (
                    "Potential Failure" if nn_failure else "Not Flagged"
                ),
                "RF Failure Probability (%)": round(
                    rf_probability * 100, 2
                ),
                "RF Result": (
                    "Potential Failure" if rf_failure else "Not Flagged"
                )
            }

            if "prediction_run_id" not in st.session_state:
                st.session_state["prediction_run_id"] = None

            try:
                save_prediction(record)
                st.success("Prediction completed and saved to history.")
            except OSError as save_error:
                st.warning(
                    "Prediction completed, but history could not be saved: "
                    + str(save_error)
                )

            st.markdown("---")
            st.markdown("## Prediction Results")

            left, right = st.columns(2, gap="large")

            with left:
                st.markdown("### 🧠 Neural Network")

                st.metric(
                    "Estimated Failure Probability",
                    f"{nn_probability:.1%}"
                )

                st.progress(
                    max(0.0, min(1.0, nn_probability))
                )

                if nn_failure:
                    st.error("⚠️ Potential Failure Detected")
                else:
                    st.success("✓ Failure Not Flagged")

                st.caption(
                    f"Classification threshold: {nn_threshold:.2f}"
                )

            with right:
                st.markdown("### 🌳 Random Forest")

                st.metric(
                    "Estimated Failure Probability",
                    f"{rf_probability:.1%}"
                )

                st.progress(
                    max(0.0, min(1.0, rf_probability))
                )

                if rf_failure:
                    st.error("⚠️ Potential Failure Detected")
                else:
                    st.success("✓ Failure Not Flagged")

                st.caption(
                    f"Classification threshold: {rf_threshold:.2f}"
                )

            st.markdown("### Model Probability Comparison")

            comparison = pd.DataFrame({
                "Model": ["Neural Network", "Random Forest"],
                "Estimated Failure Probability (%)": [
                    nn_probability * 100,
                    rf_probability * 100
                ]
            })

            probability_fig = px.bar(
                comparison,
                x="Model",
                y="Estimated Failure Probability (%)",
                color="Model",
                text=comparison[
                    "Estimated Failure Probability (%)"
                ].map(lambda value: f"{value:.1f}%"),
                color_discrete_map={
                    "Neural Network": "#4c91ff",
                    "Random Forest": "#a27aff"
                }
            )

            probability_fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                textfont_size=14
            )

            probability_fig.update_layout(
                **CHART_LAYOUT,
                height=400,
                showlegend=False,
                yaxis=dict(
                    range=[0, 110],
                    gridcolor="#263954"
                ),
                xaxis_title="",
                yaxis_title="Estimated probability (%)"
            )

            st.plotly_chart(
                probability_fig,
                use_container_width=True
            )

            st.caption(
                "Probabilities are model outputs and may not be calibrated "
                "to represent real-world failure frequencies."
            )

            st.warning(
                "These are educational model estimates, not a guarantee "
                "of machine condition. Do not use this demo alone for "
                "industrial safety or maintenance decisions."
            )

            result_download = input_data.copy()
            result_download["NN Failure Probability (%)"] = round(
                nn_probability * 100, 2
            )
            result_download["NN Flagged Failure"] = bool(nn_failure)
            result_download["RF Failure Probability (%)"] = round(
                rf_probability * 100, 2
            )
            result_download["RF Flagged Failure"] = bool(rf_failure)

            st.download_button(
                "⬇️ Download This Prediction (CSV)",
                data=result_download.to_csv(index=False).encode("utf-8"),
                file_name="machine_failure_prediction.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(
                "The models could not be loaded or used. "
                "Check the model files and library compatibility."
            )
            st.code(str(error))


# =========================================================
# PREDICTION HISTORY PAGE
# =========================================================
elif page == "Prediction History":

    st.markdown("## Prediction History")
    st.caption("Review saved predictions and compare their probabilities.")

    history = load_history()

    if history.empty:
        st.info(
            "No saved predictions yet. Open Predict Failure, enter the "
            "machine conditions and run a prediction to start your history."
        )

    else:
        total_predictions = len(history)

        nn_flagged = (
            history["NN Result"].eq("Potential Failure").sum()
            if "NN Result" in history.columns else 0
        )

        rf_flagged = (
            history["RF Result"].eq("Potential Failure").sum()
            if "RF Result" in history.columns else 0
        )

        c1, c2, c3 = st.columns(3, gap="large")

        c1.metric("Total Predictions", f"{total_predictions:,}")
        c2.metric("NN Failure Flags", f"{nn_flagged:,}")
        c3.metric("RF Failure Flags", f"{rf_flagged:,}")

        st.markdown("### Failure Probability History")

        probability_columns = [
            column for column in [
                "NN Failure Probability (%)",
                "RF Failure Probability (%)"
            ]
            if column in history.columns
        ]

        if probability_columns and "Date and Time" in history.columns:
            chart_history = history.tail(50).copy()
            chart_history["Prediction Number"] = range(
                1, len(chart_history) + 1
            )

            long_history = chart_history.melt(
                id_vars=["Prediction Number", "Date and Time"],
                value_vars=probability_columns,
                var_name="Model",
                value_name="Failure Probability (%)"
            )

            long_history["Model"] = long_history["Model"].replace({
                "NN Failure Probability (%)": "Neural Network",
                "RF Failure Probability (%)": "Random Forest"
            })

            history_fig = px.line(
                long_history,
                x="Prediction Number",
                y="Failure Probability (%)",
                color="Model",
                markers=True,
                hover_data=["Date and Time"],
                color_discrete_map={
                    "Neural Network": "#4c91ff",
                    "Random Forest": "#a27aff"
                }
            )

            history_fig.update_layout(
                **CHART_LAYOUT,
                height=440,
                xaxis_title="Prediction number (latest 50)",
                yaxis_title="Failure probability (%)",
                yaxis=dict(
                    range=[0, 100],
                    gridcolor="#263954"
                )
            )

            st.plotly_chart(
                history_fig,
                use_container_width=True
            )

        st.markdown("### Saved Prediction Records")

        st.dataframe(
            history.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Full Prediction History",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        st.caption(
            "Streamlit Cloud local files are not guaranteed to persist "
            "through restarts, rebuilds or redeployments. For permanent "
            "history, connect a persistent database."
        )


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================
elif page == "Model Performance":

    st.markdown("## Model Performance")
    st.caption(
        "Reported project results from the 1,500-row test set "
        "containing 51 actual failure cases."
    )

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Neural Network": [
            97.73, 68.09, 62.75, 65.31, 97.8
        ],
        "Random Forest": [
            97.07, 55.07, 74.51, 63.33, 97.8
        ]
    })

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.metric("Neural Network Accuracy", "97.73%")
        st.metric("NN Failures Detected", "32 / 51")

    with c2:
        st.metric("Random Forest Accuracy", "97.07%")
        st.metric("RF Failures Detected", "38 / 51")

    st.markdown("### Performance Metric Comparison")

    performance_long = performance.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Score (%)"
    )

    performance_fig = px.bar(
        performance_long,
        x="Metric",
        y="Score (%)",
        color="Model",
        barmode="group",
        text=performance_long["Score (%)"].map(
            lambda value: f"{value:.1f}%"
        ),
        color_discrete_map={
            "Neural Network": "#4c91ff",
            "Random Forest": "#a27aff"
        }
    )

    performance_fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )

    performance_fig.update_layout(
        **CHART_LAYOUT,
        height=470,
        yaxis=dict(
            range=[0, 115],
            gridcolor="#263954"
        ),
        yaxis_title="Score (%)",
        xaxis_title=""
    )

    st.plotly_chart(
        performance_fig,
        use_container_width=True
    )

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric"),
            "Neural Network": st.column_config.NumberColumn(
                "Neural Network (%)",
                format="%.2f%%"
            ),
            "Random Forest": st.column_config.NumberColumn(
                "Random Forest (%)",
                format="%.2f%%"
            )
        }
    )

    st.markdown("### Failure Detection Comparison")

    detection = pd.DataFrame({
        "Model": ["Neural Network", "Random Forest"],
        "Failures Detected": [32, 38],
        "Failures Missed": [19, 13]
    })

    detection_long = detection.melt(
        id_vars="Model",
        var_name="Outcome",
        value_name="Number of cases"
    )

    detection_fig = px.bar(
        detection_long,
        x="Model",
        y="Number of cases",
        color="Outcome",
        barmode="group",
        text="Number of cases",
        color_discrete_map={
            "Failures Detected": "#3987ff",
            "Failures Missed": "#ff647c"
        }
    )

    detection_fig.update_layout(
        **CHART_LAYOUT,
        height=400,
        yaxis=dict(gridcolor="#263954"),
        xaxis_title=""
    )

    st.plotly_chart(
        detection_fig,
        use_container_width=True
    )

    st.info(
        "In the reported results, Random Forest detected more failures, "
        "while Neural Network had fewer false positives. Recall matters "
        "because missed failures can be costly."
    )

    st.caption(
        "ROC-AUC values are approximate. These are reported project results "
        "and should be verified against saved evaluation outputs before "
        "formal publication. The models are not validated industrial "
        "safety systems."
    )

    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "rb") as file:
            prediction_file = file.read()

        st.download_button(
            "⬇️ Download Saved Test Predictions",
            data=prediction_file,
            file_name="ai4i_test_predictions.csv",
            mime="text/csv"
        )


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================
elif page == "About Project":

    st.markdown("## About the Project")

    st.markdown("""
    This project demonstrates machine-learning approaches for identifying
    potential machine failures using the AI4I 2020 Predictive Maintenance
    Dataset.

    ### 🎯 Objective

    Build an interactive application that estimates potential machine
    failure risk from machine operating-condition features.

    ### 🧠 Models Used

    **Neural Network**
    - Learns patterns from the input features.
    - Produces an estimated failure probability.

    **Random Forest**
    - Combines predictions from multiple decision trees.
    - Produces an estimated failure probability.

    ### 📥 Input Features
    - Machine type
    - Air temperature (K)
    - Process temperature (K)
    - Rotational speed (rpm)
    - Torque (Nm)
    - Tool wear (min)

    ### 📊 Dataset
    AI4I 2020 Predictive Maintenance Dataset.

    ### ⚠️ Limitations
    - Normal records greatly outnumber failure records.
    - High accuracy does not guarantee that failures will be detected.
    - Model probabilities may not correspond to real-world failure frequencies.
    - Predictions are not a substitute for inspection or validated safety systems.
    - The project has not been established as an industrially validated system.
    """)

    st.markdown("### Project Workflow")

    workflow = pd.DataFrame({
        "Step": [
            "1. Input",
            "2. Preprocessing",
            "3. Model Prediction",
            "4. Results",
            "5. History"
        ],
        "Description": [
            "Enter machine operating conditions",
            "Apply the saved training preprocessor",
            "Run Neural Network and Random Forest",
            "Compare estimated failure probabilities",
            "Save and review prediction records"
        ]
    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown("""
<div style="text-align:center; padding:18px 0 24px;">
    <div style="font-size:17px; font-weight:800; color:#a8c6ff;">
        ⚙️ Equipment Failure Prediction
    </div>
    <div style="font-size:14px; color:#9aacc8; margin-top:8px;">
        From Time Series Sensors · Neural Network + Random Forest
    </div>
    <div style="font-size:12px; color:#8497b6; margin-top:6px;">
        Educational Deep Learning PBL Project
    </div>
</div>
""", unsafe_allow_html=True)
