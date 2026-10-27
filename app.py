
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI4I Machine Failure Guardian",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f, #111b35);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #101a30;
}
h1, h2, h3 {
    color: #8bbcff;
}
div[data-testid="stMetric"] {
    background: #172540;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #2c4168;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL FILES ----------------
NN_FILE = "ai4i_equipment_failure_nn.keras"
RF_FILE = "ai4i_random_forest.joblib"
PREPROCESSOR_FILE = "ai4i_preprocessor.joblib"
THRESHOLDS_FILE = "ai4i_thresholds.json"
PREDICTIONS_FILE = "ai4i_test_predictions.csv"

@st.cache_resource
def load_models():
    nn = tf.keras.models.load_model(NN_FILE)
    rf = joblib.load(RF_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    with open(THRESHOLDS_FILE, "r") as file:
        thresholds = json.load(file)

    return nn, rf, preprocessor, thresholds

st.title("⚙️ AI4I Machine Failure Guardian")
st.caption("Machine Learning for Predictive Maintenance")
st.markdown(
    "Predict potential machine failures using trained "
    "Neural Network and Random Forest models."
)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict Failure", "Model Performance", "About"]
)

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.header("Project Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", "10,000")
    c2.metric("Normal Records", "9,661")
    c3.metric("Failure Records", "339")

    chart_data = pd.DataFrame({
        "Status": ["Normal", "Failure"],
        "Count": [9661, 339]
    })

    fig = px.pie(
        chart_data,
        names="Status",
        values="Count",
        title="AI4I Dataset: Class Distribution",
        hole=0.45
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This project uses labelled AI4I 2020 data to classify "
        "whether a machine may experience failure. It does not "
        "predict an exact future failure time."
    )

# ---------------- PREDICTION ----------------
elif page == "Predict Failure":
    st.header("Machine Failure Prediction")
    st.write("Enter the machine operating conditions.")

    with st.form("prediction_form"):
        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            help="Use the same machine-type encoding as the training dataset."
        )

        col1, col2 = st.columns(2)

        with col1:
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0, max_value=350.0,
                value=300.0, step=0.1
            )
            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0, max_value=400.0,
                value=310.0, step=0.1
            )
            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0, max_value=10000,
                value=1500
            )

        with col2:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0, max_value=200.0,
                value=40.0, step=0.1
            )
            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0, max_value=500,
                value=100
            )

        submitted = st.form_submit_button(
            "Predict with Both Models",
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

            # Apply the same preprocessing used during training.
            processed = preprocessor.transform(input_data)

            nn_probability = float(
                np.asarray(nn.predict(processed, verbose=0)).reshape(-1)[0]
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

            st.subheader("Prediction Results")
            left, right = st.columns(2)

            with left:
                st.markdown("### Neural Network")
                st.metric(
                    "Failure Probability",
                    f"{nn_probability:.1%}"
                )
                if nn_failure:
                    st.error("Potential Failure Detected")
                else:
                    st.success("Failure Not Flagged")

            with right:
                st.markdown("### Random Forest")
                st.metric(
                    "Failure Probability",
                    f"{rf_probability:.1%}"
                )
                if rf_failure:
                    st.error("Potential Failure Detected")
                else:
                    st.success("Failure Not Flagged")

            st.caption(
                f"Thresholds used: Neural Network {nn_threshold:.2f}; "
                f"Random Forest {rf_threshold:.2f}."
            )

            st.warning(
                "These are model predictions, not a guarantee of machine "
                "condition. This demonstration is not a substitute for "
                "industrial inspection or validated safety systems."
            )

            result = input_data.copy()
            result["NN failure probability"] = nn_probability
            result["NN flagged failure"] = bool(nn_failure)
            result["RF failure probability"] = rf_probability
            result["RF flagged failure"] = bool(rf_failure)

            st.download_button(
                "Download Prediction as CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="machine_failure_prediction.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(
                "The models could not be loaded or used. "
                "Check that all model files exist and that their "
                "preprocessing and library versions are compatible."
            )
            st.code(str(error))

# ---------------- PERFORMANCE ----------------
elif page == "Model Performance":
    st.header("Model Performance")
    st.write(
        "The following evaluation results were reported for the "
        "1500-row test set, containing 51 actual failure cases."
    )

    performance = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Neural Network": [97.73, 68.09, 62.75, 65.31, 97.8],
        "Random Forest": [97.07, 55.07, 74.51, 63.33, 97.8]
    })

    st.dataframe(
        performance.style.format({
            "Neural Network": "{:.2f}%",
            "Random Forest": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "ROC-AUC values are approximate. Other metrics are percentages. "
        "These reported values should be verified against the saved "
        "evaluation results before formal publication."
    )

    st.markdown("### Reported failure detection")
    c1, c2 = st.columns(2)
    c1.metric("Neural Network", "32 / 51 failures detected")
    c2.metric("Random Forest", "38 / 51 failures detected")

    st.info(
        "Random Forest detected more failures in the reported test results, "
        "while the Neural Network produced fewer false positives. "
        "Performance depends on the chosen classification thresholds."
    )

    if os.path.exists(PREDICTIONS_FILE):
        st.download_button(
            "Download Saved Test Predictions",
            data=open(PREDICTIONS_FILE, "rb").read(),
            file_name="ai4i_test_predictions.csv",
            mime="text/csv"
        )

# ---------------- ABOUT ----------------
elif page == "About":
    st.header("About the Project")

    st.markdown("""
    ### Objective
    Develop a machine-learning application that identifies potential
    machine failures from operating-condition data.

    ### Models
    - **Neural Network:** learns patterns from the input features.
    - **Random Forest:** combines multiple decision trees to classify risk.

    ### Input Features
    - Machine type
    - Air temperature
    - Process temperature
    - Rotational speed
    - Torque
    - Tool wear

    ### Dataset
    AI4I 2020 Predictive Maintenance Dataset.

    ### Important Limitations
    - Failure cases are much less common than normal cases.
    - A missed failure can be costly, so recall matters.
    - Model outputs are estimates, not certainty.
    - Results have not been established as a validated industrial safety system.
    """)

st.divider()
st.caption(
    "AI4I Machine Failure Guardian | Educational Deep Learning PBL Project"
)
