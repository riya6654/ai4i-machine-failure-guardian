
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Machine Failure Guardian",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM DESIGN =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #071426;
    --panel: #0d2039;
    --blue: #168bff;
    --purple: #9333ea;
    --text: #edf4ff;
    --muted: #91a6c4;
    --border: rgba(117,163,220,.17);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(ellipse at 70% 0%,
    rgba(22,139,255,.12), transparent 38%), #071426;
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: #08172a;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.brand {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 5px 0 22px;
}

.brand-icon {
    background: linear-gradient(135deg,#168bff,#9333ea);
    border-radius: 14px;
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.brand-name {
    font-size: 15px;
    font-weight: 800;
    color: #f1f6ff;
}

.brand-caption {
    font-size: 10px;
    color: #91a6c4;
    letter-spacing: 1px;
    margin-top: 4px;
}

.eyebrow {
    color: #65b4ff;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 9px;
}

.hero {
    background: linear-gradient(115deg,#102b4c,#0d2039 60%,#1b2450);
    border: 1px solid rgba(93,164,255,.3);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 24px;
}

.hero-chip {
    display: inline-block;
    background: rgba(22,139,255,.13);
    color: #80c4ff;
    border: 1px solid rgba(22,139,255,.32);
    border-radius: 30px;
    padding: 7px 12px;
    font-size: 11px;
    margin-bottom: 17px;
}

.hero h1 {
    color: #f3f7ff;
    font-size: clamp(28px,3.4vw,43px);
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 14px;
}

.hero p {
    color: #a9bfdc;
    font-size: 14px;
    max-width: 720px;
    line-height: 1.8;
}

.metric-card {
    background: linear-gradient(145deg,#102847,#0b1b30);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    min-height: 135px;
}

.metric-icon {
    float: right;
    font-size: 21px;
    background: rgba(22,139,255,.14);
    border-radius: 10px;
    padding: 8px;
}

.metric-label {
    font-size: 12px;
    color: #a2b5d0;
    margin: 5px 0 13px;
}

.metric-value {
    font-size: 29px;
    color: #f5f8ff;
    font-weight: 800;
}

.metric-note {
    font-size: 11px;
    color: #7e9abb;
    margin-top: 8px;
}

.panel {
    background: rgba(13,32,57,.92);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 22px;
    margin-bottom: 20px;
}

.panel-title {
    font-size: 16px;
    font-weight: 700;
    color: #edf4ff;
    margin-bottom: 6px;
}

.panel-subtitle {
    color: #8ea6c5;
    font-size: 12px;
    margin-bottom: 20px;
}

.info-box {
    background: rgba(22,139,255,.08);
    border: 1px solid rgba(22,139,255,.2);
    border-radius: 12px;
    padding: 15px;
    color: #b9cce4;
    font-size: 12px;
    line-height: 1.8;
    margin-bottom: 18px;
}

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg,#0788f9,#813df2);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 43px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    box-shadow: 0 0 20px rgba(22,139,255,.3);
    color: white;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #102847;
    border-color: #24466d;
    border-radius: 9px;
}

.footer {
    border-top: 1px solid var(--border);
    padding-top: 20px;
    margin-top: 35px;
    color: #7188a8;
    font-size: 11px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ================= HELPER FUNCTIONS =================
def metric_card(icon, label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def heading(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="eyebrow">{eyebrow}</div>
    <h2 style="color:#edf4ff;margin:0 0 8px;font-size:28px">
        {title}
    </h2>
    <p style="color:#91a6c4;font-size:13px;margin-bottom:24px">
        {subtitle}
    </p>
    """, unsafe_allow_html=True)


def chart_style(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=10,r=10,t=25,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a9bfd8",family="Inter"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.12
        ),
        xaxis=dict(gridcolor="rgba(145,166,196,.10)"),
        yaxis=dict(gridcolor="rgba(145,166,196,.10)")
    )
    return fig


# ================= NAVIGATION =================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">⚙️</div>
        <div>
            <div class="brand-name">AI Machine<br>Failure Guardian</div>
            <div class="brand-caption">PREDICTIVE MAINTENANCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Overview Dashboard",
            "Predict Machine Failure",
            "Model Performance",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**🟢 AI4I 2020 Dataset**")
    st.caption("Neural Network + Random Forest")
    st.caption("Deep Learning Project · 2026")


# ================= DASHBOARD =================
if page == "Overview Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="hero-chip">✦ AI-POWERED PREDICTIVE MAINTENANCE</div>
        <h1>AI Machine Failure<br>Guardian</h1>
        <p>
        Intelligent machine monitoring with Machine Learning and Deep Learning.
        Explore operating conditions, compare prediction models, and understand
        potential equipment failure risks through an interactive dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    heading(
        "YOUR WORKSPACE",
        "Dashboard Overview",
        "An overview of the AI4I 2020 predictive maintenance dataset."
    )

    st.caption("Dataset counts below are based on the AI4I 2020 dataset.")

    cols = st.columns(4, gap="medium")

    with cols[0]:
        metric_card("📊","Dataset Records","10,000","Total dataset rows")

    with cols[1]:
        metric_card("🟢","Normal Records","9,661","96.61% of dataset")

    with cols[2]:
        metric_card("🔴","Failure Records","339","3.39% of dataset")

    with cols[3]:
        metric_card("🤖","ML Models","02","NN + Random Forest")

    st.write("")

    left, right = st.columns([1.5,1], gap="large")

    with left:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Dataset Class Distribution</div>'
            '<div class="panel-subtitle">Normal versus failure records</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure(data=[go.Pie(
            labels=["Normal","Failure"],
            values=[9661,339],
            hole=.72,
            marker=dict(colors=["#168bff","#9333ea"]),
            textinfo="label+percent",
            textfont=dict(color="white",size=12)
        )])

        fig.update_layout(
            height=310,
            margin=dict(l=5,r=5,t=15,b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a9bfd8")
        )

        st.plotly_chart(
            fig,use_container_width=True,
            config={"displayModeBar":False}
        )
        st.markdown("</div>",unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Project Workflow</div>'
            '<div class="panel-subtitle">From machine data to prediction</div>',
            unsafe_allow_html=True
        )

        steps = [
            ("01","Dataset Preparation","Clean and prepare operating data."),
            ("02","Model Training","Train NN and Random Forest models."),
            ("03","Evaluation","Compare classification metrics."),
            ("04","Prediction","Evaluate new machine parameters.")
        ]

        for number,title,description in steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin:20px 0">
                <div style="min-width:35px;height:35px;border-radius:10px;
                display:flex;align-items:center;justify-content:center;
                background:rgba(22,139,255,.15);color:#65b4ff;
                font-weight:800">{number}</div>
                <div>
                    <div style="font-weight:700;color:#edf4ff;font-size:13px">
                    {title}</div>
                    <div style="color:#91a6c4;font-size:11px;margin-top:5px">
                    {description}</div>
                </div>
            </div>
            """,unsafe_allow_html=True)

        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Technology Stack</div>'
        '<div class="panel-subtitle">Tools used in this project</div>',
        unsafe_allow_html=True
    )

    tech_cols = st.columns(4)
    tech = [
        ("🧠","Neural Network","Deep Learning"),
        ("🌳","Random Forest","Machine Learning"),
        ("🐍","Python","Model development"),
        ("📈","Plotly","Data visualization")
    ]

    for col,(icon,name,description) in zip(tech_cols,tech):
        with col:
            st.markdown(f"""
            <div style="padding:16px;background:#102847;border-radius:12px;
            border:1px solid rgba(117,163,220,.12)">
                <div style="font-size:23px">{icon}</div>
                <div style="font-size:13px;font-weight:700;color:#edf4ff;
                margin-top:9px">{name}</div>
                <div style="font-size:11px;color:#91a6c4;margin-top:5px">
                {description}</div>
            </div>
            """,unsafe_allow_html=True)

    st.markdown("</div>",unsafe_allow_html=True)


# ================= PREDICTION PAGE =================
elif page == "Predict Machine Failure":

    heading(
        "PREDICTIVE ANALYTICS",
        "Predict Machine Failure",
        "Enter operating conditions to prepare a machine assessment."
    )

    st.markdown("""
    <div class="info-box">
        <b>Preview mode:</b> This page currently collects machine parameters.
        The saved Neural Network and Random Forest models have not yet been
        connected, so no real failure prediction is made here.
    </div>
    """,unsafe_allow_html=True)

    with st.form("machine_parameters"):

        st.markdown("### ⚙️ Machine Parameters")
        st.caption("Enter the six features required by the models.")

        c1,c2 = st.columns(2,gap="large")

        with c1:
            product_type = st.selectbox(
                "Product Type",
                ["M","L","H"],
                help="M = Medium, L = Low, H = High"
            )

            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,max_value=350.0,
                value=300.0,step=.1
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,max_value=400.0,
                value=310.0,step=.1
            )

        with c2:
            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0,max_value=5000,
                value=1500,step=10
            )

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,max_value=100.0,
                value=40.0,step=.5
            )

            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0,max_value=300,
                value=100,step=1
            )

        submitted = st.form_submit_button(
            "⚡ Prepare Prediction",
            use_container_width=True
        )

    if submitted:
        st.session_state["machine_input"] = {
            "Type":product_type,
            "Air temperature [K]":air_temp,
            "Process temperature [K]":process_temp,
            "Rotational speed [rpm]":rotational_speed,
            "Torque [Nm]":torque,
            "Tool wear [min]":tool_wear
        }

    if "machine_input" in st.session_state:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Input Summary</div>'
            '<div class="panel-subtitle">Review your selected parameters</div>',
            unsafe_allow_html=True
        )

        summary = pd.DataFrame([st.session_state["machine_input"]])
        st.dataframe(summary,use_container_width=True,hide_index=True)

        st.download_button(
            "⬇️ Download Input CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="machine_input.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.info(
            "Inputs captured successfully. Real model predictions will be "
            "available after connecting the trained model files."
        )
        st.markdown("</div>",unsafe_allow_html=True)


# ================= PERFORMANCE PAGE =================
elif page == "Model Performance":

    heading(
        "MODEL EVALUATION",
        "Model Performance",
        "Understand the metrics used to evaluate failure classifiers."
    )

    st.markdown("""
    <div class="info-box">
        Performance figures on this page are illustrative design placeholders.
        They are not your trained models' actual results. We will replace them
        with metrics calculated from your saved test predictions.
    </div>
    """,unsafe_allow_html=True)

    model = st.selectbox(
        "Choose model",
        ["Neural Network","Random Forest"]
    )

    cols = st.columns(4)
    metrics = [
        ("Accuracy","92.0%","Overall correct predictions"),
        ("Precision","84.0%","Positive prediction quality"),
        ("Recall","79.0%","Actual failures detected"),
        ("F1-score","81.0%","Precision-recall balance")
    ]

    for col,(name,value,note) in zip(cols,metrics):
        with col:
            metric_card("📊",name,value,note)

    st.write("")
    left,right = st.columns(2,gap="large")

    with left:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Confusion Matrix</div>'
            '<div class="panel-subtitle">Illustrative example only</div>',
            unsafe_allow_html=True
        )

        cm = go.Figure(data=go.Heatmap(
            z=[[920,30],[50,100]],
            x=["Predicted Normal","Predicted Failure"],
            y=["Actual Normal","Actual Failure"],
            colorscale=[
                [0,"#102847"],
                [.5,"#087ee1"],
                [1,"#9333ea"]
            ],
            showscale=False,
            text=[[920,30],[50,100]],
            texttemplate="%{text}",
            textfont={"size":18,"color":"white"}
        ))

        cm.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label"
        )

        st.plotly_chart(
            chart_style(cm),
            use_container_width=True,
            config={"displayModeBar":False}
        )
        st.markdown("</div>",unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Metric Breakdown</div>'
            '<div class="panel-subtitle">Illustrative example only</div>',
            unsafe_allow_html=True
        )

        scores = pd.DataFrame({
            "Metric":["Accuracy","Precision","Recall","F1-score"],
            "Score":[.92,.84,.79,.81]
        })

        fig = go.Figure(go.Bar(
            x=scores["Score"],
            y=scores["Metric"],
            orientation="h",
            marker=dict(
                color=scores["Score"],
                colorscale=[[0,"#168bff"],[1,"#9333ea"]]
            ),
            text=[f"{x:.0%}" for x in scores["Score"]],
            textposition="outside"
        ))

        fig.update_layout(
            xaxis=dict(range=[0,1.12],tickformat=".0%"),
            yaxis=dict(autorange="reversed")
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True,
            config={"displayModeBar":False}
        )
        st.markdown("</div>",unsafe_allow_html=True)


# ================= ABOUT PAGE =================
elif page == "About Project":

    heading(
        "PROJECT INFORMATION",
        "About the Project",
        "Project objective, workflow, input parameters and limitations."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-chip">AI4I 2020 · PREDICTIVE MAINTENANCE</div>
        <h1>Smarter insights.<br>Better maintenance planning.</h1>
        <p>
        AI Machine Failure Guardian explores how machine operating data can
        be used to classify potential equipment failures using a Neural Network
        and a Random Forest classifier.
        </p>
    </div>
    """,unsafe_allow_html=True)

    left,right = st.columns(2,gap="large")

    with left:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown("### 🎯 Project Objective")

        st.write(
            "Build and compare a Neural Network and a Random Forest classifier "
            "using operating parameters from the AI4I 2020 dataset."
        )

        st.divider()
        st.markdown("### 📥 Input Parameters")

        for item in [
            "Product Type (L, M, H)",
            "Air Temperature (K)",
            "Process Temperature (K)",
            "Rotational Speed (rpm)",
            "Torque (Nm)",
            "Tool Wear (minutes)"
        ]:
            st.markdown(f"- {item}")

        st.markdown("</div>",unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown("### 🧠 Project Workflow")

        workflow = [
            ("01","Data Preparation","Prepare labelled operating data."),
            ("02","Model Training","Train both classifiers."),
            ("03","Evaluation","Compare classification metrics."),
            ("04","Prediction","Evaluate new machine inputs.")
        ]

        for num,title,desc in workflow:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin:20px 0">
                <div style="color:#67baff;font-weight:800">{num}</div>
                <div>
                    <div style="color:#edf4ff;font-weight:700;font-size:13px">
                    {title}</div>
                    <div style="color:#91a6c4;font-size:11px;margin-top:5px">
                    {desc}</div>
                </div>
            </div>
            """,unsafe_allow_html=True)

        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("### ⚠️ Limitations & Responsible Use")

    st.markdown("""
    - This is an educational predictive-maintenance prototype.
    - Dataset results do not guarantee performance on real industrial machines.
    - Predictions depend on input quality and the trained model.
    - This tool should not be the sole basis for safety-critical maintenance.
    - Further validation is required before practical industrial deployment.
    """)
    st.markdown("</div>",unsafe_allow_html=True)


# ================= FOOTER =================
st.markdown("""
<div class="footer">
    AI MACHINE FAILURE GUARDIAN · AI4I 2020
    <br><br>
    Educational predictive-maintenance prototype
</div>
""",unsafe_allow_html=True)
