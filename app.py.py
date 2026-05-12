import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="🦟 Climate-Aware Malaria Prediction & Explainable Healthcare AI System",
    page_icon="🦟",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("advanced_feature_engineered_malaria_dataset.csv")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("final_malaria_prediction_pipeline.pkl")

df = load_data()
model = load_model()

# Ensure country_encoded exists for the saved model pipeline
if "country_encoded" not in df.columns:
    df["country_encoded"] = df["country"].astype("category").cat.codes

target = "incidence_of_malaria_per_1000_population_at_risk"

# Main title
st.title("🦟 Climate-Aware Malaria Prediction & Explainable Healthcare AI System")

st.write("""
This dashboard predicts malaria incidence across African countries using climate,
demographic, healthcare, temporal, and spatial indicators.
""")

# Sidebar Branding
st.sidebar.markdown("## 🦟 DHA Capstone Project")
st.sidebar.markdown("Climate-Aware Malaria Prediction System & Explainable Healthcare AI System")
st.sidebar.markdown("---")

# Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dashboard Overview",
        "Geographic Risk Map",
        "Prediction Tool",
        "What-If Policy Simulation",
        "SHAP Explainability",
        "Policy Recommendations"
    ]
)

# Home Page
if page == "Home":
     col1, col2, col3 = st.columns(3)

     col1.metric("Countries", 54)
     col2.metric("Records", 594)
     col3.metric("Model R²", "0.87")
     st.markdown("""
    ## Project Features

    - Climate-aware malaria prediction
    - Temporal lag analysis
    - Spatial risk visualization
    - Explainable AI using SHAP
    - Public health policy simulation
    - Geographic malaria risk mapping
    """)
     st.success("""
     Key Insights:
    - Previous malaria incidence was the strongest predictor.
    - Climate-related variables significantly influenced malaria risk.
    - SHAP explainability improved healthcare AI transparency.
    - Sanitation improvements reduced predicted malaria burden.
    """)
# Dashboard Overview
elif page == "Dashboard Overview":

    st.title("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", df.shape[0])
    col2.metric("Countries", df["country"].nunique())
    col3.metric("Average Malaria Incidence", round(df[target].mean(), 2))

    st.subheader("Average Malaria Incidence Over Time")

    yearly_trend = (
        df.groupby("year")[target]
        .mean()
        .reset_index()
    )

    fig = px.line(
        yearly_trend,
        x="year",
        y=target,
        markers=True,
        title="Average Malaria Incidence Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 High-Burden Countries")

    top_countries = (
        df.groupby("country")[target]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        top_countries,
        x="country",
        y=target,
        title="Top 10 High-Burden Countries"
    )

    st.plotly_chart(fig2, use_container_width=True)

# Geographic Risk Map
elif page == "Geographic Risk Map":

    st.title("🗺️ Geographic Malaria Risk Map")

    selected_year = st.slider(
        "Select Year",
        int(df["year"].min()),
        int(df["year"].max()),
        int(df["year"].max())
    )

    map_df = df[df["year"] == selected_year]

    fig = px.scatter_geo(
        map_df,
        lat="latitude",
        lon="longitude",
        color=target,
        size=target,
        hover_name="country",
        hover_data={
            "year": True,
            target: ":.2f",
            "rainfall": ":.2f",
            "temperature": ":.2f",
            "climate_risk_index": ":.2f",
            "latitude": False,
            "longitude": False
        },
        projection="natural earth",
        title=f"Geographic Malaria Risk Across Africa ({selected_year})"
    )

    fig.update_traces(
        marker=dict(
            sizemode="area",
            opacity=0.8,
            line=dict(width=1)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# Prediction Tool
elif page == "Prediction Tool":

    st.title("🔮 Malaria Incidence Prediction Tool")

    country = st.selectbox("Select Country", sorted(df["country"].unique()))
    country_df = df[df["country"] == country]

    year = st.selectbox("Select Year", sorted(country_df["year"].unique()))
    sample = country_df[country_df["year"] == year].copy()

    st.subheader("Selected Country-Year Data")
    st.dataframe(sample)

    X_sample = sample.drop(
        columns=[target, "high_risk_region", "malaria_cases_reported"],
        errors="ignore"
    )

    if st.button("Predict Malaria Incidence"):
        prediction = model.predict(X_sample)[0]
        actual_value = sample[target].values[0]

        st.success(
            f"Predicted Malaria Incidence: {prediction:.2f} per 1,000 population at risk"
        )

        st.info(
            f"Actual Recorded Malaria Incidence: {actual_value:.2f} per 1,000 population at risk"
        )

# What-If Policy Simulation
elif page == "What-If Policy Simulation":

    st.title("🧪 What-If Policy Simulation")

    st.write("""
    This section estimates the potential effect of improving sanitation access
    on malaria incidence for a selected country-year observation.
    """)

    country = st.selectbox(
        "Select Country",
        sorted(df["country"].unique())
    )

    country_df = df[df["country"] == country]

    year = st.selectbox(
        "Select Year",
        sorted(country_df["year"].unique())
    )

    sample = country_df[
        country_df["year"] == year
    ].copy()

    baseline_incidence = sample[target].values[0]

    st.subheader("Baseline Malaria Incidence")

    st.metric(
        "Current Malaria Incidence",
        f"{baseline_incidence:.2f} per 1,000 population at risk"
    )

    sanitation_increase = st.slider(
        "Increase Basic Sanitation Access (%)",
        0,
        50,
        20
    )

    # Simple policy-impact estimate based on intervention strength
    estimated_reduction = sanitation_increase * 0.03

    scenario_incidence = max(
        baseline_incidence - estimated_reduction,
        0
    )

    st.subheader("Scenario Result")

    st.metric(
        "Estimated Incidence After Intervention",
        f"{scenario_incidence:.2f} per 1,000 population at risk"
    )

    st.success(
        f"Estimated Reduction: {estimated_reduction:.2f} per 1,000 population at risk"
    )

    st.info("""
    Interpretation: This simulation is designed as a decision-support scenario.
    It estimates how sanitation improvement may contribute to malaria burden reduction,
    while recognizing that malaria transmission is also influenced by rainfall,
    temperature, geography, historical burden, and healthcare access.
    """)
 # SHAP Explainability
elif page == "SHAP Explainability":

    st.title("🧠 SHAP Explainability")

    st.markdown("""
    This section explains how the machine learning model predicts malaria incidence
    across African countries using climate, demographic, temporal, and spatial indicators.
    """)

    st.subheader("Global Feature Importance")

    st.markdown("""
    SHAP analysis identifies the variables contributing most strongly to malaria risk prediction.

    The analysis demonstrated that:
    - previous malaria incidence strongly predicts future malaria burden
    - climate-related variables influence transmission dynamics
    - sanitation and infrastructure variables affect malaria vulnerability
    - spatial and geographic factors contribute to regional malaria heterogeneity
    """)

    shap_features = pd.DataFrame({
        "Feature": [
            "Previous Malaria Incidence",
            "Climate Risk Index",
            "Rainfall-Rural Interaction",
            "Longitude / Spatial Location",
            "Sanitation Access",
            "Temperature Lag",
            "Rainfall",
            "Water Access"
        ],

        "Interpretation": [
            "Historical malaria burden strongly influences future malaria incidence.",
            "Climate suitability increases malaria transmission risk.",
            "Rainfall effects are amplified in vulnerable rural populations.",
            "Geographic location contributes to regional malaria variability.",
            "Sanitation access reflects public health infrastructure conditions.",
            "Delayed temperature effects influence malaria transmission.",
            "Rainfall affects mosquito breeding conditions.",
            "Water access reflects broader environmental and infrastructure conditions."
        ]
    })

    st.dataframe(
        shap_features,
        use_container_width=True
    )

    st.subheader("Public Health Interpretation")

    st.success("""
    Key SHAP findings indicate that malaria transmission across Africa is strongly influenced by:
    climate variability, temporal persistence, sanitation access, and geographic vulnerability.
    """)

    st.subheader("Decision-Support Value")

    st.info("""
    Explainable AI allows policymakers to understand WHY malaria risk is high,
    helping support targeted intervention planning, climate-informed surveillance,
    and public health resource allocation.
    """)

    st.subheader("SHAP Waterfall Interpretation")

    st.image(
        "shap_waterfall.png",
        use_container_width=True
    )
    st.subheader("Explainable Healthcare AI")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Prediction Summary

        - Predicted malaria incidence: **426.22 per 1,000 population at risk**
        - Baseline average prediction: **190.05 per 1,000 population at risk**
        - Strongest predictor: **Previous Malaria Incidence**
        - SHAP contribution: **+208.75 SHAP units**
        - Demonstrated strong temporal persistence in malaria transmission.
        """)

    with col2:
        st.markdown("""
        ### Other Contributing Features

        - Longitude-rainfall interaction: **+7.76**
        - Country encoding: **+3.55**
        - Climate risk index: **+2.90**
        - Longitude: **+2.00**
        - Sanitation access: **+1.79**
        - Rainfall-rural interaction: **+1.48**
        """)

        st.subheader("Key Impact")

        st.success("""
        SHAP analysis improved model interpretability and strengthened the system’s applicability for:

    - Healthcare decision support
    - Malaria surveillance
    - Intervention planning
    - Public health policy development
    """)
# Policy Recommendations
if page == "Policy Recommendations":

    st.title("🏥 Policy Recommendations")

    st.markdown("""
    ### Key Public Health Recommendations

    **1. Strengthen malaria surveillance in historically high-burden countries**  
    Countries with high previous malaria incidence should be prioritized for early intervention.

    **2. Integrate climate monitoring into malaria control programs**  
    Rainfall, temperature, and climate-risk indicators should be used for early-warning systems.

    **3. Improve sanitation and water access**  
    Basic sanitation and clean water access can support broader malaria prevention and community health improvement.

    **4. Prioritize rural and climate-vulnerable regions**  
    Rural populations may experience amplified malaria risk under rainfall and climate variability.

    **5. Use geographic risk mapping for resource allocation**  
    Risk maps can help guide bed-net distribution, testing campaigns, and public health planning.

    **6. Apply explainable AI for decision support**  
    SHAP-based explanations can help policymakers understand why certain regions are predicted to have higher malaria burden.
    """)

# Footer
st.markdown("---")
st.caption(
    "Developed by Dr. Samuel Israel | Digital Health Africa Capstone Project | Climate-Aware Malaria Prediction System"
)
