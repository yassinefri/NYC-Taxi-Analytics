# =============================================================================
# NYC TAXI FARE PREDICTION - DASHBOARD KPIs
# =============================================================================
# Streamlit Dashboard for NYC Taxi Fare Prediction Model
# Authors: Ayoub REZALA & Yassine FRIKICH
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="NYC Taxi Fare Prediction - Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# LOAD DATA AND MODEL
# =============================================================================
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv("datasets/original_cleaned_nyc_taxi_data_2018.csv")
    return df


@st.cache_resource
def load_model():
    """Load the trained model"""
    model = joblib.load("fare_prediction_model.pkl")
    return model


# Load data and model
try:
    df = load_data()
    model = load_model()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Yellow_Cab_NYC.jpg/320px-Yellow_Cab_NYC.jpg",
    width=280,
)
st.sidebar.markdown("## NYC Taxi Fare Prediction")
st.sidebar.markdown("---")

# Sample size selector
sample_size = st.sidebar.slider(
    "Sample Size for Analysis",
    min_value=10000,
    max_value=100000,
    value=50000,
    step=10000,
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.markdown("**Authors:** Ayoub REZALA & Yassine FRIKICH")
st.sidebar.markdown("**Dataset:** NYC TLC 2018")
st.sidebar.markdown("**Model:** Linear Regression")

# =============================================================================
# MAIN CONTENT
# =============================================================================
st.markdown(
    '<p class="main-header">NYC Taxi Fare Prediction Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

if data_loaded:
    # Sample data
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Clean data
    df_clean = df_sample[
        (df_sample["fare_amount"] > 0)
        & (df_sample["fare_amount"] < 200)
        & (df_sample["trip_distance"] > 0)
        & (df_sample["trip_distance"] < 100)
        & (df_sample["trip_duration"] > 60)
        & (df_sample["trip_duration"] < 7200)
    ].copy()

    # =========================================================================
    # SECTION 1: KEY METRICS
    # =========================================================================
    st.markdown(
        '<p class="section-header">Key Performance Indicators (KPIs)</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Trips (Sample)",
            value=f"{len(df_clean):,}",
            delta=f"{len(df):,} total in dataset",
        )

    with col2:
        avg_fare = df_clean["fare_amount"].mean()
        st.metric(
            label="Average Fare",
            value=f"${avg_fare:.2f}",
            delta=f"Median: ${df_clean['fare_amount'].median():.2f}",
        )

    with col3:
        avg_distance = df_clean["trip_distance"].mean()
        st.metric(
            label="Average Distance",
            value=f"{avg_distance:.2f} mi",
            delta=f"Median: {df_clean['trip_distance'].median():.2f} mi",
        )

    with col4:
        avg_tip = df_clean["tip_amount"].mean()
        st.metric(
            label="Average Tip",
            value=f"${avg_tip:.2f}",
            delta=f"{(avg_tip / avg_fare) * 100:.1f}% of fare",
        )

    st.markdown("---")

    # =========================================================================
    # SECTION 2: MODEL PERFORMANCE
    # =========================================================================
    st.markdown(
        '<p class="section-header">Model Performance Metrics</p>',
        unsafe_allow_html=True,
    )

    # Prepare data for prediction
    features = [
        "trip_distance",
        "rate_code",
        "payment_type",
        "extra",
        "mta_tax",
        "tip_amount",
    ]
    X = df_clean[features]
    y = df_clean["fare_amount"]

    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="R² Score",
            value=f"{r2:.4f}",
            delta=f"{r2 * 100:.1f}% variance explained",
        )

    with col2:
        st.metric(label="RMSE", value=f"${rmse:.2f}", delta="Root Mean Squared Error")

    with col3:
        st.metric(label="MAE", value=f"${mae:.2f}", delta="Mean Absolute Error")

    st.markdown("---")

    # =========================================================================
    # SECTION 3: VISUALIZATIONS
    # =========================================================================
    st.markdown(
        '<p class="section-header">Data Visualizations</p>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        # Predictions vs Actual
        st.markdown("#### Predictions vs Actual Values")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(y, y_pred, alpha=0.3, s=10, c="#3B82F6")
        ax1.plot(
            [y.min(), y.max()],
            [y.min(), y.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        ax1.set_xlabel("Actual Fare ($)")
        ax1.set_ylabel("Predicted Fare ($)")
        ax1.set_title(f"Model Accuracy (R² = {r2:.3f})")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        # Residual Distribution
        st.markdown("#### Prediction Error Distribution")
        residuals = y - y_pred
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="#10B981")
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
        ax2.set_xlabel("Prediction Error ($)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Residual Distribution")
        ax2.legend()
        st.pyplot(fig2)

    st.markdown("---")

    # =========================================================================
    # SECTION 4: FEATURE ANALYSIS
    # =========================================================================
    st.markdown(
        '<p class="section-header">Feature Analysis</p>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        # Feature Importance (Coefficients)
        st.markdown("#### Model Coefficients")
        coef_df = pd.DataFrame(
            {"Feature": features, "Coefficient": model.coef_}
        ).sort_values("Coefficient", key=abs, ascending=True)

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        colors = ["#10B981" if c > 0 else "#EF4444" for c in coef_df["Coefficient"]]
        ax3.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
        ax3.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax3.set_xlabel("Coefficient Value")
        ax3.set_title("Feature Impact on Fare")
        st.pyplot(fig3)

    with col2:
        # Fare Distribution by Payment Type
        st.markdown("#### Fare by Payment Type")
        payment_labels = {
            1: "Credit Card",
            2: "Cash",
            3: "No Charge",
            4: "Dispute",
            5: "Unknown",
        }
        df_clean["payment_label"] = df_clean["payment_type"].map(payment_labels)

        fig4, ax4 = plt.subplots(figsize=(8, 6))
        payment_counts = df_clean["payment_label"].value_counts()
        ax4.pie(
            payment_counts.values,
            labels=payment_counts.index,
            autopct="%1.1f%%",
            colors=["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"],
        )
        ax4.set_title("Distribution by Payment Type")
        st.pyplot(fig4)

    st.markdown("---")

    # =========================================================================
    # SECTION 5: DISTRIBUTION ANALYSIS
    # =========================================================================
    st.markdown(
        '<p class="section-header">Distribution Analysis</p>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Fare Distribution")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.hist(
            df_clean["fare_amount"],
            bins=40,
            edgecolor="black",
            alpha=0.7,
            color="#3B82F6",
        )
        ax5.axvline(
            x=df_clean["fare_amount"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: ${df_clean['fare_amount'].mean():.2f}",
        )
        ax5.set_xlabel("Fare ($)")
        ax5.set_ylabel("Frequency")
        ax5.legend()
        st.pyplot(fig5)

    with col2:
        st.markdown("#### Distance Distribution")
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        ax6.hist(
            df_clean["trip_distance"],
            bins=40,
            edgecolor="black",
            alpha=0.7,
            color="#10B981",
        )
        ax6.axvline(
            x=df_clean["trip_distance"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {df_clean['trip_distance'].mean():.2f} mi",
        )
        ax6.set_xlabel("Distance (miles)")
        ax6.set_ylabel("Frequency")
        ax6.legend()
        st.pyplot(fig6)

    with col3:
        st.markdown("#### Trip Duration Distribution")
        df_clean["trip_duration_min"] = df_clean["trip_duration"] / 60
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        ax7.hist(
            df_clean["trip_duration_min"],
            bins=40,
            edgecolor="black",
            alpha=0.7,
            color="#F59E0B",
        )
        ax7.axvline(
            x=df_clean["trip_duration_min"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {df_clean['trip_duration_min'].mean():.1f} min",
        )
        ax7.set_xlabel("Duration (minutes)")
        ax7.set_ylabel("Frequency")
        ax7.legend()
        st.pyplot(fig7)

    st.markdown("---")

    # =========================================================================
    # SECTION 6: FARE PREDICTOR
    # =========================================================================
    st.markdown('<p class="section-header">Fare Predictor</p>', unsafe_allow_html=True)

    st.markdown("Enter trip details to predict the fare:")

    col1, col2, col3 = st.columns(3)

    with col1:
        pred_distance = st.number_input(
            "Trip Distance (miles)", min_value=0.1, max_value=50.0, value=3.0, step=0.5
        )
        pred_rate_code = st.selectbox(
            "Rate Code",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "Standard",
                2: "JFK",
                3: "Newark",
                4: "Nassau/Westchester",
                5: "Negotiated",
                6: "Group Ride",
            }[x],
        )

    with col2:
        pred_payment = st.selectbox(
            "Payment Type",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Credit Card",
                2: "Cash",
                3: "No Charge",
                4: "Dispute",
            }[x],
        )
        pred_extra = st.number_input(
            "Extra Charges ($)", min_value=0.0, max_value=10.0, value=0.5, step=0.5
        )

    with col3:
        pred_mta_tax = st.number_input(
            "MTA Tax ($)", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )
        pred_tip = st.number_input(
            "Expected Tip ($)", min_value=0.0, max_value=20.0, value=2.0, step=0.5
        )

    if st.button("Predict Fare", type="primary"):
        # Make prediction
        input_data = np.array(
            [
                [
                    pred_distance,
                    pred_rate_code,
                    pred_payment,
                    pred_extra,
                    pred_mta_tax,
                    pred_tip,
                ]
            ]
        )
        predicted_fare = model.predict(input_data)[0]

        st.success(f"**Predicted Fare: ${predicted_fare:.2f}**")

        # Show breakdown
        st.markdown("#### Fare Breakdown:")
        breakdown_df = pd.DataFrame(
            {
                "Component": [
                    "Base Fare (Intercept)",
                    "Distance Component",
                    "Rate Code",
                    "Payment Type",
                    "Extra Charges",
                    "MTA Tax",
                    "Tip Correlation",
                ],
                "Value": [
                    f"${model.intercept_:.2f}",
                    f"${model.coef_[0] * pred_distance:.2f}",
                    f"${model.coef_[1] * pred_rate_code:.2f}",
                    f"${model.coef_[2] * pred_payment:.2f}",
                    f"${model.coef_[3] * pred_extra:.2f}",
                    f"${model.coef_[4] * pred_mta_tax:.2f}",
                    f"${model.coef_[5] * pred_tip:.2f}",
                ],
            }
        )
        st.table(breakdown_df)

    st.markdown("---")

    # =========================================================================
    # SECTION 7: DATA SUMMARY
    # =========================================================================
    st.markdown('<p class="section-header">Dataset Summary</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Descriptive Statistics")
        stats_df = df_clean[
            ["fare_amount", "trip_distance", "trip_duration", "tip_amount"]
        ].describe()
        st.dataframe(stats_df.round(2))

    with col2:
        st.markdown("#### Correlation with Fare Amount")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        correlations = (
            df_clean[numeric_cols]
            .corr()["fare_amount"]
            .drop("fare_amount")
            .sort_values(ascending=False)
        )

        corr_df = pd.DataFrame(
            {"Variable": correlations.index, "Correlation": correlations.values}
        )
        st.dataframe(corr_df.round(4))

else:
    st.error(
        "Could not load data. Please ensure the dataset and model files are in the correct location."
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    <p>NYC Taxi Fare Prediction Dashboard | Concepts et Technologies IA</p>
    <p>Ayoub REZALA & Yassine FRIKICH | January 2026</p>
</div>
""",
    unsafe_allow_html=True,
)
