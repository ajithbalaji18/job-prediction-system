import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay 

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Job Acceptance Prediction System")

st.markdown(
    """
    This dashboard provides recruitment analytics and predicts
    whether a candidate is likely to accept a job offer using the
    trained Random Forest model.
    """
)

# -------------------------------
# Load Dataset and Saved Objects
# -------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("placement_feature_engineered (1).csv")

@st.cache_resource
def load_model():
    model = joblib.load("job_acceptance_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, scaler, label_encoders


df = load_data()
model, scaler, label_encoders = load_model()


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Model Performance"
    ]
)


if page == "Dashboard":

    st.header("📈 Recruitment Dashboard")

    # -----------------------
    # KPI Calculations
    # -----------------------

    dashboard_df = df.dropna(subset=["status"])

    total_candidates = len(dashboard_df)

    placed_candidates = (dashboard_df["status"] == "Placed").sum()

    placement_rate = (placed_candidates / total_candidates) * 100

    avg_interview_score = dashboard_df["interview_performance"].mean()

    avg_skills_match = dashboard_df["skills_match_percentage"].mean()

    avg_expected_ctc = dashboard_df["expected_ctc_lpa"].mean()

    most_common_experience = dashboard_df["years_of_experience"].mode()[0]

    # -----------------------
    # KPI Cards
    # -----------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Candidates", total_candidates)

    with col2:
        st.metric("Placed Candidates", placed_candidates)

    with col3:
        st.metric("Placement Rate", f"{placement_rate:.2f}%")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Avg Interview Performance", f"{avg_interview_score:.2f}")

    with col5:
        st.metric("Avg Skills Match", f"{avg_skills_match:.2f}%")

    with col6:
        st.metric("Most Common Experience", most_common_experience)

    with col1:
        st.metric("Avg Expected CTC", f"{avg_expected_ctc:.2f} LPA")


    st.markdown("---")
    st.subheader("Candidate Performance Analysis")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        data=dashboard_df,
        x="status",
        y="academic_average",
        ax=ax
    )

    ax.set_title("Academic Average vs Placement Outcome")
    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Academic Average")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.scatterplot(
        data=dashboard_df,
        x="skills_match_percentage",
        y="interview_performance",
        hue="status",
        alpha=0.7,
        ax=ax
        )

    ax.set_title("Skills Match vs Interview Performance")
    ax.set_xlabel("Skills Match (%)")
    ax.set_ylabel("Interview Performance")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("Certification Impact on Job Acceptance")

    cert_data = (
        dashboard_df
        .groupby("status")["certifications_count"]
        .mean()
        .reset_index()
        )

    fig, ax = plt.subplots(figsize=(6,4))

    sns.barplot(
        data=cert_data,
        x="status",
        y="certifications_count",
        palette="Set2",
        ax=ax
        )

    ax.set_title("Average Certifications by Placement Status")
    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Average Certifications")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Placement & Acceptance Analysis")

    placement_rate = (
        dashboard_df.groupby("company_tier")["status"]
        .apply(lambda x: (x == "Placed").mean() * 100)
        .reset_index(name="placement_rate")
        )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        data=placement_rate,
        x="company_tier",
        y="placement_rate",
        palette="viridis",
        ax=ax
    )

    ax.set_title("Placement Rate by Company Tier")
    ax.set_xlabel("Company Tier")
    ax.set_ylabel("Placement Rate (%)")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=dashboard_df,
        x="years_of_experience",
        hue="status",
        palette="Set2",
        ax=ax
        )

    ax.set_title("Experience vs Placement Success")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Number of Candidates")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Interview & Evaluation Analysis")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        data=dashboard_df,
        x="status",
        y="interview_performance",
        palette="Set3",
        ax=ax
        )

    ax.set_title("Interview Performance vs Placement Outcome")
    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Interview Performance")

    st.pyplot(fig)
    plt.close(fig)

    score_df = dashboard_df[
        ["technical_score", "aptitude_score"]
        ].mean().reset_index()

    score_df.columns = ["Test", "Average Score"]

    fig, ax = plt.subplots(figsize=(7,4))

    sns.barplot(
        data=score_df,
        x="Test",
        y="Average Score",
        palette="coolwarm",
        ax=ax
        )

    ax.set_title("Average Employability Test Scores")
    ax.set_xlabel("")
    ax.set_ylabel("Average Score")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f")

    st.pyplot(fig)
    plt.close(fig)


if page == "Model Performance":

    st.header("🤖 Model Performance")

    st.write("""
    This page summarizes the performance of the trained Random Forest model
    after hyperparameter tuning.
    """)

    st.subheader("Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "88.52%")

    with col2:
        st.metric("Precision", "85.02%")

    with col3:
        st.metric("Recall", "75.35%")

    with col4:
        st.metric("F1 Score", "79.90%")

    st.markdown("---")
    st.subheader("Classification Report")

    report_df = pd.DataFrame({
        "Class":["Not Accepted","Accepted","Overall Accuracy"],
        "Precision":[0.89,0.85,0.885],
        "Recall":[0.94,0.75,0.885],
        "F1 Score":[0.92,0.80,0.885]
    })

    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrix")

    cm = np.array([
        [6587,403],
        [748,2287]
    ])

    fig, ax = plt.subplots(figsize=(6,5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Accepted","Accepted"]
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Top 10 Important Features")

    importance_df = pd.DataFrame({
        "Feature":[
            "technical_score",
            "years_of_experience",
            "skills_match_percentage",
            "interview_performance",
            "expected_ctc_lpa",
            "previous_ctc_lpa",
            "job_role_match",
            "communication_score",
            "aptitude_score",
            "internship_experience"
        ],
        "Importance":[
            0.187987,
            0.125408,
            0.091196,
            0.087240,
            0.064597,
            0.042779,
            0.042663,
            0.034463,
            0.034344,
            0.026935
        ]
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8,6))

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.success("""
    Model Selected : Random Forest Classifier

    ✔ Hyperparameter tuning improved performance.

    Final Accuracy : 88.52%

    The most influential predictors are:

    • Technical Score
    • Years of Experience
    • Skills Match Percentage
    • Interview Performance
    • Expected CTC
    """)