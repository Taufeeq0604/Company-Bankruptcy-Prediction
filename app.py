import pandas as pd
import joblib
import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="📊",
    layout="wide"
)


# LOAD MODEL
model = joblib.load("rf_final_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# SIDEBAR INPUTS
st.sidebar.header("Input Features")

industrial_risk = st.sidebar.slider(
    "Industrial Risk",
    0.0,
    1.0,
    0.5
)

management_risk = st.sidebar.slider(
    "Management Risk",
    0.0,
    1.0,
    0.5
)

financial_flexibility = st.sidebar.slider(
    "Financial Flexibility",
    0.0,
    1.0,
    0.5
)

credibility = st.sidebar.slider(
    "Credibility",
    0.0,
    1.0,
    0.5
)

competitiveness = st.sidebar.slider(
    "Competitiveness",
    0.0,
    1.0,
    0.5
)

operating_risk = st.sidebar.slider(
    "Operating Risk",
    0.0,
    1.0,
    0.5
)


# PREDICTION BUTTON
predict_button = st.sidebar.button("Predict Bankruptcy Risk")



# HEADER
st.title("Bankruptcy Prediction System")
st.caption(
    "Machine Learning based system to predict bankruptcy risk using company risk factors."
)



# Model Comparison
model_comparison = pd.DataFrame({
    "Model": [
        "KNN Classifier",
        "LightGBM Classifier",
        "SVM Classifier",
        "Random Forest",
        "Gradient Boosting",
        "CatBoost Classifier",
        "XGBoost Classifier",
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        0.98,
        0.98,
        1.00,
        1.00,
        0.98,
        1.00,
        0.98,
        1.00,
        0.98
    ],
    "Precision": [
        0.981,
        0.981,
        1.000,
        1.000,
        0.981,
        1.000,
        0.981,
        1.000,
        0.981
    ],
    "Recall": [
        0.98,
        0.98,
        1.00,
        1.00,
        0.98,
        1.00,
        0.98,
        1.00,
        0.98
    ],
    "F1-Score": [
        0.98,
        0.98,
        1.00,
        1.00,
        0.98,
        1.00,
        0.98,
        1.00,
        0.98
    ],
    "ROC-AUC": [
        1.000,
        1.000,
        1.000,
        1.000,
        1.000,
        1.000,
        1.000,
        1.000,
        0.983
    ]
})



# DEFAULT VALUES
prediction_label = None
no_bankruptcy_probability = 0
bankruptcy_probability = 0


# PREDICTION
if predict_button:

    input_data = {
        'industrial_risk':industrial_risk,
        'management_risk':management_risk,
        'financial_flexibility':financial_flexibility,
        'credibility':credibility,
        'competitiveness':competitiveness,
        'operating_risk':operating_risk
    }

    input_df = pd.DataFrame([input_data])


    # MODEL PREDICTION & PROBABILITY
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Convert Prediction to Label
    prediction_label = "Bankruptcy" if prediction == 1 else "No-Bankruptcy"

    # Probability Calculation
    no_bankruptcy_probability = probability[0] * 100
    bankruptcy_probability = probability[1] * 100



# =========================
# TOP SECTION
# =========================

left, right = st.columns([1.2, 1])

with left:

    st.subheader("Prediction")

    if prediction_label is not None:

        st.success(prediction_label)
        st.warning("Model Used: Random Forest")

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "No Bankruptcy",
                f"{no_bankruptcy_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Bankruptcy",
                f"{bankruptcy_probability:.2f}%"
            )

    else:

        st.info(
            "Adjust the risk factors from the sidebar and click 'Predict Bankruptcy Risk'."
        )

with right:

    st.subheader("Deployed Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Accuracy",
            "100.00%"
        )

    with col3:
        st.metric(
            "Precision",
            "100.00%"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Recall",
            "100.00%"
        )

    with col5:
        st.metric(
            "F1-Score",
            "100.00%"
        )

    with col6:
        st.metric(
            "ROC-AUC",
            "1.000"
        )

st.divider()


# MODEL COMPARISON TABLE
st.subheader("Model Performance Comparison")

st.dataframe(
    model_comparison.style.format({
        "Accuracy": "{:.2f}",
        "Precision": "{:.3f}",
        "Recall": "{:.2f}",
        "F1-Score": "{:.2f}",
        "ROC-AUC": "{:.3f}"
    }),
    use_container_width=True
)



# INPUT SUMMARY
st.subheader("Selected Risk Factors")
input_summary_df = pd.DataFrame({
    "Feature": [
        "Industrial Risk",
        "Management Risk",
        "Financial Flexibility",
        "Credibility",
        "Competitiveness",
        "Operating Risk"
    ],
    "Value": [
        industrial_risk,
        management_risk,
        financial_flexibility,
        credibility,
        competitiveness,
        operating_risk
    ]
})

st.dataframe(
    input_summary_df,
    use_container_width=True
)