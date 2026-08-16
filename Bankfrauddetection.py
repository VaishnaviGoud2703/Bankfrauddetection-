import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE


# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="🏦"
)

st.title("🏦 Bank Fraud Detection")
st.write("Machine Learning Based Fraud Detection System")


# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard_small.csv")

    # Remove rows where target is missing
    df = df.dropna(subset=["Class"])

    # Fill missing values in features
    X = df.drop("Class", axis=1)
    X = X.fillna(X.mean())

    df[X.columns] = X

    return df


# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(df):

    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    # Logistic Regression
    model = LogisticRegression(
        max_iter=5000
    )

    model.fit(
        X_train_resampled,
        y_train_resampled
    )

    # Prediction
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return (
        model,
        scaler,
        X.columns.tolist(),
        accuracy,
        precision,
        recall,
        f1
    )


# -----------------------------
# Start application
# -----------------------------
try:

    df = load_data()

    (
        model,
        scaler,
        feature_names,
        accuracy,
        precision,
        recall,
        f1
    ) = train_model(df)


    # -----------------------------
    # Model performance
    # -----------------------------
    st.success("Model trained successfully!")

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    col2.metric(
        "Precision",
        f"{precision:.2%}"
    )

    col3.metric(
        "Recall",
        f"{recall:.2%}"
    )

    col4.metric(
        "F1 Score",
        f"{f1:.2%}"
    )


    # -----------------------------
    # Dataset information
    # -----------------------------
    st.subheader("📁 Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric(
        "Transactions",
        f"{len(df):,}"
    )

    col2.metric(
        "Features",
        len(feature_names)
    )

    st.write("Transaction Class Distribution")

    st.bar_chart(
        df["Class"].value_counts()
    )


    # -----------------------------
    # Transaction prediction
    # -----------------------------
    st.subheader("🔍 Check a Transaction")

    st.write(
        "Enter the transaction feature values below."
    )

    transaction_data = {}

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        if i % 2 == 0:

            with col1:

                transaction_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f"
                )

        else:

            with col2:

                transaction_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f"
                )


    # -----------------------------
    # Prediction button
    # -----------------------------
    if st.button(
        "🚨 Check Transaction",
        type="primary"
    ):

        transaction = pd.DataFrame(
            [transaction_data],
            columns=feature_names
        )

        # Scale transaction
        transaction_scaled = scaler.transform(
            transaction
        )

        # Prediction
        prediction = model.predict(
            transaction_scaled
        )[0]

        # Fraud probability
        probability = model.predict_proba(
            transaction_scaled
        )[0][1]


        if prediction == 1:

            st.error(
                "🚨 Fraudulent Transaction Detected!"
            )

        else:

            st.success(
                "✅ Transaction Appears Legitimate."
            )


        st.write(
            f"Fraud Probability: **{probability:.2%}**"
        )


except FileNotFoundError:

    st.error(
        "❌ creditcard_small.csv was not found. "
        "Make sure it is in the same GitHub repository "
        "as Bankfrauddetection.py."
    )


except Exception as e:

    st.error(
        f"❌ Application Error: {e}"
    )
