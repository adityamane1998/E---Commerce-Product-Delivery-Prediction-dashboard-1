import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Title and Description
st.title("E-Commerce Product Delivery Prediction")
st.write("""
This app predicts whether a product delivery will be on-time or delayed using machine learning.
Upload your dataset to get started!
""")

# Upload CSV File
uploaded_file = st.file_uploader("/content/drive/MyDrive/Colab Notebooks/Capstone Project /E - Commerce_Product_Delivery_Model_Implementation_Evaluation.csv", type="csv")
uploaded_file = st.file_uploader("/content/drive/MyDrive/Colab Notebooks/Capstone Project /E - Commerce_Product_Delivery_Prediction_1.csv", type="csv")
uploaded_file = st.file_uploader("/content/drive/MyDrive/Colab Notebooks/Capstone Project /E - Commerce_Product_Delivery_Prediction_Data.csv", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(/content/drive/MyDrive/Colab Notebooks/Capstone Project /E - Commerce_Product_Delivery_Model_Implementation_Evaluation.csv)
    df = pd.read_csv(/content/drive/MyDrive/Colab Notebooks/Capstone Project /E - Commerce_Product_Delivery_Prediction_1.csv)
    df = pd.read_csv(/content/drive/MyDrive/Colab Notebooks/Capstone Project /E - Commerce_Product_Delivery_Prediction_Data.csv)
    st.write("Data Preview:", df.head())

    # Define features and target variable
    X = df.drop(columns=["OrderStatus", "OrderStatus_Delivered"])  # Features
    y = df["OrderStatus_Delivered"]  # Target variable

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    st.sidebar.header("Model Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Trees in Random Forest", 10, 200, step=10, value=100)
    max_depth = st.sidebar.slider("Maximum Depth of Trees", 1, 20, step=1, value=10)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Display Evaluation Metrics
    st.header("Model Performance Metrics")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature Importances
    st.subheader("Feature Importances")
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    st.bar_chart(feature_importances.set_index("feature"))

    # Prediction on New Data
    st.header("Predict on New Data")
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.number_input(f"Input {feature}", value=0)
    input_df = pd.DataFrame([input_data])

    if st.button("Predict Delivery Status"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]
        if prediction[0] == 1:
            st.success(f"Prediction: On-Time (Confidence: {prediction_proba[0]:.2f})")
        else:
            st.error(f"Prediction: Delayed (Confidence: {1 - prediction_proba[0]:.2f})")


    # Display Confusion Matrix
    st.header("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
