import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# App title and description
st.title("E-Commerce Product Delivery Prediction App")
st.header("Dashboard")
st.write("""
This application predicts whether an e-commerce product delivery will be on time or delayed.
Upload your dataset, train the model, and make predictions.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")


if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.success("Executed Sucessfully") 
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Basic EDA - Summary statistics and data info
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("Summary Statistics")
    st.write(df.describe())

    # Identify categorical columns and encode them
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].nunique() == 2:
            df[col] = df[col].astype('category').cat.codes  # Binary encoding for two-value categorical features
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Display correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Select target and features
    target_column = "OrderStatus_Delivered"  # Change based on your target variable
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sidebar for model configuration
        st.sidebar.header("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of trees in Random Forest", 10, 200, step=10, value=100)
        max_depth = st.sidebar.slider("Maximum depth of trees", 1, 20, step=1, value=10)

        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Predictions and evaluation metrics
        y_pred = model.predict(X_test)
        st.subheader("Model Performance Metrics")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))

        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", ax=ax)
        st.pyplot(fig)

        # Feature importance
        st.subheader("Feature Importance")
        feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values(by='importance', ascending=False)
        st.bar_chart(feature_importances.set_index("feature"))

        # Input form for new prediction
        st.header("Make a Prediction on New Data")
        input_data = {feature: st.number_input(f"Enter {feature}", value=0.0) for feature in X.columns}
        input_df = pd.DataFrame([input_data])
      

        # Prediction button
        if st.button("Predict Delivery Status"):
            prediction = model.predict(input_df)[0]
            result = "On-Time" if prediction == 1 else "Delayed"
            st.write(f"Prediction: **{result}**")
    else:
        st.error(f"Target column '{target_column}' not found in the data.")
else:
    st.info("Please upload a CSV file.")


    
