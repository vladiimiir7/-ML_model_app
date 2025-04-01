import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go


st.title(" ML Model Trainer App")

datasets = sns.get_dataset_names()
dataset_choice = st.selectbox("Choose a dataset", options=datasets + ["Upload your own"])

# Placeholder variables for uploaded file and dataframe
uploaded_file = None
df = None

# Check if user wants to upload their own dataset
if dataset_choice == "Upload your own":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Custom dataset loaded!")
else:
    try:
        df = sns.load_dataset(dataset_choice)
        st.success(f"Dataset '{dataset_choice}' loaded!")
    except:
        st.warning("Could not load selected dataset.")

# Proceed only if a dataset is loaded and show a preview of the dataset
if df is not None:
    st.subheader("Data Overview")
    st.dataframe(df.head())

    # Automatically detect numeric and categorical features
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    st.subheader("Model Configuration")

    # Use a form so model training only happens after clicking the "Train" button and let the user pick the different options
    with st.form("model_form"):
        target = st.selectbox("Target variable", options=all_cols)

        feature_candidates = [col for col in all_cols if col != target]

        selected_num_features = st.multiselect("Quantitative input features", options=[col for col in numeric_cols if col != target])
        selected_cat_features = st.multiselect("Qualitative input features", options=[col for col in categorical_cols if col != target])
        selected_features = selected_num_features + selected_cat_features

        st.markdown("### Select a model please")
        model_type = st.selectbox("Model", ["Linear Regression", "Random Forest"])

        # Train/test split configuration
        test_size = st.slider("Test size (for train/test split)", min_value=0.1, max_value=0.5, value=0.2)

        # If Random Forest is selected then show hyperparameter sliders
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 300, step=10, value=100)
            max_depth = st.slider("Max depth", 1, 20, value=5)
        else:
            # For Linear Regression there is no need for tree-based parameters
            n_estimators = None
            max_depth = None

        submitted = st.form_submit_button("Train the Model")

    # Proceed with model training only after the user submits the form
    if submitted and target and selected_features:
        df_model = df[[target] + selected_features].dropna()

        df_encoded = df_model.copy()
        le = LabelEncoder()
        for col in selected_cat_features:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        if df_encoded[target].dtype == "object":
            df_encoded[target] = le.fit_transform(df_encoded[target].astype(str))

        # Split into input features and target
        X = df_encoded[selected_features]
        y = df_encoded[target]

        # Auto-detect task type: classification is if <= 10 unique values in target, else choose regression
        task_type = "regression"
        if len(y.unique()) <= 10 and y.dtype in [np.int64, np.int32, int]:
            task_type = "classification"

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Initialize and train the chosen model
        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

        model.fit(X_train, y_train)      
        y_pred = model.predict(X_test)   

        # Show model performance
        st.subheader("Model Performance")

        # If regression: Show MSE, R², and residuals
        if task_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.metric("Mean Squared Error", round(mse, 2))
            st.metric("R² Score", round(r2, 2))

            # Plot residual distribution
            residuals = y_test - y_pred
            fig = px.histogram(residuals, nbins=30, title="Residual Distribution")
            st.plotly_chart(fig)

        # If classification: Show metrics and plots
        else:
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm)

            # If binary classification, show ROC curve
            if len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
                fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc:.2f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc)

        # Show feature importance if supported
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            fig_imp = px.bar(x=selected_features, y=importance, title="Feature Importance")
            st.plotly_chart(fig_imp)

    elif submitted:
        st.error("Please select both a target AND at least one input feature.")



