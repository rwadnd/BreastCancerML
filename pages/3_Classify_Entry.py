import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
from modules.nav import Navbar

# Import necessary sklearn modules for models and preprocessing (for applying transformers)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import preprocessing modules (needed by apply_preprocessing_to_dataframe)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, RFE, f_classif
from sklearn.decomposition import PCA
from scipy.stats import zscore, mstats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Set Streamlit page configuration for wide layout
st.set_page_config(page_title="Classify Custom Entry", layout="wide")

# Helper function to apply preprocessing transformers to a DataFrame
# This function uses the *fitted* transformers passed from session state
def apply_preprocessing_to_dataframe(df_input, original_features, fitted_transformers):
    """
    Applies a sequence of fitted preprocessing transformers to an input DataFrame.
    This is crucial for ensuring new data is processed consistently with the training data.

    Args:
        df_input (pd.DataFrame): The DataFrame to apply transformations to.
        original_features (list): List of feature names from the original dataset.
                                  Ensures the input DataFrame has the correct columns.
        fitted_transformers (dict): A dictionary containing fitted scikit-learn transformers
                                    (e.g., 'imputer', 'scaler', 'feature_selector', 'pca').

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = df_input.copy()

    # Ensure columns match original features before applying transformations.
    # This is critical if feature selection/PCA changed the number of columns.
    # For a new single input row, it's assumed to have all original features.
    df = df[original_features].copy()


    # Apply Imputation if an imputer was fitted and provided
    if 'imputer' in fitted_transformers and fitted_transformers['imputer'] is not None:
        df = pd.DataFrame(fitted_transformers['imputer'].transform(df), columns=df.columns)

    # Apply Scaling if a scaler was fitted and provided
    if 'scaler' in fitted_transformers and fitted_transformers['scaler'] is not None:
        df = pd.DataFrame(fitted_transformers['scaler'].transform(df), columns=df.columns)

    # Apply Feature Selection/Reduction if a selector or PCA was fitted
    if 'feature_selector' in fitted_transformers and fitted_transformers['feature_selector'] is not None:
        # For SelectKBest/RFE, transform and then get new column names
        df = pd.DataFrame(fitted_transformers['feature_selector'].transform(df),
                          columns=fitted_transformers['feature_selector'].get_feature_names_out(df.columns))
    elif 'pca' in fitted_transformers and fitted_transformers['pca'] is not None:
        # For PCA, transform and create new column names (PC_1, PC_2, etc.)
        df = pd.DataFrame(fitted_transformers['pca'].transform(df),
                          columns=[f'PC_{j+1}' for j in range(fitted_transformers['pca'].n_components_)])
    
    return df


def main():
    # Display the navigation bar
    Navbar()

    # Initialize session state for trained models if not already present
    if "trained_models_custom_entry" not in st.session_state:
        st.session_state.trained_models_custom_entry = {}

    # Define a list of default model types for the 9 model configuration slots
    default_model_types = [
        "Logistic Regression", "Gradient Boosting", "Decision Tree",
        "LightGBM", "Random Forest", "K-Nearest Neighbors",
        "Naive Bayes", "SVM", "XGBoost", "None" # "None" allows a slot to be empty
    ]

    # Initialize models configuration for this page if not present
    # This ensures consistent model selections across page reloads
    if "models_custom_entry" not in st.session_state:
        st.session_state.models_custom_entry = []
        # Assign default model types to slots
        for i in range(9):
            # Set the last two models (slots 8 and 9, which are indices 7 and 8) to "None"
            if i >= 7: # For slots with index 7 (8th slot) and 8 (9th slot)
                model_type_to_assign = "None"
            else:
                model_type_to_assign = default_model_types[i]
            st.session_state.models_custom_entry.append({"type": model_type_to_assign, "params": {}})


    # Load dataset from session state
    if "data" in st.session_state:
        data = st.session_state.data
        st.write("Data is loaded from Kaggle Hub. Shape:", data.shape)
    else:
        st.warning("Data not found. Please return to the \"Intro\" page to load it.")
        return # Exit if data is not loaded


    st.title("Train Models & Classify Custom Entry")
    # Identify feature columns for input sliders, excluding ID, diagnosis, and unnamed columns
    features_for_input = [col for col in data.columns if col not in ['id', 'diagnosis', 'Unnamed: 32']]

    # --- Custom Feature Input Section ---
    st.subheader("Enter Custom Feature Values:")

    # Function to select a random row from the dataset and update slider values
    def use_random_entry():
        if data is not None and not data.empty:
            random_row_index = np.random.randint(0, len(data))
            random_entry = data.iloc[random_row_index]

            # Update session state for each slider with values from the random row
            for feature in features_for_input:
                st.session_state[f"input_slider_{feature}"] = float(random_entry[feature])

            # Store the selected row's info for display
            st.session_state.selected_random_row_index = random_row_index
            st.session_state.selected_random_row_diagnosis = "Malignant" if random_entry['diagnosis'] == 1 else "Benign"
        else:
            st.warning("No data available to select a random entry.")

    # Button to trigger loading a random entry
    if st.button("🎲 Load a random entry from the dataset"):
        use_random_entry()
        st.session_state.classify_now = True # Trigger classification after random entry selection

    # Display information about the selected random entry if available
    if "selected_random_row_index" in st.session_state:
        st.info(f"Last Loaded Row Index: **{st.session_state.selected_random_row_index}** (Actual Label: **{st.session_state.selected_random_row_diagnosis}**)")


    # Divide features into 3 groups (mean, se, worst) for display in 3 columns
    # Assumes features are ordered by these groups.
    feature_groups = [features_for_input[i:i + 10] for i in range(0, len(features_for_input), 10)]
    feature_values = {} # Dictionary to store user's input values from sliders

    # Create 3 columns for feature input sliders with explicit spacing
    cols_input_sliders = st.columns([1, 0.05, 1, 0.05, 1]) # Added small spacer columns for visual separation

    # Map content to the correct column indices (0, 2, 4)
    content_col_indices = [0, 2, 4]

    for idx, group in enumerate(feature_groups):
        with cols_input_sliders[content_col_indices[idx]]: # Place content in the designated content column
            column_value = "mean" if idx == 0 else "se" if idx == 1 else "worst"
            st.markdown(f"**Feature Set ({column_value})**")
            for feature in group:
                # Get min, max, and mean values from the dataset for slider range and default
                min_val = float(data[feature].min() - 0.1 * data[feature].std()) # Extend range slightly
                max_val = float(data[feature].max() + 0.1 * data[feature].std()) # Extend range slightly
                mean_val = float(data[feature].mean())

                # Use value from session state if set by random entry, otherwise use mean as default
                current_slider_value = st.session_state.get(f"input_slider_{feature}", mean_val)

                # Create a slider for each feature
                feature_values[feature] = st.slider(
                    f"{feature} ",
                    min_val,
                    max_val,
                    current_slider_value,
                    key=f"input_slider_{feature}" # Unique key for each slider
                )

    # Convert the collected feature values into a Pandas DataFrame for prediction
    input_df_raw = pd.DataFrame([feature_values])
    # Ensure the order of columns matches the original features
    input_df_raw = input_df_raw[features_for_input]

    st.markdown("---")

    # --- Model Configuration Grid ---
    st.subheader("Configure Models")

    # Define the total number of model slots (3x3 grid = 9 slots)
    num_model_slots = 9

    # Create the 3x3 grid layout for model configuration
    for i in range(math.ceil(num_model_slots / 3)): # Loop for rows
        # Use explicit widths for columns to add space between them
        row_cols = st.columns([1, 0.05, 1, 0.05, 1]) # Added small spacer columns
        content_col_indices = [0, 2, 4] # Indices for the actual content columns

        for j in range(3): # Loop for columns within a row
            idx = i * 3 + j # Calculate the current model slot index
            if idx < num_model_slots: # Ensure we don't go beyond 9 slots
                with row_cols[content_col_indices[j]]: # Place content in the designated content column
                    st.markdown(f"#### Model Slot {idx+1}")

                    # Selectbox for choosing the model type for this slot
                    model_type = st.selectbox(
                        f"Model Type {idx+1}",
                        options=default_model_types,
                        index=default_model_types.index(st.session_state.models_custom_entry[idx]["type"])
                            if st.session_state.models_custom_entry[idx]["type"] in default_model_types else 0,
                        key=f"model_type_slot_{idx}"
                    )

                    model_params = {} # Dictionary to store model-specific parameters for this slot

                    # Conditional parameter inputs based on selected model type
                    if model_type == "Logistic Regression":
                        C = st.select_slider(f"C ", options=[0.01, 0.1, 1, 10, 100], value=1, key=f"C_{idx}")
                        penalty = st.selectbox(f"Penalty ", ["l2", "l1", "elasticnet"], key=f"penalty_{idx}")
                        # Solver options depend on the selected penalty
                        if penalty == 'l1': available_solvers = ["liblinear", "saga"]
                        elif penalty == 'elasticnet': available_solvers = ["saga"]
                        elif penalty == 'none': available_solvers = ["lbfgs", "saga"]
                        else: available_solvers = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
                        solver = st.selectbox(f"Solver ",available_solvers, key=f"solver_{idx}")
                        fit_intercept = st.checkbox(f"Fit Intercept ", value=True, key=f"fit_int_{idx}")
                        class_weight = st.selectbox(f"Class Weight ", [None, "balanced"], key=f"lw_{idx}")
                        l1_ratio = 1 if penalty == 'elasticnet' else None # l1_ratio only relevant for elasticnet
                        model_params = {"C": C, "penalty": penalty, "solver": solver, "max_iter": 1000, "fit_intercept": fit_intercept, "class_weight": class_weight, "l1_ratio": l1_ratio}

                    elif model_type == "Random Forest":
                        n_estimators = st.slider(f"Trees ", 10, 500, step=10, value=100, key=f"n_slot_{idx}")
                        max_depth = st.slider(f"Max Depth ", 5, 500, step=10, key=f"depth_slot_{idx}")
                        criterion = st.selectbox(f"Criterion ", ["gini", "entropy", "log_loss"], key=f"crit_slot_{idx}")
                        min_samples_split = st.slider(f"Min Samples Split ", 2, 20, value=2, key=f"min_split_slot_{idx}")
                        min_samples_leaf = st.slider(f"Min Samples Leaf ", 1, 20, value=1, key=f"min_leaf_slot_{idx}")
                        max_features = st.selectbox(f"Max Features ", ["sqrt", "log2", None], key=f"max_feat_slot_{idx}")
                        bootstrap = st.checkbox(f"Bootstrap ", value=True, key=f"bootstrap_slot_{idx}")
                        model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "criterion": criterion,
                                        "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
                                        "max_features": max_features, "bootstrap": bootstrap, "random_state": 42}

                    elif model_type == "SVM":
                        st.warning("If you are on the cloud, SVM models can take long time to train. Apply feature scaling to make it a lot faster")
                        C = st.select_slider(f"SVM C ", options=[0.1, 0.5, 1, 5,10], value=0.1, key=f"svmC_{idx}")
                        kernel = st.selectbox(f"Kernel ", ["linear", "rbf", "poly", "sigmoid"], key=f"kernel_{idx}")
                        gamma = st.selectbox(f"Gamma ", ["auto", "scale"], key=f"gamma_{idx}")
                        degree = st.slider(f"Degree (for poly) ", 2, 5, value=3, key=f"degree_{idx}", disabled=(kernel != "poly"))
                        coef0 = st.slider(f"Coef0 (for poly, sigmoid) ", -10.0, 10.0, value=0.0, step=0.1, key=f"coef0_{idx}", disabled=(kernel not in ["poly", "sigmoid"]))
                        shrinking = st.checkbox(f"Shrinking ", value=True, key=f"shrinking_{idx}")
                        model_params = {"C": C, "kernel": kernel, "gamma": gamma, "degree": degree, "coef0": coef0, "shrinking": shrinking}

                    elif model_type == "Gradient Boosting":
                        n_estimators = st.slider(f"GB Trees ", 50, 300, step=50, value=100, key=f"gb_n_slot_{idx}")
                        learning_rate = st.select_slider(f"GB Learning Rate ", options=[0.01, 0.05, 0.1, 0.2], value=0.1, key=f"gb_lr_slot_{idx}")
                        max_depth = st.slider(f"GB Max Depth ", 3, 8, step=1, key=f"gb_depth_{idx}")
                        subsample = st.select_slider(f"GB Subsample ", options=[0.7, 0.8, 0.9, 1.0], value=1.0, key=f"gb_subsample_slot_{idx}")
                        model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample, "random_state": 42}

                    elif model_type == "K-Nearest Neighbors":
                        n_neighbors = st.slider(f"KNN Neighbors ", 1, 20, value=5, key=f"knn_n_slot_{idx}")
                        weights = st.selectbox(f"KNN Weights ", ["uniform", "distance"], key=f"knn_weights_slot_{idx}")
                        algorithm = st.selectbox(f"KNN Algorithm ", ["auto", "ball_tree", "kd_tree", "brute"], key=f"knn_algo_slot_{idx}")
                        model_params = {"n_neighbors": n_neighbors, "weights": weights, "algorithm": algorithm}

                    elif model_type == "Naive Bayes":
                        var_smoothing = st.select_slider(f"NB Var Smoothing ", options=[1e-10, 1e-9, 1e-8, 1e-7], value=1e-9, key=f"nb_smooth_slot_{idx}")
                        model_params = {"var_smoothing": var_smoothing}

                    elif model_type == "Decision Tree":
                        max_depth = st.slider(f"DT Max Depth ", 5, 50, step=1, key=f"dt_depth_{idx}")
                        min_samples_split = st.slider(f"Min Samples Split ", 2, 20, value=2, key=f"min_split_slot_{idx}")
                        min_samples_leaf = st.slider(f"Min Samples Leaf ", 1, 20, value=1, key=f"min_leaf_slot_{idx}")
                        criterion = st.selectbox(f"Criterion ", ["gini", "entropy", "log_loss"], key=f"dt_crit_slot_{idx}")
                        model_params = {"max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "criterion": criterion, "random_state": 42}

                    elif model_type == "XGBoost":
                        st.warning("If you are on the cloud, XGBoost models can take long time to train. Apply feature scaling to make it a lot faster")
                        n_estimators = st.slider(f"XGB Trees ", 50, 300, step=50, value=100, key=f"xgb_n_slot_{idx}")
                        learning_rate = st.select_slider(f"XGB Learning Rate ", options=[0.01, 0.05, 0.1, 0.2], value=0.1, key=f"xgb_lr_slot_{idx}")
                        max_depth = st.slider(f"XGB Max Depth ", 3, 10, value=6, key=f"xgb_depth_slot_{idx}")
                        subsample = st.select_slider(f"XGB Subsample ", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=1.0, key=f"xgb_subsample_slot_{idx}")
                        colsample_bytree = st.select_slider(f"XGB Colsample ", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=1.0, key=f"xgb_colsample_slot_{idx}")
                        gamma = st.select_slider(f"XGB Gamma ", options=[0, 0.1, 0.2, 0.4], value=0, key=f"xgb_gamma_slot_{idx}")
                        model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth,
                                        "subsample": subsample, "colsample_bytree": colsample_bytree, "gamma": gamma,
                                        "use_label_encoder": False, "eval_metric": "logloss", "random_state": 42}

                    elif model_type == "LightGBM":
                        n_estimators = st.slider(f"LGBM Trees ", 50, 300, step=50, value=100, key=f"lgbm_n_slot_{idx}")
                        learning_rate = st.select_slider(f"LGBM Learning Rate ", options=[0.01, 0.05, 0.1, 0.2], value=0.1, key=f"lgbm_lr_slot_{idx}")
                        num_leaves = st.slider(f"LGBM Num Leaves ", 20, 60, value=31, key=f"lgbm_leaves_slot_{idx}")
                        max_depth = st.slider(f"LGBM Max Depth ", -1, 10, value=-1, key=f"lgbm_depth_slot_{idx}")
                        reg_alpha = st.select_slider(f"LGBM L1 Reg ", options=[0, 0.1, 0.5, 1], value=0, key=f"lgbm_reg_a_slot_{idx}")
                        reg_lambda = st.select_slider(f"LGBM L2 Reg ", options=[0, 0.1, 0.5, 1], value=0, key=f"lgbm_reg_l_slot_{idx}")
                        model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "num_leaves": num_leaves,
                                        "max_depth": max_depth, "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,
                                        "random_state": 42}


                    # Store the configured model type and parameters for this slot in session state
                    st.session_state.models_custom_entry[idx] = {"type": model_type, "params": model_params}

        # Add a horizontal separator after each row of model configurations
        st.markdown("---")

    # --- Sidebar for Data Source Selection and Action Button ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("Data Source")
        # Radio button to select data source for model training
        data_source_option = st.radio(
            "Select data for training models:",
            ("Original Data", "Preprocessed Data"),
            key="data_source_selector"
        )

        # Determine if the train button should be disabled based on preprocessed data availability
        train_button_disabled = False
        if data_source_option == "Preprocessed Data":
            if "preprocessed_X" not in st.session_state or st.session_state.preprocessed_X is None or st.session_state.preprocessed_X.empty:
                st.warning("Preprocessed data not found or is empty. Please select at least one preprocessing step from the \"ML Models\" page.")
                train_button_disabled = True
            else:
                st.info(f"Preprocessed Data Shape: {st.session_state.preprocessed_X.shape}")

        st.markdown("---")
        st.subheader("Action")
        # Button to trigger training all configured models and classifying the custom entry
        train_and_classify_button = st.button("🏁 Train All Models & Classify Custom Entry", disabled=train_button_disabled)

    # --- Training and Prediction Logic (triggered by the single button) ---
    if train_and_classify_button:
        with st.spinner("Training all configured models and classifying custom entry... This might take a moment."):
            st.session_state.trained_models_custom_entry = {} # Clear previous trained models

            # Prepare the base data for training
            X_train_data_raw = data.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors='ignore')
            y_train_target = data["diagnosis"]

            # Initialize variables for data used in model training and for storing fitted transformers
            X_data_for_model_training = None
            y_data_for_model_training = None
            data_source_type_for_model_storage = data_source_option.lower().replace(" ", "_") # 'original_data' or 'preprocessed_data'
            fitted_transformers_for_model_storage = {} # Stores transformers if preprocessing is applied locally here

            if data_source_option == "Original Data":
                X_data_for_model_training = X_train_data_raw.copy()
                y_data_for_model_training = y_train_target.copy()
            else: # "Preprocessed Data" selected
                # Retrieve preprocessing configuration from the "Compare Models" page's session state
                if "preprocessing_config_applied" not in st.session_state or not st.session_state.preprocessing_config_applied:
                    st.error("Preprocessed data is selected but no preprocessing configuration found from 'Compare Models' page. Cannot apply transformations.")
                    st.session_state.classify_now = False
                    return # Stop execution

                # Apply preprocessing steps to the raw data to prepare it for training models on this page
                X_processed_for_training = X_train_data_raw.copy()
                y_processed_for_training = y_train_target.copy()
                preprocessing_config_from_compare = st.session_state.preprocessing_config_applied

                # Manually apply preprocessing steps to X_processed_for_training and capture fitted transformers
                # Outlier Removal (applied directly as it changes sample size)
                if 'outlier_removal' in preprocessing_config_from_compare:
                    if preprocessing_config_from_compare['outlier_removal'] == "IQR":
                        Q1 = X_processed_for_training.quantile(0.25)
                        Q3 = X_processed_for_training.quantile(0.75)
                        IQR = Q3 - Q1
                        mask = ~((X_processed_for_training < (Q1 - 1.5 * IQR)) | (X_processed_for_training > (Q3 + 1.5 * IQR))).any(axis=1)
                        X_processed_for_training, y_processed_for_training = X_processed_for_training[mask], y_processed_for_training[mask]
                    elif preprocessing_config_from_compare['outlier_removal'] == "Z-Score":
                        numeric_cols = X_processed_for_training.select_dtypes(include=np.number).columns
                        z_scores = np.abs(zscore(X_processed_for_training[numeric_cols]))
                        mask = (z_scores < 3).all(axis=1)
                        X_processed_for_training, y_processed_for_training = X_processed_for_training[mask], y_processed_for_training[mask]
                    elif preprocessing_config_from_compare['outlier_removal'] == "Winsorization":
                        lower_bound, upper_bound = preprocessing_config_from_compare['winsor_bounds']
                        X_processed_for_training = X_processed_for_training.apply(lambda col: mstats.winsorize(col, limits=(lower_bound, 1 - upper_bound)), axis=0)
                        X_processed_for_training = pd.DataFrame(X_processed_for_training, columns=X_train_data_raw.columns)

                # Missing Value Imputation
                if 'missing_values' in preprocessing_config_from_compare and X_processed_for_training.isnull().sum().sum() > 0:
                    if preprocessing_config_from_compare['missing_values'] == "Drop Rows":
                        combined_df = pd.concat([X_processed_for_training, y_processed_for_training], axis=1)
                        combined_df.dropna(inplace=True)
                        X_processed_for_training = combined_df.drop(columns=["diagnosis"])
                        y_processed_for_training = combined_df["diagnosis"]
                    elif preprocessing_config_from_compare['missing_values'] == "KNN Imputation":
                        imputer = KNNImputer(n_neighbors=preprocessing_config_from_compare['knn_imputer_neighbors'])
                        X_processed_for_training = pd.DataFrame(imputer.fit_transform(X_processed_for_training), columns=X_processed_for_training.columns)
                        fitted_transformers_for_model_storage['imputer'] = imputer
                    else: # Mean/Median Imputation
                        strategy = "mean" if preprocessing_config_from_compare['missing_values'] == "Mean Imputation" else "median"
                        imputer = SimpleImputer(strategy=strategy)
                        X_processed_for_training = pd.DataFrame(imputer.fit_transform(X_processed_for_training), columns=X_processed_for_training.columns)
                        fitted_transformers_for_model_storage['imputer'] = imputer

                # Feature Scaling
                if 'scaling' in preprocessing_config_from_compare:
                    if preprocessing_config_from_compare['scaling'] == "Standard Scaler":
                        scaler = StandardScaler()
                    elif preprocessing_config_from_compare['scaling'] == "Min-Max Scaler":
                        scaler = MinMaxScaler()
                    elif preprocessing_config_from_compare['scaling'] == "Robust Scaler":
                        scaler = RobustScaler()
                    X_processed_for_training = pd.DataFrame(scaler.fit_transform(X_processed_for_training), columns=X_processed_for_training.columns)
                    fitted_transformers_for_model_storage['scaler'] = scaler

                # Feature Selection/Reduction
                if 'feature_selection' in preprocessing_config_from_compare:
                    if preprocessing_config_from_compare['feature_selection'] == "SelectKBest":
                        selector = SelectKBest(score_func=f_classif, k=preprocessing_config_from_compare['k_features'])
                        X_transformed = selector.fit_transform(X_processed_for_training, y_processed_for_training)
                        X_processed_for_training = pd.DataFrame(X_transformed, columns=selector.get_feature_names_out(X_processed_for_training.columns))
                        fitted_transformers_for_model_storage['feature_selector'] = selector
                    elif preprocessing_config_from_compare['feature_selection'] == "RFE":
                        base_model_rfe = LogisticRegression(max_iter=1000)
                        selector = RFE(base_model_rfe, n_features_to_select=preprocessing_config_from_compare['k_features'])
                        X_transformed = selector.fit_transform(X_processed_for_training, y_processed_for_training)
                        X_processed_for_training = pd.DataFrame(X_transformed, columns=selector.get_feature_names_out(X_processed_for_training.columns))
                        fitted_transformers_for_model_storage['feature_selector'] = selector
                    elif preprocessing_config_from_compare['feature_selection'] == "PCA":
                        pca = PCA(n_components=preprocessing_config_from_compare['pca_components'])
                        X_transformed = pca.fit_transform(X_processed_for_training)
                        X_processed_for_training = pd.DataFrame(X_transformed, columns=[f'PC_{j+1}' for j in range(pca.n_components_)])
                        fitted_transformers_for_model_storage['pca'] = pca

                # Handle Class Imbalance
                if 'imbalance_handling' in preprocessing_config_from_compare and preprocessing_config_from_compare['imbalance_handling'] != "None":
                    if preprocessing_config_from_compare['imbalance_handling'] == "Oversampling (SMOTE)":
                        oversampler = SMOTE(random_state=42)
                        X_processed_for_training, y_processed_for_training = oversampler.fit_resample(X_processed_for_training, y_processed_for_training)
                    elif preprocessing_config_from_compare['imbalance_handling'] == "Undersampling (Random)":
                        undersampler = RandomUnderSampler(random_state=42)
                        X_processed_for_training, y_processed_for_training = undersampler.fit_resample(X_processed_for_training, y_processed_for_training)

                X_data_for_model_training = X_processed_for_training
                y_data_for_model_training = y_processed_for_training


            # Loop through each model slot to train and store the model
            for i in range(num_model_slots):
                config = st.session_state.models_custom_entry[i]
                if not config or config['type'] == "None": # Skip if slot is empty or set to "None"
                    continue

                # Instantiate the classifier model based on user selection
                model = None
                if config['type'] == "Logistic Regression": model = LogisticRegression(**config['params'])
                elif config['type'] == "Random Forest": model = RandomForestClassifier(**config['params'])
                elif config['type'] == "SVM": model = SVC(probability=True, **config['params'])
                elif config['type'] == "Gradient Boosting": model = GradientBoostingClassifier(**config['params'])
                elif config['type'] == "K-Nearest Neighbors": model = KNeighborsClassifier(**config['params'])
                elif config['type'] == "Naive Bayes": model = GaussianNB(**config['params'])
                elif config['type'] == "Decision Tree": model = DecisionTreeClassifier(**config['params'])
                elif config['type'] == "XGBoost": model = XGBClassifier(**config['params'])
                elif config['type'] == "LightGBM": model = LGBMClassifier(**config['params'])

                if model is not None:
                    try:
                        # Train the model using the prepared data
                        st.info(model)
                        model.fit(X_data_for_model_training, y_data_for_model_training)

                        # Store the trained model and associated information in session state
                        st.session_state.trained_models_custom_entry[f"Model Slot {i+1}: {config['type']}"] = {
                            'model': model,
                            'original_features': features_for_input, # Always store original feature names for consistent input
                            'model_params': config['params'],
                            'data_source_type': data_source_type_for_model_storage, # Store which data type was used for training
                            'fitted_transformers': fitted_transformers_for_model_storage # Store fitted transformers for new predictions
                        }

                    except Exception as e:
                        st.error(f"Error training Model Slot {i+1} ({config['type']}): {e}")
                        st.warning("Please check model parameters and data consistency. If using preprocessed data, ensure it's available from 'Compare Models' page.")

            st.success("All configured models have been trained!")
            st.session_state.classify_now = True # Trigger classification display after training


    # --- Display Classification Results in Model Boxes ---
    # This section runs after models are trained and 'classify_now' is True
    if "classify_now" in st.session_state and st.session_state.classify_now:
        st.markdown("---")
        st.subheader("Custom Entry Classification Results")

        # Create a grid for displaying results from each model slot
        for i in range(math.ceil(num_model_slots / 3)):
            row_cols = st.columns([1, 0.05, 1, 0.05, 1])
            content_col_indices = [0, 2, 4]

            for j in range(3):
                idx = i * 3 + j
                if idx < num_model_slots:
                    with row_cols[content_col_indices[j]]:
                        model_type = st.session_state.models_custom_entry[idx]['type']
                        trained_model_key = f"Model Slot {idx+1}: {model_type}"

                        st.markdown(f"#### Model Slot {idx+1}: {model_type}")

                        with st.container(border=True):
                            # Check if the model for this slot was successfully trained
                            if trained_model_key in st.session_state.trained_models_custom_entry:
                                stored_model_info = st.session_state.trained_models_custom_entry[trained_model_key]
                                model_for_display = stored_model_info['model']
                                data_source_type_used = stored_model_info['data_source_type']

                                # Prepare input for prediction based on data source type used during training
                                input_for_prediction_display = None
                                if data_source_type_used == 'original_data':
                                    # If trained on original data, use raw input directly
                                    input_for_prediction_display = input_df_raw[features_for_input]
                                else: # 'preprocessed_data'
                                    # Apply the same preprocessing steps to the custom input using stored transformers
                                    fitted_transformers_used = stored_model_info['fitted_transformers']
                                    if fitted_transformers_used:
                                        try:
                                            input_for_prediction_display = apply_preprocessing_to_dataframe(
                                                input_df_raw,
                                                stored_model_info['original_features'],
                                                fitted_transformers_used
                                            )
                                        except Exception as e_prep:
                                            st.error(f"Error preprocessing custom input for {model_type}: {e_prep}")
                                            st.info("Ensure preprocessing steps on 'Compare Models' page were compatible with single-row input.")
                                            input_for_prediction_display = None
                                    else:
                                        # If no transformers were fitted (e.g., only outlier removal or resampling),
                                        # the raw input is used as the preprocessed input directly.
                                        input_for_prediction_display = input_df_raw[features_for_input]

                                # Perform prediction if input is ready
                                if input_for_prediction_display is not None:
                                    try:
                                        prediction = model_for_display.predict(input_for_prediction_display)[0]
                                        prediction_label = "Malignant" if prediction == 1 else "Benign"

                                        # Display prediction and confidence if available
                                        if hasattr(model_for_display, 'predict_proba'):
                                            probabilities = model_for_display.predict_proba(input_for_prediction_display)[0]
                                            confidence = probabilities[prediction] * 100
                                            if prediction == 1:
                                                st.error(f"Prediction: **{prediction_label}**")
                                            elif prediction == 0:
                                                st.success(f"Prediction: **{prediction_label}**")
                                            st.info(f"Confidence: {confidence:.2f}%")
                                        else:
                                            st.success(f"Prediction: **{prediction_label}**")
                                            st.info("Confidence (probability) not available for this model type.")
                                    except Exception as e:
                                        st.warning(f"Could not classify with Model {model_type}: {e}")
                                        st.info("Please re-train the models if you changed input features or parameters. Check logs for details.")
                                else:
                                    st.warning("Prediction skipped due to preprocessing error.")
                            else:
                                st.info("No model trained in this slot.")
                                st.markdown("Prediction: N/A")
                                st.markdown("Confidence: N/A")
        st.session_state.classify_now = False # Reset flag after displaying results

if __name__ == "__main__":
    main()