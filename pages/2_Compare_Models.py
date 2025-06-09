import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.nav import Navbar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from scipy.stats import zscore, mstats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def main():
    Navbar()

    # Sidebar buttons to manage models
    with st.sidebar:
        st.header("Manage Models")
        if "models" not in st.session_state:
            st.session_state.models = [{}]

        if st.button("➕ Add Model"):
            if len(st.session_state.models) < 4:
                st.session_state.models.append({})
            else:
                st.warning("🚫 Maximum of 4 models allowed.")
        if st.button("🗑️ Delete Last Model") and len(st.session_state.models) > 1:
            st.session_state.models.pop()

        run_all = st.button("🏁 Run All Models")

    # Load dataset from session
    if "data" in st.session_state:
        data = st.session_state.data
        st.write("Data is loaded from Kaggle Hub. Shape:", data.shape)
    else:
        st.warning("Data not found. Please return to the \"Intro\" page to load it.")
        return # Exit if data is not loaded
    
    st.title("ML Models")

    # Prepare data for preprocessing. It will be modified in place.
    df = data.copy()
    X_original = df.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors='ignore')
    y_original = df["diagnosis"]
    X_processed = X_original.copy()
    y_processed = y_original.copy()
    fitted_preprocessing_transformers = {}
    
    st.subheader("🔄 Preprocessing Options")
    preprocessing_config = {}

    if st.checkbox("Remove Outliers", key="remove_outliers_cb"):
        outlier_method = st.selectbox("Select outlier removal method", ["IQR", "Z-Score", "Winsorization"], key="outlier")
        preprocessing_config['outlier_removal'] = outlier_method
        if outlier_method == "Winsorization":
            lower_bound = st.slider("Winsorization Lower Bound (quantile)", 0.0, 0.1, value=0.01, step=0.005, key="winsor_lower")
            upper_bound = st.slider("Winsorization Upper Bound (quantile)", 0.9, 1.0, value=0.99, step=0.005, key="winsor_upper")
            preprocessing_config['winsor_bounds'] = (lower_bound, upper_bound)

    if st.checkbox("Handle Missing Values"):
        missing_counts = X_processed.isnull().sum().sum()
        if missing_counts == 0:
            st.success("✅ No missing values found in the dataset.")
        else:
            missing_method = st.selectbox("Select missing value strategy", ["Mean Imputation", "Median Imputation", "Drop Rows", "KNN Imputation"], key="missing")
            preprocessing_config['missing_values'] = missing_method
            if missing_method == "KNN Imputation":
                knn_imputer_neighbors = st.slider("KNN Imputer Neighbors", 1, 10, value=5, key="knn_impute_n")
                preprocessing_config['knn_imputer_neighbors'] = knn_imputer_neighbors

    if st.checkbox("Apply Feature Scaling"):
        scaler_method = st.selectbox("Select scaler", ["Min-Max Scaler", "Standard Scaler", "Robust Scaler"], key="scaling")
        preprocessing_config['scaling'] = scaler_method

    if st.checkbox("Apply Feature Selection/Reduction"):
        selection_method = st.selectbox("Select feature selection/reduction method", ["SelectKBest", "RFE", "PCA"], key="feature_selection")
        preprocessing_config['feature_selection'] = selection_method
        num_features_in_data = X_processed.shape[1]
        if selection_method in ["SelectKBest", "RFE"]:
            default_k = min(10, num_features_in_data)
            k_features = st.slider(f"Number of features (k) for {selection_method}", 1, num_features_in_data, value=default_k, key=f"k_features_{selection_method}")
            preprocessing_config['k_features'] = k_features
        elif selection_method == "PCA":
            default_components = min(5, num_features_in_data)
            n_components = st.slider("Number of components for PCA", 1, num_features_in_data, value=default_components, key="pca_components")
            preprocessing_config['pca_components'] = n_components

    if st.checkbox("Handle Class Imbalance"):
        imbalance_method = st.selectbox("Select imbalance handling method", ["Oversampling (SMOTE)", "Undersampling (Random)"], key="imbalance_method")
        preprocessing_config['imbalance_handling'] = imbalance_method

    # --- Apply preprocessing immediately based on selections ---
    if preprocessing_config:
        with st.spinner("Applying preprocessing steps..."):
            # Apply Outlier Removal
            if 'outlier_removal' in preprocessing_config:
                if preprocessing_config['outlier_removal'] == "IQR":
                    Q1 = X_processed.quantile(0.25)
                    Q3 = X_processed.quantile(0.75)
                    IQR = Q3 - Q1
                    mask = ~((X_processed < (Q1 - 1.5 * IQR)) | (X_processed > (Q3 + 1.5 * IQR))).any(axis=1)
                    X_processed, y_processed = X_processed[mask], y_processed[mask]
                elif preprocessing_config['outlier_removal'] == "Z-Score":
                    numeric_cols = X_processed.select_dtypes(include=np.number).columns
                    z_scores = np.abs(zscore(X_processed[numeric_cols]))
                    mask = (z_scores < 3).all(axis=1)
                    X_processed, y_processed = X_processed[mask], y_processed[mask]
                elif preprocessing_config['outlier_removal'] == "Winsorization":
                    lower_bound, upper_bound = preprocessing_config['winsor_bounds']
                    X_processed = X_processed.apply(lambda col: mstats.winsorize(col, limits=(lower_bound, 1 - upper_bound)), axis=0)
                    X_processed = pd.DataFrame(X_processed, columns=X_original.columns)

            # Apply Missing Value Imputation
            if 'missing_values' in preprocessing_config:
                if X_processed.isnull().sum().sum() > 0:
                    if preprocessing_config['missing_values'] == "Drop Rows":
                        combined_df = pd.concat([X_processed, y_processed], axis=1)
                        combined_df.dropna(inplace=True)
                        X_processed = combined_df.drop(columns=["diagnosis"])
                        y_processed = combined_df["diagnosis"]
                    elif preprocessing_config['missing_values'] == "KNN Imputation":
                        imputer = KNNImputer(n_neighbors=preprocessing_config['knn_imputer_neighbors'])
                        X_processed = pd.DataFrame(imputer.fit_transform(X_processed), columns=X_processed.columns)
                        fitted_preprocessing_transformers['imputer'] = imputer
                    else:
                        strategy = "mean" if preprocessing_config['missing_values'] == "Mean Imputation" else "median"
                        imputer = SimpleImputer(strategy=strategy)
                        X_processed = pd.DataFrame(imputer.fit_transform(X_processed), columns=X_processed.columns)
                        fitted_preprocessing_transformers['imputer'] = imputer

            # Apply Feature Scaling
            if 'scaling' in preprocessing_config:
                if preprocessing_config['scaling'] == "Standard Scaler": scaler = StandardScaler()
                elif preprocessing_config['scaling'] == "Min-Max Scaler": scaler = MinMaxScaler()
                elif preprocessing_config['scaling'] == "Robust Scaler": scaler = RobustScaler()
                X_processed = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns)
                fitted_preprocessing_transformers['scaler'] = scaler

            # Apply Feature Selection/Reduction
            if 'feature_selection' in preprocessing_config:
                if preprocessing_config['feature_selection'] == "SelectKBest":
                    selector = SelectKBest(score_func=f_classif, k=preprocessing_config['k_features'])
                    X_transformed = selector.fit_transform(X_processed, y_processed)
                    selected_features = X_processed.columns[selector.get_support()]
                    X_processed = pd.DataFrame(X_transformed, columns=selected_features)
                    fitted_preprocessing_transformers['feature_selector'] = selector
                elif preprocessing_config['feature_selection'] == "RFE":
                    base_model_rfe = LogisticRegression(max_iter=1000)
                    selector = RFE(base_model_rfe, n_features_to_select=preprocessing_config['k_features'])
                    X_transformed = selector.fit_transform(X_processed, y_processed)
                    selected_features = X_processed.columns[selector.get_support()]
                    X_processed = pd.DataFrame(X_transformed, columns=selected_features)
                    fitted_preprocessing_transformers['feature_selector'] = selector
                elif preprocessing_config['feature_selection'] == "PCA":
                    pca = PCA(n_components=preprocessing_config['pca_components'])
                    X_transformed = pca.fit_transform(X_processed)
                    X_processed = pd.DataFrame(X_transformed, columns=[f'PC_{j+1}' for j in range(X_transformed.shape[1])])
                    fitted_preprocessing_transformers['pca'] = pca

            # Handle Class Imbalance
            if 'imbalance_handling' in preprocessing_config:
                dist = {label: int(y_processed.value_counts().get(label, 0)) for label in [0, 1]}
                st.badge(f"Class distribution before imbalance handling: {dist}")

                if preprocessing_config['imbalance_handling'] == "Oversampling (SMOTE)":
                    oversampler = SMOTE(random_state=42)
                    X_processed, y_processed = oversampler.fit_resample(X_processed, y_processed)
                elif preprocessing_config['imbalance_handling'] == "Undersampling (Random)":
                    undersampler = RandomUnderSampler(random_state=42)
                    X_processed, y_processed = undersampler.fit_resample(X_processed, y_processed)

                dist = {label: int(y_processed.value_counts().get(label, 0)) for label in [0, 1]}
                st.badge(f"Class distribution after imbalance handling: {dist}")
                

            st.session_state.preprocessed_X = X_processed
            st.session_state.preprocessed_y = y_processed
            st.session_state.preprocessing_config_applied = preprocessing_config
            st.session_state.fitted_preprocessing_transformers = fitted_preprocessing_transformers
            st.markdown("---")
            st.success("Preprocessing complete!")
            st.write("Data Shape After Preprocessing:", X_processed.shape)

    st.markdown("---")

    st.subheader("📊 Model Selection")
    model_cols = st.columns(len(st.session_state.models))
    for i, col in enumerate(model_cols):
        with col:
            st.markdown(f"### ⚙️ Model {i + 1}")
            model_type = st.selectbox(f"Model Type {i+1}", [
                "Logistic Regression", "Random Forest", "SVM", "Gradient Boosting",
                "K-Nearest Neighbors", "Naive Bayes", "Decision Tree", "XGBoost", "LightGBM"
            ], key=f"model_type_{i}")
            model_params = {}

            if model_type == "Logistic Regression":
                C = st.select_slider(f"C ", options=[0.01, 0.1, 1, 10, 100], value=1, key=f"C_{i}")
                penalty = st.selectbox(f"Penalty ", ["l2", "l1", "elasticnet"], key=f"penalty_{i}")
                if penalty == 'l1': available_solvers = ["liblinear", "saga"]
                elif penalty == 'elasticnet': available_solvers = ["saga"]
                elif penalty == 'none': available_solvers = ["lbfgs", "saga"]
                else: available_solvers = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
                solver = st.selectbox(f"Solver ",available_solvers, key=f"solver_{i}")
                fit_intercept = st.checkbox(f"Fit Intercept ", value=True, key=f"fit_int_{i}")
                class_weight = st.selectbox(f"Class Weight ", [None, "balanced"], key=f"lw_{i}")
                model_params = {"C": C, "penalty": penalty, "solver": solver, "max_iter": 1000, "fit_intercept": fit_intercept, "class_weight": class_weight}

            elif model_type == "Random Forest":
                n_estimators = st.slider(f"Trees ", 10, 500, step=10, value=100, key=f"n_{i}")
                max_depth = st.slider(f"Max Depth ", 5, 500, step=10, key=f"depth_slot_{i}")
                criterion = st.selectbox(f"Criterion ", ["gini", "entropy", "log_loss"], key=f"crit_{i}")
                min_samples_split = st.slider(f"Min Samples Split ", 2, 20, value=2, key=f"min_split_{i}")
                min_samples_leaf = st.slider(f"Min Samples Leaf ", 1, 20, value=1, key=f"min_leaf_{i}")
                max_features = st.selectbox(f"Max Features ", ["sqrt", "log2", None], key=f"max_feat_{i}")
                bootstrap = st.checkbox(f"Bootstrap ", value=True, key=f"bootstrap_{i}")
                model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "criterion": criterion,
                                "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
                                "max_features": max_features, "bootstrap": bootstrap, "random_state": 42}

            elif model_type == "SVM":
                st.warning("If you are on the cloud, SVM models can take long time to train.")
                C = st.select_slider(f"SVM C ", options=[0.1, 1, 10, 100, 1000], value=1, key=f"svmC_{i}")
                kernel = st.selectbox(f"Kernel ", ["linear", "rbf", "poly", "sigmoid"], key=f"kernel_{i}")
                gamma = st.selectbox(f"Gamma ", ["auto", "scale"], key=f"gamma_{i}")
                degree = st.slider(f"Degree (for poly) ", 2, 5, value=3, key=f"degree_{i}", disabled=(kernel != "poly"))
                coef0 = st.slider(f"Coef0 (for poly, sigmoid) ", -10.0, 10.0, value=0.0, step=0.1, key=f"coef0_{i}", disabled=(kernel not in ["poly", "sigmoid"]))
                shrinking = st.checkbox(f"Shrinking ", value=True, key=f"shrinking_{i}")
                model_params = {"C": C, "kernel": kernel, "gamma": gamma, "degree": degree, "coef0": coef0, "shrinking": shrinking}
               

            elif model_type == "Gradient Boosting":
                n_estimators = st.slider(f"GB Trees ", 50, 300, step=50, value=100, key=f"gb_n_{i}")
                learning_rate = st.select_slider(f"GB Learning Rate ", options=[0.01, 0.05, 0.1, 0.2], value=0.1, key=f"gb_lr_{i}")
                max_depth = st.slider(f"GB Max Depth ", 3, 8, step=1, key=f"gb_depth_{i}")
                subsample = st.select_slider(f"GB Subsample ", options=[0.7, 0.8, 0.9, 1.0], value=1.0, key=f"gb_subsample_{i}")
                model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample, "random_state": 42}

            elif model_type == "K-Nearest Neighbors":
                n_neighbors = st.slider(f"KNN Neighbors ", 1, 20, value=5, key=f"knn_n_{i}")
                weights = st.selectbox(f"KNN Weights ", ["uniform", "distance"], key=f"knn_weights_{i}")
                algorithm = st.selectbox(f"KNN Algorithm ", ["auto", "ball_tree", "kd_tree", "brute"], key=f"knn_algo_{i}")
                model_params = {"n_neighbors": n_neighbors, "weights": weights, "algorithm": algorithm}

            elif model_type == "Naive Bayes":
                var_smoothing = st.select_slider(f"NB Var Smoothing ", options=[1e-10, 1e-9, 1e-8, 1e-7], value=1e-9, key=f"nb_smooth_{i}")
                model_params = {"var_smoothing": var_smoothing}

            elif model_type == "Decision Tree":
                max_depth = st.slider(f"DT Max Depth ", 5, 50, step=1, key=f"dt_depth_{i}")
                min_samples_split = st.slider(f"Min Samples Split ", 2, 20, value=2, key=f"min_split_{i}")
                min_samples_leaf = st.slider(f"Min Samples Leaf ", 1, 20, value=1, key=f"min_leaf_{i}")
                criterion = st.selectbox(f"Criterion ", ["gini", "entropy", "log_loss"], key=f"crit_{i}")
                model_params = {"max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "criterion": criterion, "random_state": 42}

            elif model_type == "XGBoost":
                st.warning("If you are on the cloud, XGBoost models can take long time to train.")
                n_estimators = st.slider(f"XGB Trees ", 50, 300, step=50, value=100, key=f"xgb_n_{i}")
                learning_rate = st.select_slider(f"XGB Learning Rate ", options=[0.01, 0.05, 0.1, 0.2], value=0.1, key=f"xgb_lr_{i}")
                max_depth = st.slider(f"XGB Max Depth ", 3, 10, value=6, key=f"xgb_depth_{i}")
                subsample = st.select_slider(f"XGB Subsample ", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=1.0, key=f"xgb_subsample_{i}")
                colsample_bytree = st.select_slider(f"XGB Colsample ", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=1.0, key=f"xgb_colsample_{i}")
                gamma = st.select_slider(f"XGB Gamma ", options=[0, 0.1, 0.2, 0.4], value=0, key=f"xgb_gamma_{i}")
                model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth,
                                "subsample": subsample, "colsample_bytree": colsample_bytree, "gamma": gamma,
                                "use_label_encoder": False, "eval_metric": "logloss", "random_state": 42}

            elif model_type == "LightGBM":
                st.warning("If you are on the cloud, LightGBM models can take long time to train.")
                n_estimators = st.slider(f"LGBM Trees ", 50, 300, step=50, value=100, key=f"lgbm_n_{i}")
                learning_rate = st.select_slider(f"LGBM Learning Rate ", options=[0.01, 0.05, 0.1, 0.2], value=0.1, key=f"lgbm_lr_{i}")
                num_leaves = st.slider(f"LGBM Num Leaves ", 20, 60, value=31, key=f"lgbm_leaves_{i}")
                max_depth = st.slider(f"LGBM Max Depth ", -1, 10, value=-1, key=f"lgbm_depth_{i}")
                reg_alpha = st.select_slider(f"LGBM L1 Reg ", options=[0, 0.1, 0.5, 1], value=0, key=f"lgbm_reg_a_{i}")
                reg_lambda = st.select_slider(f"LGBM L2 Reg ", options=[0, 0.1, 0.5, 1], value=0, key=f"lgbm_reg_l_{i}")
                model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "num_leaves": num_leaves,
                                "max_depth": max_depth, "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,
                                "random_state": 42}
            st.session_state.models[i] = {"type": model_type, "params": model_params}

    if run_all:
        for i, config in enumerate(st.session_state.models):
            st.markdown("---")
            st.markdown(f"## 🚀 Results for Model {i+1}: {config['type']}")
            
            with st.spinner(f"Training Model {i+1}..."):
                # Instantiate model
                if config['type'] == "Logistic Regression": model = LogisticRegression(**config['params'])
                elif config['type'] == "Random Forest": model = RandomForestClassifier(**config['params'])
                elif config['type'] == "SVM": model = SVC(probability=True, **config['params'])
                elif config['type'] == "Gradient Boosting": model = GradientBoostingClassifier(**config['params'])
                elif config['type'] == "K-Nearest Neighbors": model = KNeighborsClassifier(**config['params'])
                elif config['type'] == "Naive Bayes": model = GaussianNB(**config['params'])
                elif config['type'] == "Decision Tree": model = DecisionTreeClassifier(**config['params'])
                elif config['type'] == "XGBoost": model = XGBClassifier(**config['params'])
                elif config['type'] == "LightGBM": model = LGBMClassifier(**config['params'])

                X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.code(classification_report(y_test, y_pred, digits=3), language="text")

                # ROC Curve
                if hasattr(model, "predict_proba"):
                    y_scores = model.predict_proba(X_test)[:, 1]
                else:
                    y_scores = model.decision_function(X_test)

                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC Curve (AUC = {roc_auc:.2f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
                fig_roc.update_layout(title='📉 ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), width=300, height=400)

                cm = confusion_matrix(y_test, y_pred)
                z = cm.tolist()
                x_labels, y_labels = ['Pred 0', 'Pred 1'], ['Actual 0', 'Actual 1']
                z_text = [[str(val) for val in row] for row in z]

                fig_cm = ff.create_annotated_heatmap(z, x=x_labels, y=y_labels, annotation_text=z_text, colorscale='Blues')
                fig_cm.update_layout(title='🧮 Confusion Matrix', width=300, height=400)

                spacer1, col1, spacer2, col2, spacer3, col3, spacer4 = st.columns([0.1, 1.5, 0.4, 1, 0.5, 1, 0.1], vertical_alignment="center")

                with col1:
                    st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_{i}")
                with col2:
                    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{i}")
                with col3:
                    if not isinstance(X_processed, pd.DataFrame): X_for_cv = np.array(X_processed)
                    else: X_for_cv = X_processed
                    cv_score = cross_val_score(model, X_for_cv, y_processed, cv=5, scoring='accuracy')
                    st.metric(label="CV Accuracy (5-fold)", value=f"{cv_score.mean():.3f}", delta=f"± {cv_score.std():.3f}")

if __name__ == "__main__":
    main()