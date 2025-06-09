import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go
from modules.nav import Navbar # Assuming this is your navigation module

# Import preprocessing modules for applying transformations (needed for apply_preprocessing_to_dataframe_dl)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, RFE, f_classif
from sklearn.decomposition import PCA
from scipy.stats import zscore, mstats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter # For displaying imbalance

st.set_page_config(page_title="Deep Learning Models", layout="wide")

# Helper function to apply preprocessing transformers to a DataFrame
# This function will use the *fitted* transformers passed from session state
@st.cache_data(show_spinner=False) # Cache transformation results
def apply_preprocessing_to_dataframe_dl(df_input, original_features, fitted_transformers):
    df = df_input.copy()
    
    # Ensure columns match original features before applying transformations
    df = df[original_features].copy()

    # Apply Imputation (if imputer was fitted)
    if 'imputer' in fitted_transformers and fitted_transformers['imputer'] is not None:
        df = pd.DataFrame(fitted_transformers['imputer'].transform(df), columns=df.columns)

    # Apply Scaling (if scaler was fitted)
    if 'scaler' in fitted_transformers and fitted_transformers['scaler'] is not None:
        df = pd.DataFrame(fitted_transformers['scaler'].transform(df), columns=df.columns)

    # Apply Feature Selection/Reduction (if selector/pca was fitted)
    if 'feature_selector' in fitted_transformers and fitted_transformers['feature_selector'] is not None:
        df = pd.DataFrame(fitted_transformers['feature_selector'].transform(df), columns=fitted_transformers['feature_selector'].get_feature_names_out(df.columns))
    elif 'pca' in fitted_transformers and fitted_transformers['pca'] is not None:
        df = pd.DataFrame(fitted_transformers['pca'].transform(df), columns=[f'PC_{j+1}' for j in range(fitted_transformers['pca'].n_components_)])
    
    return df


def main():
    Navbar() # Display the navigation bar (assumed to handle general navigation, not page-specific sidebar controls)
    

    # Load dataset from session
    if "data" in st.session_state:
        data = st.session_state.data
        st.write("Data is loaded from Kaggle Hub. Shape:", data.shape)
    else:
        st.warning("Data not found. Please return to the \"Intro\" page to load it.")
        return
    
    st.title("Deep Learning Models")
    # Identify feature columns
    features_for_model = [col for col in data.columns if col not in ['id', 'diagnosis', 'Unnamed: 32']]
    X_original = data[features_for_model]
    y_original = data['diagnosis']

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Data Source")
        data_source_option = st.radio(
            "Select data for training:",
            ("Original Data", "Preprocessed Data"),
            key="dl_data_source_selector"
        )

        # Determine if the train button should be disabled
        train_button_disabled = False 
        if data_source_option == "Preprocessed Data":
            if "preprocessed_X" not in st.session_state or st.session_state.preprocessed_X is None or st.session_state.preprocessed_X.empty:
                st.warning("Preprocessed data not found or is empty. Please select at least one preprocessing step from the \" ML Models\" page")
                train_button_disabled = True # Disable if data is missing or empty
            else:
                st.info(f"Preprocessed Data Shape: {st.session_state.preprocessed_X.shape}") # Display shape here
        
        st.markdown("---") # Separator in sidebar
        st.subheader("Action")
        # The button is always rendered, its disabled state is controlled by the variable
        train_dl_model_button = st.button("🚀 Train Deep Learning Model", disabled=train_button_disabled)


    # --- Deep Learning Model Configuration ---
    st.subheader("⚙️ Deep Learning Model Configuration")

    # Group common sliders into 2 columns
    optimizer_type = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
    col1, col2 = st.columns(2)
    with col1:
        num_layers = st.slider("Number of Hidden Layers", 1, 5, 2)
        epochs = st.slider("Epochs", 10, 200, 50)
        dropout_rate = st.slider("Dropout Rate (for hidden layers)", 0.0, 0.5, 0.2, step=0.05)

    with col2:
        learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        batch_size = st.slider("Batch Size", 16, 256, 32)


    # Sliders for neurons in each hidden layer - displayed in 2 columns below general config
    st.markdown("---")
    st.subheader("Hidden Layer Neuron Configuration")
    hidden_layer_neurons = []
    
    # Create columns for the neuron sliders
    cols_neurons = st.columns(2)
    
    for i in range(num_layers):
        with cols_neurons[i % 2]: # Distribute sliders between the two columns
            neurons = st.slider(f"Neurons in Hidden Layer {i+1}", 16, 256, 64, key=f"neurons_l{i}")
            hidden_layer_neurons.append(neurons)

    optimizer_map = {
        "Adam": Adam(learning_rate=learning_rate),
        "SGD": SGD(learning_rate=learning_rate),
        "RMSprop": RMSprop(learning_rate=learning_rate)
    }

    st.markdown("---") # Separator in main content area

    # --- Training and Evaluation Logic (triggered by the sidebar button) ---
    if train_dl_model_button:
        # Prepare data based on selection
        X_train_data = None
        y_train_target = None
        current_fitted_transformers = {} # Transformers used for training in this run

        if data_source_option == "Original Data":
            X_train_data = X_original.copy()
            y_train_target = y_original.copy()
        else: # Preprocessed Data
            if "preprocessed_X" in st.session_state and st.session_state.preprocessed_X is not None and not st.session_state.preprocessed_X.empty:
                X_train_data = st.session_state.preprocessed_X.copy()
                y_train_target = st.session_state.preprocessed_y.copy()
                # Use the fitted transformers from the Compare Models page for subsequent transformations
                current_fitted_transformers = st.session_state.fitted_preprocessing_transformers
            else:
                # This case should ideally be prevented by train_button_disabled
                st.error("Preprocessed data is selected but not available. Please go to 'Compare Models' page and run preprocessing.")
                return # Stop execution if data is missing

        if X_train_data is None or X_train_data.empty:
            st.error("Training data is not prepared. Please check data loading and preprocessing steps.")
            return

        with st.spinner("Training Deep Learning Model... This might take a while."):
            # Convert to NumPy arrays for Keras
            X_np = X_train_data.values
            y_np = y_train_target.values

            # Ensure target is float32 if using binary_crossentropy and not one-hot encoded
            y_np = y_np.astype(np.float32)

            # Split data for training and validation within this page
            X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

            # Build the Keras Model
            model = Sequential()
            # Input Layer
            model.add(Dense(units=X_train.shape[1], activation='relu', input_shape=(X_train.shape[1],)))

            # Hidden Layers - using the values from the dynamically created sliders
            for i in range(num_layers):
                model.add(Dense(units=hidden_layer_neurons[i], activation='relu'))
                model.add(Dropout(dropout_rate))

            # Output Layer (Binary Classification)
            model.add(Dense(units=1, activation='sigmoid')) # Sigmoid for binary classification probability

            # Compile the model
            model.compile(optimizer=optimizer_map[optimizer_type],
                          loss='binary_crossentropy', # Appropriate loss for binary classification
                          metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

            # Train the model
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(X_val, y_val),
                                verbose=0) # Suppress verbose output during training

            st.success("Deep Learning Model Trained Successfully!")

            # Store the trained model and associated info in session state
            st.session_state.trained_dl_model = {
                'model': model,
                'features_used': features_for_model, # Store original feature names
                'data_source_type': data_source_option.lower().replace(" ", "_"),
                'fitted_transformers': current_fitted_transformers, # Store transformers used if preprocessed
                'dl_model_config': {
                    'num_layers': num_layers,
                    'hidden_layer_neurons': hidden_layer_neurons, # Store the neuron counts
                    'optimizer_type': optimizer_type,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'dropout_rate': dropout_rate,
                }
            }

            # --- Evaluation ---
            st.subheader("📊 Model Evaluation")

            # Make predictions on the validation set
            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions

            st.markdown("##### Classification Report:")
            st.code(classification_report(y_val, y_pred, digits=3), language="text")

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba.flatten()) # Ensure y_pred_proba is 1D
            roc_auc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC Curve (AUC = {roc_auc:.2f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
            fig_roc.update_layout(
                title='📉 ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                width=400,
                height=400
            )

            # Confusion Matrix
            cm = confusion_matrix(y_val, y_pred)
            z = cm.tolist()
            x_labels = ['Pred Benign (0)', 'Pred Malignant (1)']
            y_labels = ['Actual Benign (0)', 'Actual Malignant (1)']
            z_text = [[str(val) for val in row] for row in z]

            fig_cm = ff.create_annotated_heatmap(
                z, x=x_labels, y=y_labels, annotation_text=z_text, colorscale='Blues'
            )
            fig_cm.update_layout(
                title='🧮 Confusion Matrix',
                width=400,
                height=400
            )
            
            # Display plots side-by-side
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.plotly_chart(fig_roc, use_container_width=True)
            with col_metrics2:
                st.plotly_chart(fig_cm, use_container_width=True)
            
            st.info(f"Model trained on {data_source_option}. Evaluation metrics calculated on a 20% validation split.")
            
            # Display training history
            st.subheader("📈 Training History (Loss & Accuracy)")
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(x=list(range(epochs)), y=history.history['loss'], mode='lines', name='Training Loss'))
            fig_history.add_trace(go.Scatter(x=list(range(epochs)), y=history.history['val_loss'], mode='lines', name='Validation Loss'))
            # Use secondary y-axis for accuracy
            fig_history.add_trace(go.Scatter(x=list(range(epochs)), y=history.history['accuracy'], mode='lines', name='Training Accuracy', yaxis='y2'))
            fig_history.add_trace(go.Scatter(x=list(range(epochs)), y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy', yaxis='y2'))
            fig_history.update_layout(
                title='Training & Validation Loss/Accuracy',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                yaxis=dict(title='Loss', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
                yaxis2=dict(title='Accuracy', overlaying='y', side='right', titlefont=dict(color='red'), tickfont=dict(color='red')),
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_history, use_container_width=True)


if __name__ == "__main__":
    # Set TensorFlow verbosity to suppress excessive output
    tf.get_logger().setLevel('ERROR')
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
