import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
import plotly.express as px
import plotly.graph_objects as go
import math
from scipy.stats import gaussian_kde
from modules.nav import Navbar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set Streamlit page configuration for wide layout
st.set_page_config(page_title="Breast Cancer Classification App", layout="wide")

def main():
    # Render the navigation bar
    Navbar()

    # Check if data is loaded in session state, if not, display a warning and stop execution
    if "data" in st.session_state:
        data = st.session_state.data
        st.write("Data is loaded from Kaggle Hub. Shape:", data.shape)
    else:
        st.warning("Data not found. Please return to the \"Intro\" page to load it.")
        return

    # Feature Grouping based on common suffixes
    # Exclude 'id', 'diagnosis', and 'Unnamed: 32' (if it exists) from features
    all_numeric_features = [col for col in data.columns if col not in ['id', 'diagnosis', 'Unnamed: 32']]
    # Identify features ending with '_mean'
    mean_features = [col for col in all_numeric_features if col.endswith('_mean')]
    # Identify features ending with '_se' (standard error)
    se_features = [col for col in all_numeric_features if col.endswith('_se')]
    # Identify features ending with '_worst'
    worst_features = [col for col in all_numeric_features if col.endswith('_worst')]

    # Display the main title for the Data Analysis page
    st.title("Data Analysis")

    # Layout for data preview and diagnosis information
    col1, _, col2 = st.columns([6, 0.2, 1], vertical_alignment="center")

    with col1:
        st.subheader("Data Preview")
        # Display the first few rows of the loaded dataset
        st.dataframe(data.head(), hide_index=True)

    with col2:
        st.subheader("Diagnosis Info")
        # Provide a legend for diagnosis categories
        st.success("**Benignant (B)** - 0")
        st.error("**Malignant (M)** - 1")

    st.markdown("---")
    st.subheader("Feature Grouping by Suffix")
    # Display the identified feature groups
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Mean Features")
        st.write(mean_features)
    with col2:
        st.markdown("### Standard Error (SE) Features")
        st.write(se_features)
    with col3:
        st.markdown("### Worst Features")
        st.write(worst_features)

    st.markdown("---")
    st.subheader("Diagnosis Distribution")

    # Prepare data for diagnosis distribution plots
    diag_counts = data['diagnosis'].replace({1: 'Malignant', 0: 'Benign'}).value_counts().reset_index()
    diag_counts.columns = ['Diagnosis', 'Count']

    # Create a bar chart for diagnosis counts
    fig_diag = px.bar(
        diag_counts,
        x='Diagnosis',
        y='Count',
        color='Diagnosis',
        color_discrete_map={"Benign": "green", "Malignant": "red"},
        title="Diagnosis Count (Bar Chart)"
    )
    fig_diag.update_layout(bargap=0.1)

    # Create a pie chart for diagnosis ratios
    fig_pie = px.pie(
        diag_counts,
        names='Diagnosis',
        values='Count',
        color='Diagnosis',
        color_discrete_map={"Benign": "green", "Malignant": "red"},
        title="Diagnosis Ratio (Pie Chart)"
    )
    fig_pie.update_layout(showlegend=False)
    fig_pie.update_traces(textinfo='percent+label')

    # Display bar and pie charts side-by-side
    col1, _, col2 = st.columns([2, 0.2, 2])

    with col1:
        st.plotly_chart(fig_diag, use_container_width=True)

    with col2:
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Correlation Heatmaps Side by Side")

    # Layout for correlation heatmaps and controls
    col1, _, col2 = st.columns([2, 0.2, 2])

    with col1:
        # Slider to set correlation threshold for filtering features
        threshold1 = st.slider("Correlation Threshold", 0.6, 1.0, 0.7, 0.05, key="thresh1")
        # Radio buttons to select correlation type
        correlation_mode = st.radio(
            "Select Correlation Type",
            ["Correlation with Diagnosis", "Correlation Among Features"],
            key="correlation_mode"
        )
        # Calculate absolute correlation matrix for numeric columns
        corr_matrix = data.corr(numeric_only=True).abs()

        if correlation_mode == "Correlation with Diagnosis":
            # Filter features based on correlation with 'diagnosis' above the threshold
            if 'diagnosis' in corr_matrix.columns:
                selected_features = corr_matrix[corr_matrix['diagnosis'] > threshold1].index.tolist()
                # Ensure 'diagnosis' itself is included for plotting
                if "diagnosis" not in selected_features:
                    selected_features.append("diagnosis")

                if len(selected_features) < 2:
                    st.warning("No features found with the selected threshold. Try lowering the threshold.")
                else:
                    # Create heatmap for selected features' correlation
                    filtered_corr = data[selected_features].corr(numeric_only=True)
                    fig_filtered = px.imshow(
                        filtered_corr,
                        text_auto=True,
                        x=filtered_corr.columns,
                        y=filtered_corr.columns,
                        color_continuous_scale='viridis',
                        title="Correlation with Diagnosis"
                    )
                    fig_filtered.update_layout(width=800, height=600, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_filtered, use_container_width=False)
            else:
                st.warning("Diagnosis column not found for correlation analysis.")
        else:  # Correlation Among Features
            # Exclude non-feature columns for mutual feature correlation
            features_for_mutual_corr = [col for col in data.columns if col not in ['id', 'diagnosis', 'Unnamed: 32']]
            mutual_corr_matrix = data[features_for_mutual_corr].corr(numeric_only=True).abs()

            # Mask the upper triangle and diagonal for cleaner display of pairs
            mask = np.triu(np.ones(mutual_corr_matrix.shape), k=1).astype(bool)
            corr_long = mutual_corr_matrix.where(mask).stack().reset_index()
            corr_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

            # Filter for strong correlation pairs
            strong_pairs = corr_long[corr_long['Correlation'] > threshold1]

            if strong_pairs.empty:
                st.warning("No feature pairs meet the threshold. Try lowering it.")
            else:
                # Get all unique features involved in strong pairs
                selected_features_for_mutual_corr_plot = list(set(strong_pairs['Feature 1']).union(set(strong_pairs['Feature 2'])))
                filtered_corr = data[selected_features_for_mutual_corr_plot].corr(numeric_only=True)

                fig_filtered = px.imshow(
                    filtered_corr,
                    text_auto=True,
                    x=filtered_corr.columns,
                    y=filtered_corr.columns,
                    color_continuous_scale='viridis',
                    title="Mutual Feature Correlation"
                )
                fig_filtered.update_layout(width=800, height=600, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_filtered, use_container_width=False)

    with col2:
        # Slider for thresholding absolute sorted correlation with diagnosis
        threshold2 = st.slider("Threshold for Absolute Sorted Correlation", 0.3, 1.0, 0.7, 0.05, key="thresh2")
        show_nums = True if threshold2 > 0.55 else False # Conditional display of correlation values

        if 'diagnosis' in data.columns:
            # Calculate and sort absolute correlations with 'diagnosis'
            abs_corr_sorted = data.corr(numeric_only=True)[['diagnosis']].abs().sort_values(by='diagnosis', ascending=False)
            filtered_abs_corr = abs_corr_sorted[abs_corr_sorted['diagnosis'] > threshold2]

            if filtered_abs_corr.empty:
                st.warning("No features found with the selected threshold. Try lowering the threshold.")
            else:
                # Display absolute correlation heatmap
                fig_sorted = px.imshow(filtered_abs_corr,
                                     text_auto=show_nums,
                                     color_continuous_scale='viridis',
                                     title="Absolute Correlation with Diagnosis")
                fig_sorted.update_layout(
                    width=800,
                    height=700,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_sorted, use_container_width=False)
        else:
            st.warning("Diagnosis column not found for sorted correlation analysis.")

    st.markdown("---")
    st.subheader("Feature Distribution by Diagnosis")

    # Layout for histogram plot and its controls
    hist_plot_col, _, hist_controls_col = st.columns([5, 0.2, 1], vertical_alignment="center")

    with hist_controls_col:
        # Select box for feature group type (mean, se, worst)
        feature_type = st.selectbox(
        "Select Feature Type",
        ["mean", "se", "worst"],
        index=0,
        key="feature_type"
        )

        # Filter features based on selected type
        current_feature_group_cols = []
        if feature_type == "mean":
            current_feature_group_cols = mean_features
        elif feature_type == "se":
            current_feature_group_cols = se_features
        elif feature_type == "worst":
            current_feature_group_cols = worst_features

        # Get base names for features (e.g., 'radius' from 'radius_mean')
        available_features_base_names = [f.replace("_" + feature_type, '') for f in current_feature_group_cols]

        if not available_features_base_names:
            st.warning(f"No features found for type '{feature_type}'.")
            feature_selected_base = None
        else:
            # Select box for specific feature name
            feature_selected_base = st.selectbox(
                "Select Feature",
                available_features_base_names,
                index=0,
                key="feature_name"
            )

        # Slider to control the number of bins in the histogram
        bin_count = st.slider("Number of bins", min_value=10, max_value=100, value=30, step=5)

    with hist_plot_col:
        if feature_selected_base:
            # Construct the full feature name
            feature_full_name = feature_selected_base + "_" + feature_type

            # Separate data for malignant and benign diagnoses
            malignant_values = data[data["diagnosis"] == 1][feature_full_name]
            benign_values = data[data["diagnosis"] == 0][feature_full_name]

            # Create Plotly figure with overlapping histograms
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=malignant_values,
                name='Malignant',
                opacity=0.5,
                marker_color='red',
                nbinsx=bin_count
            ))

            fig.add_trace(go.Histogram(
                x=benign_values,
                name='Benign',
                opacity=0.5,
                marker_color='green',
                nbinsx=bin_count
            ))

            fig.update_layout(
                barmode='overlay', # Overlay histograms
                title=f"Histogram of {feature_selected_base.capitalize()} {feature_type.replace('_', '').capitalize()} for Benign and Malignant Tumors",
                xaxis_title=f"{feature_selected_base.capitalize()} {feature_type.replace('_', '').capitalize()} Values",
                yaxis_title="Frequency",
                legend_title="Diagnosis",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a feature to display its distribution.")

    st.markdown("---")
    st.subheader("Distribution of Feature Group with KDE (Interactive Grid)")

    # Select box to choose which feature group to display for KDE plots
    box_kde_feature_group = st.selectbox("Select Feature Group to Display", ["mean", "se", "worst"], key="grid_kde_feature_group")

    # Assign features based on the selected group
    if box_kde_feature_group == "mean":
        selected_features_for_kde = mean_features
    elif box_kde_feature_group == "se":
        selected_features_for_kde = se_features
    else:
        selected_features_for_kde = worst_features

    # Determine grid layout for plots
    num_features_in_kde_group = len(selected_features_for_kde)
    cols_per_row = 5
    rows = math.ceil(num_features_in_kde_group / cols_per_row)

    # Render a grid of histogram + KDE overlay plots for each feature in the selected group
    for row in range(rows):
        row_cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            idx = row * cols_per_row + i
            if idx < num_features_in_kde_group:
                feature = selected_features_for_kde[idx]
                x = data[feature].dropna()

                if not x.empty: # Only plot if data exists for the feature
                    try:
                        # Estimate Kernel Density Estimation (KDE)
                        kde = gaussian_kde(x)
                        x_range = np.linspace(x.min(), x.max(), 40)
                        y_kde = kde(x_range)

                        # Create combined histogram + KDE plot
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=x,
                            nbinsx=30,
                            name="Histogram",
                            marker_color="lightblue",
                            opacity=0.6,
                            showlegend=False,
                        ))

                        # Scale KDE to match histogram height
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_kde * len(x) * (x_range[1] - x_range[0]),
                            mode='lines',
                            name='KDE',
                            line=dict(color='blue'),
                            showlegend=False

                        ))

                        fig.update_layout(
                            title=feature,
                            height=300,
                            margin=dict(l=10, r=10, t=30, b=20),
                            xaxis_title='',
                            yaxis_title='',
                            template='plotly_white'
                        )

                        with row_cols[i]:
                            st.plotly_chart(fig, use_container_width=True)
                    except np.linalg.LinAlgError:
                        with row_cols[i]:
                            st.warning(f"KDE failed for {feature}. Data might be too sparse.")
                else:
                    with row_cols[i]:
                        st.info(f"No data for {feature} to plot KDE.")

    st.markdown("---")
    st.subheader("Pairwise Feature Relationships")

    # Define all possible features for the pair plot, excluding IDs and unnamed columns
    all_plot_features = [col for col in data.columns if col != 'id' and col != 'Unnamed: 32']
    # Set default features to display in the pair plot
    default_pairplot_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
    # Filter defaults to ensure they exist in the current data
    default_pairplot_features = [f for f in default_pairplot_features if f in data.columns]

    # Multiselect widget for user to choose features for the pair plot
    subset_features = st.multiselect(
        "Select 3-9 Features to Visualize (include 'diagnosis' for coloring)",
        options=all_plot_features,
        default=default_pairplot_features
    )

    # Input validation for pair plot feature selection
    if 'diagnosis' not in subset_features:
        st.warning("Include 'diagnosis' in your selection for coloring.")
    else:
        # Extract numeric dimensions for the scatter matrix
        numeric_dimensions = [f for f in subset_features if f != 'diagnosis']
        if len(numeric_dimensions) < 2:
            st.warning("Select at least 2 numeric features to generate the scatter matrix.")
        elif len(subset_features) > 9:
            st.warning("Please select no more than 9 features for better readability.")
        else:
            # Create a scatter matrix (pair plot) colored by diagnosis
            fig = px.scatter_matrix(
                data_frame=data,
                dimensions=numeric_dimensions,
                color=data['diagnosis'].replace({0: "Benign", 1: "Malignant"}),
                color_discrete_map={"Benign": "green", "Malignant": "red"},
                title="Pairplot-like Scatter Matrix Colored by Diagnosis",
                height=900
            )
            fig.update_traces(diagonal_visible=False) # Hide diagonal plots (histograms/KDEs)
            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                font=dict(size=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Boxplot Grid of Features by Diagnosis")

    # Select box to choose which group of features to visualize in boxplots
    box_feature_group = st.selectbox("Select Feature Group for Boxplots", ["mean", "se", "worst"], key="box_group")

    # Assign features for boxplotting based on the selected group
    feature_columns_for_boxplot = []
    if box_feature_group == "mean":
        feature_columns_for_boxplot = mean_features
    elif box_feature_group == "se":
        feature_columns_for_boxplot = se_features
    else:
        feature_columns_for_boxplot = worst_features

    # Grid settings for displaying multiple boxplots
    n_cols = 5
    n_rows = (len(feature_columns_for_boxplot) + n_cols - 1) // n_cols

    # Loop through the grid to show each boxplot in a cell
    for row in range(n_rows):
        row_cols = st.columns(n_cols)
        for i in range(n_cols):
            idx = row * n_cols + i
            if idx < len(feature_columns_for_boxplot):
                feature = feature_columns_for_boxplot[idx]
                # Create boxplot for the current feature, colored by diagnosis
                fig = px.box(
                    data,
                    x="diagnosis",
                    y=feature,
                    color=data["diagnosis"].replace({0: "Benign", 1: "Malignant"}),
                    color_discrete_map={"Benign": "green", "Malignant": "red"},
                    points="all",  # Show individual data points
                    title=feature
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=20),
                    xaxis_title="",
                    yaxis_title="",
                    showlegend=False
                )
                row_cols[i].plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Principal Component Analysis (PCA)")

    # Prepare data for PCA: exclude non-feature columns (like 'diagnosis' itself for PCA)
    pca_features = data.drop(columns=['diagnosis'], errors='ignore')

    if pca_features.empty:
        st.warning("No numeric features available for PCA. Check your dataset and column exclusions.")
    else:
        # Standardize features before applying PCA
        scaler_pca = StandardScaler()
        scaled_pca_features = scaler_pca.fit_transform(pca_features)
        # Convert scaled features back to a DataFrame, preserving index and column names
        scaled_pca_df = pd.DataFrame(scaled_pca_features, index=pca_features.index, columns=pca_features.columns)

        # Layout for PCA plot and information/controls
        pca_plot_col, pca_info_col = st.columns([3, 1], vertical_alignment="center")

        with pca_info_col:
            # Determine available PCA dimensions (max 3 for 3D plot)
            available_pca_dims = min(3, scaled_pca_df.shape[1])

            # Radio buttons to select 2 or 3 dimensions for PCA
            pca_dimensions = st.radio(
                "Select number of dimensions for PCA",
                options=[d for d in [2, 3] if d <= available_pca_dims],
                index=0 if 2 <= available_pca_dims else (1 if 3 <= available_pca_dims else 0),
                key="pca_dimensions_radio"
            )

        with pca_plot_col:
            if pca_dimensions > scaled_pca_df.shape[1]:
                st.warning(f"Cannot select {pca_dimensions} dimensions for PCA. Only {scaled_pca_df.shape[1]} features available after exclusions.")
            else:
                # Perform PCA with the selected number of components
                pca = PCA(n_components=pca_dimensions)
                principal_components = pca.fit_transform(scaled_pca_df)

                # Create a DataFrame for the principal components
                pca_df = pd.DataFrame(data=principal_components,
                                      index=scaled_pca_df.index,
                                      columns=[f'PC_{i+1}' for i in range(pca_dimensions)])

                # Add diagnosis column to PCA DataFrame for coloring
                pca_df['diagnosis'] = data['diagnosis'].replace({0: "Benign", 1: "Malignant"})

                # Merge original numeric features for hover data in plots
                df_for_plotting = pca_df.merge(data[all_numeric_features], left_index=True, right_index=True, how='left')

                # Generate 2D or 3D scatter plot based on selected dimensions
                if pca_dimensions == 2:
                    fig_pca = px.scatter(df_for_plotting,
                                         x='PC_1',
                                         y='PC_2',
                                         color='diagnosis',
                                         title='PCA - 2 Components',
                                         hover_data=all_numeric_features, # Show original feature values on hover
                                         color_discrete_map={"Benign": "green", "Malignant": "red"}, height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)
                elif pca_dimensions == 3:
                    fig_pca = px.scatter_3d(df_for_plotting,
                                            x='PC_1',
                                            y='PC_2',
                                            z='PC_3',
                                            color='diagnosis',
                                            title='PCA - 3 Components',
                                            hover_data=all_numeric_features, # Show original feature values on hover
                                            color_discrete_map={"Benign": "green", "Malignant": "red"}, height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)

        with pca_info_col:
            # Display explained variance ratio if PCA was successfully performed
            if 'pca' in locals() and pca_dimensions >= 2 and pca_dimensions <= scaled_pca_df.shape[1]:
                st.markdown(f"**Explained Variance Ratio (Cumulative):** {pca.explained_variance_ratio_.sum():.2f}")
                st.write("Individual Explained Variance Ratio per Component:")
                # Display individual variance ratios in a DataFrame
                variance_df = pd.DataFrame(pca.explained_variance_ratio_,
                                            index=[f'PC_{i+1}' for i in range(pca_dimensions)],
                                            columns=['Variance Ratio'])
                st.dataframe(variance_df, use_container_width=True)
            else:
                st.info("Select PCA dimensions to view explained variance ratio.")

    st.markdown("---")

if __name__ == "__main__":
    main()