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
from sklearn.decomposition import PCA # Import PCA for the new section
from sklearn.preprocessing import StandardScaler # Needed for PCA

st.set_page_config(page_title="Breast Cancer Classification App", layout="wide")


def main():
    Navbar()

    if "data" in st.session_state:
        data = st.session_state.data
        st.write("Data is loaded from Kaggle. Shape:", data.shape)
    else:
        st.warning("Data not found. Please return to the main page to load it.")
        return # Exit if data is not loaded

    # Feature Grouping
    # Ensure these lists are generated from the actual columns after initial data loading
    # and dropping of non-feature columns like 'id', 'Unnamed: 32', 'diagnosis'.
    # This prevents errors if data is not fully loaded or columns are missing.
    all_numeric_features = [col for col in data.columns if col not in ['id', 'diagnosis', 'Unnamed: 32']]
    mean_features = [col for col in all_numeric_features if col.endswith('_mean')]
    se_features = [col for col in all_numeric_features if col.endswith('_se')]
    worst_features = [col for col in all_numeric_features if col.endswith('_worst')]


    # Page Rendering
    st.title("Data Analysis")


    col1, col2 = st.columns([4, 1],vertical_alignment="center")

    with col1:
        st.subheader("Data Preview")
        st.dataframe(data.head(), hide_index=True)

    with col2:
        st.subheader("Diagnosis Info")
        st.markdown("**Benignant (B)** - 0\n\n**Malignant (M)** - 1")


    st.markdown("---")
    st.subheader("Feature Grouping by Suffix")
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

    # Prepare the data
    diag_counts = data['diagnosis'].replace({1: 'Malignant', 0: 'Benign'}).value_counts().reset_index()
    diag_counts.columns = ['Diagnosis', 'Count']

    # Bar Chart
    fig_diag = px.bar(
        diag_counts,
        x='Diagnosis',
        y='Count',
        color='Diagnosis',
        color_discrete_map={"Benign": "green", "Malignant": "red"},
        title="Diagnosis Count (Bar Chart)"
    )
    fig_diag.update_layout(bargap=0.1)

    # Pie Chart
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

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_diag, use_container_width=True)

    with col2:
        st.plotly_chart(fig_pie, use_container_width=True)


    st.markdown("---")
    st.subheader("Interactive Correlation Heatmaps Side by Side")

    col1, col2 = st.columns(2)

    with col1:
        threshold1 = st.slider("Correlation Threshold", 0.6, 1.0, 0.7, 0.05, key="thresh1")
        correlation_mode = st.radio(
            "Select Correlation Type",
            ["Correlation with Diagnosis", "Correlation Among Features"],
            key="correlation_mode"
        )
        # Get correlation matrix
        corr_matrix = data.corr(numeric_only=True).abs() # Added numeric_only=True

        if correlation_mode == "Correlation with Diagnosis":
            # Ensure 'diagnosis' is in the columns of corr_matrix.
            # If it's not and threshold is applied, it will cause an error.
            if 'diagnosis' in corr_matrix.columns:
                selected_features = corr_matrix[corr_matrix['diagnosis'] > threshold1].index.tolist()

                # Always include diagnosis itself if it was a feature in the correlation matrix
                if "diagnosis" not in selected_features:
                    selected_features.append("diagnosis")

                if len(selected_features) < 2:
                    st.warning("No features found with the selected threshold. Try lowering the threshold.")
                else:
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
            # Exclude diagnosis for feature-to-feature correlation
            features_for_mutual_corr = [col for col in data.columns if col not in ['id', 'diagnosis', 'Unnamed: 32']]
            mutual_corr_matrix = data[features_for_mutual_corr].corr(numeric_only=True).abs()

            # Mask lower triangle and diagonal
            mask = np.triu(np.ones(mutual_corr_matrix.shape), k=1).astype(bool)
            corr_long = mutual_corr_matrix.where(mask).stack().reset_index()
            corr_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

            strong_pairs = corr_long[strong_pairs['Correlation'] > threshold1]

            if strong_pairs.empty:
                st.warning("No feature pairs meet the threshold. Try lowering it.")
            else:
                # Get all involved features
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
        threshold2 = st.slider("Threshold for Absolute Sorted Correlation", 0.3, 1.0, 0.6, 0.05, key="thresh2")
        
        # Ensure 'diagnosis' exists before attempting correlation
        if 'diagnosis' in data.columns:
            abs_corr_sorted = data.corr(numeric_only=True)[['diagnosis']].abs().sort_values(by='diagnosis', ascending=False)
            filtered_abs_corr = abs_corr_sorted[abs_corr_sorted['diagnosis'] > threshold2]
        
            if filtered_abs_corr.empty:
                st.warning("No features found with the selected threshold. Try lowering the threshold.")
            else:
                fig_sorted = px.imshow(filtered_abs_corr,
                                     text_auto=True,
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
    
    # New column layout for Histogram and its controls
    hist_plot_col, hist_controls_col = st.columns([3, 1], vertical_alignment="top") # Adjust ratios as needed

    with hist_controls_col: # Controls go into the right column
        # First Dropdown: Feature Type
        feature_type = st.selectbox(
        "Select Feature Type",
        ["mean", "se", "worst"],
        index=0,
        key="feature_type"
        )

        # Filter Features by Type
        current_feature_group_cols = []
        if feature_type == "mean":
            current_feature_group_cols = mean_features
        elif feature_type == "se":
            current_feature_group_cols = se_features
        elif feature_type == "worst":
            current_feature_group_cols = worst_features
        
        available_features_base_names = [f.replace("_" + feature_type, '') for f in current_feature_group_cols]

        if not available_features_base_names:
            st.warning(f"No features found for type '{feature_type}'.")
            feature_selected_base = None
        else:
            # Second Dropdown: Feature Name
            feature_selected_base = st.selectbox(
                "Select Feature",
                available_features_base_names,
                index=0,
                key="feature_name"
            )

        bin_count = st.slider("Number of bins", min_value=10, max_value=100, value=30, step=5)

    with hist_plot_col: # Plot goes into the left column
        if feature_selected_base:
            # Build Full Feature Name Correctly
            feature_full_name = feature_selected_base + "_" + feature_type
            
            # Data Preparation
            malignant_values = data[data["diagnosis"] == 1][feature_full_name]
            benign_values = data[data["diagnosis"] == 0][feature_full_name]

            # Create Plotly Overlapping Histograms
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
                barmode='overlay',
                title=f"Histogram of {feature_selected_base.capitalize()} {feature_type.replace('_', '').capitalize()} for Benign and Malignant Tumors",
                xaxis_title=f"{feature_selected_base.capitalize()} {feature_type.replace('_', '').capitalize()} Values",
                yaxis_title="Frequency",
                legend_title="Diagnosis",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True) # use_container_width=True to fill the column
        else:
            st.info("Select a feature to display its distribution.")

    
    st.markdown("---")
    st.subheader("Distribution of Feature Group with KDE (Interactive Grid)")

    # Select feature group
    box_kde_feature_group = st.selectbox("Select Feature Group to Display", ["mean", "se", "worst"], key="grid_kde_feature_group")

    # Get columns by suffix
    if box_kde_feature_group == "mean":
        selected_features_for_kde = mean_features
    elif box_kde_feature_group == "se":
        selected_features_for_kde = se_features
    else:
        selected_features_for_kde = worst_features

    # Layout
    num_features_in_kde_group = len(selected_features_for_kde)
    cols_per_row = 5
    rows = math.ceil(num_features_in_kde_group / cols_per_row)

    # Render grid of histogram + KDE overlay
    for row in range(rows):
        row_cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            idx = row * cols_per_row + i
            if idx < num_features_in_kde_group:
                feature = selected_features_for_kde[idx]
                x = data[feature].dropna()

                if not x.empty: # Only plot if there's data
                    # Estimate KDE
                    try:
                        kde = gaussian_kde(x)
                        x_range = np.linspace(x.min(), x.max(), 40)
                        y_kde = kde(x_range)

                        # Create combined histogram + KDE
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=x,
                            nbinsx=30,
                            name="Histogram",
                            marker_color="lightblue",
                            opacity=0.6,
                            showlegend=False,
                        ))

                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_kde * len(x) * (x_range[1] - x_range[0]),  # Scale KDE to histogram
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
    # Select features for the pair plot
    st.subheader("Pairwise Feature Relationships")

    # Feature selection (max 12)
    # Ensure 'diagnosis' is always an option and not included in dimensions automatically
    all_plot_features = [col for col in data.columns if col != 'id' and col != 'Unnamed: 32']
    default_pairplot_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
    # Filter defaults to ensure they exist in the current data
    default_pairplot_features = [f for f in default_pairplot_features if f in data.columns]


    subset_features = st.multiselect(
        "Select 3-9 Features to Visualize (include 'diagnosis' for coloring)",
        options=all_plot_features,
        default=default_pairplot_features
    )

    # Input validation
    if 'diagnosis' not in subset_features:
        st.warning("Include 'diagnosis' in your selection for coloring.")
    else:
        numeric_dimensions = [f for f in subset_features if f != 'diagnosis']
        if len(numeric_dimensions) < 2:
            st.warning("Select at least 2 numeric features to generate the scatter matrix.")
        elif len(subset_features) > 9: # Check total selected features, not just numeric ones
            st.warning("Please select no more than 9 features for better readability.")
        else:
            fig = px.scatter_matrix(
                data_frame=data,
                dimensions=numeric_dimensions, # Only numeric features for dimensions
                color=data['diagnosis'].replace({0: "Benign", 1: "Malignant"}),
                color_discrete_map={"Benign": "green", "Malignant": "red"},
                title="Pairplot-like Scatter Matrix Colored by Diagnosis",
                height=900
            )
            fig.update_traces(diagonal_visible=False)
            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                font=dict(size=10),
            )
            st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")
    st.subheader("Boxplot Grid of Features by Diagnosis")

    # Let user pick which group of features to view
    box_feature_group = st.selectbox("Select Feature Group for Boxplots", ["mean", "se", "worst"], key="box_group")

    feature_columns_for_boxplot = []
    if box_feature_group == "mean":
        feature_columns_for_boxplot = mean_features
    elif box_feature_group == "se":
        feature_columns_for_boxplot = se_features
    else:
        feature_columns_for_boxplot = worst_features

    # Grid settings
    n_cols = 5
    n_rows = (len(feature_columns_for_boxplot) + n_cols - 1) // n_cols

    # Loop through the grid and show each plot in a cell
    for row in range(n_rows):
        row_cols = st.columns(n_cols)
        for i in range(n_cols):
            idx = row * n_cols + i
            if idx < len(feature_columns_for_boxplot):
                feature = feature_columns_for_boxplot[idx]
                fig = px.box(
                    data,
                    x="diagnosis",
                    y=feature,
                    color=data["diagnosis"].replace({0: "Benign", 1: "Malignant"}),
                    color_discrete_map={"Benign": "green", "Malignant": "red"},
                    points="all",  # show individual data points
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

    # Prepare data for PCA: exclude non-numeric, non-feature columns, and scale
    pca_features = data.drop(columns=[ 'diagnosis'], errors='ignore')
    
    if pca_features.empty:
        st.warning("No numeric features available for PCA. Check your dataset and column exclusions.")
    else:
        # Standardize the features before applying PCA
        scaler_pca = StandardScaler()
        scaled_pca_features = scaler_pca.fit_transform(pca_features)
        # Create a DataFrame for scaled features, preserving the original index and column names
        scaled_pca_df = pd.DataFrame(scaled_pca_features, index=pca_features.index, columns=pca_features.columns) 

        # Split PCA section into two columns
        pca_plot_col, pca_info_col = st.columns([3, 1], vertical_alignment="center") # Adjust ratios as needed

        with pca_info_col: # PCA controls and info go into the right column
            # Limit PCA dimensions to actual features available after exclusions
            available_pca_dims = min(3, scaled_pca_df.shape[1]) # Now only 2 or 3 is available
            
            # Use radio buttons for PCA dimensions
            pca_dimensions = st.radio(
                "Select number of dimensions for PCA",
                options=[d for d in [2, 3] if d <= available_pca_dims], # Only show 2 or 3 if available
                index=0 if 2 <= available_pca_dims else (1 if 3 <= available_pca_dims else 0), # Default to 2, or 3 if only 3 is available
                key="pca_dimensions_radio"
            )


        with pca_plot_col: # PCA plot goes into the left column
            if pca_dimensions > scaled_pca_df.shape[1]:
                st.warning(f"Cannot select {pca_dimensions} dimensions for PCA. Only {scaled_pca_df.shape[1]} features available after exclusions.")
            else:
                pca = PCA(n_components=pca_dimensions)
                principal_components = pca.fit_transform(scaled_pca_df)

                pca_df = pd.DataFrame(data=principal_components,
                                      index=scaled_pca_df.index, # Preserve index from scaled_pca_df
                                      columns=[f'PC_{i+1}' for i in range(pca_dimensions)])
                
                # Add diagnosis to pca_df for coloring
                pca_df['diagnosis'] = data['diagnosis'].replace({0: "Benign", 1: "Malignant"})

                # Merge original features into pca_df for hover_data
                df_for_plotting = pca_df.merge(data[all_numeric_features], left_index=True, right_index=True, how='left')


                if pca_dimensions == 2:
                    fig_pca = px.scatter(df_for_plotting, # Use the merged DataFrame
                                         x='PC_1',
                                         y='PC_2',
                                         color='diagnosis',
                                         title='PCA - 2 Components',
                                         hover_data=all_numeric_features, # Use original feature names for hover
                                         color_discrete_map={"Benign": "green", "Malignant": "red"},height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)
                elif pca_dimensions == 3:
                    fig_pca = px.scatter_3d(df_for_plotting, # Use the merged DataFrame
                                            x='PC_1',
                                            y='PC_2',
                                            z='PC_3',
                                            color='diagnosis',
                                            title='PCA - 3 Components',
                                            hover_data=all_numeric_features, # Use original feature names for hover
                                            color_discrete_map={"Benign": "green", "Malignant": "red"},height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)
                # Removed previous else warning for dimensions > 3, as radio buttons limit choice.
                
        with pca_info_col: # Explained variance ratio goes into the right column
            # Ensure PCA object exists and dimensions are valid before trying to display variance
            if 'pca' in locals() and pca_dimensions >= 2 and pca_dimensions <= scaled_pca_df.shape[1]: 
                st.markdown(f"**Explained Variance Ratio (Cumulative):** {pca.explained_variance_ratio_.sum():.2f}")
                st.write("Individual Explained Variance Ratio per Component:")
                # Display as a dataframe for better readability
                variance_df = pd.DataFrame(pca.explained_variance_ratio_, 
                                            index=[f'PC_{i+1}' for i in range(pca_dimensions)], 
                                            columns=['Variance Ratio'])
                st.dataframe(variance_df, use_container_width=True)
            else:
                st.info("Select PCA dimensions to view explained variance ratio.")


    st.markdown("---")
    st.markdown("Developed by [Your Name or Team Name]")


if __name__ == "__main__":
    main()
