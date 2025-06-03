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

st.set_page_config(page_title="Breast Cancer Classification App", layout="wide")


def main():
    Navbar()

    if "data" in st.session_state:
        data = st.session_state.data
        st.write("Data is loaded from Kaggle. Shape:", data.shape)
    else:
        st.warning("Data not found. Please return to the main page to load it.")

    # Feature Grouping
    mean_features = [col for col in data.columns if col.endswith('_mean')]
    se_features = [col for col in data.columns if col.endswith('_se')]
    worst_features = [col for col in data.columns if col.endswith('_worst')]


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
        color_discrete_sequence=['orange', 'blue'],
        title="Diagnosis Count (Bar Chart)"
    )
    fig_diag.update_layout(bargap=0.1)

    # Pie Chart
    fig_pie = px.pie(
        diag_counts,
        names='Diagnosis',
        values='Count',
        color='Diagnosis',
        color_discrete_sequence=['orange', 'blue'],
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
        # Selection Mode: Diagnosis-based or Feature-to-Feature
    

        threshold1 = st.slider("Correlation Threshold", 0.3, 1.0, 0.6, 0.05, key="thresh1")
        correlation_mode = st.radio(
            "Select Correlation Type",
            ["Correlation with Diagnosis", "Correlation Among Features"],
            key="correlation_mode"
        )
        # Get correlation matrix
        corr_matrix = data.corr().abs()

        if correlation_mode == "Correlation with Diagnosis":
            selected_features = corr_matrix[corr_matrix['diagnosis'] > threshold1].index.tolist()

            # Always include diagnosis itself
            if "diagnosis" not in selected_features:
                selected_features.append("diagnosis")

            if len(selected_features) < 2:
                st.warning("No features found with the selected threshold. Try lowering the threshold.")
            else:
                filtered_corr = data[selected_features].corr()
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

        else:  # Correlation Among Features
            # Mask lower triangle and diagonal
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            corr_long = corr_matrix.where(mask).stack().reset_index()
            corr_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

            strong_pairs = corr_long[corr_long['Correlation'] > threshold1]

            if strong_pairs.empty:
                st.warning("No feature pairs meet the threshold. Try lowering it.")
            else:
                # Get all involved features
                selected_features = list(set(strong_pairs['Feature 1']).union(set(strong_pairs['Feature 2'])))
                filtered_corr = data[selected_features].corr()
    
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
        abs_corr_sorted = data.corr()[['diagnosis']].abs().sort_values(by='diagnosis', ascending=False)
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

    
    # Histograms
    st.markdown("---")
    st.subheader("Feature Distribution by Diagnosis")
    left_col, right_col = st.columns([4, 1],vertical_alignment="center")

    with right_col:


        # First Dropdown: Feature Type
        feature_type = st.selectbox(
        "Select Feature Type",
        ["mean", "se", "worst"],
        index=0,
        key="feature_type"
    )

    # Filter Features by Type
        available_features = [f.replace("_" + feature_type, '') for f in data.columns if f.endswith("_" + feature_type)]

        # Second Dropdown: Feature Name
        feature_selected_base = st.selectbox(
            "Select Feature",
            available_features,
            index=0,
            key="feature_name"
        )

        # Build Full Feature Name Correctly
        feature_full_name = feature_selected_base + "_" + feature_type
        bin_count = st.slider("Number of bins", min_value=10, max_value=100, value=30, step=5)

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
        
    with left_col:
            st.plotly_chart(fig, use_container_width=False)






    
    st.markdown("---")
    st.subheader("Distribution of Feature Group with KDE (Interactive Grid)")

    # Select feature group
    feature_group = st.selectbox("Select Feature Group to Display", ["mean", "se", "worst"], key="grid_kde_feature_group")

    # Get columns by suffix
    if feature_group == "mean":
        selected_features = mean_features
    elif feature_group == "se":
        selected_features = se_features
    else:
        selected_features = worst_features

    # Layout
    num_features = len(selected_features)
    cols_per_row = 5
    rows = math.ceil(num_features / cols_per_row)

    # Render grid of histogram + KDE overlay
    for row in range(rows):
        row_cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            idx = row * cols_per_row + i
            if idx < num_features:
                feature = selected_features[idx]
                x = data[feature].dropna()

                # Estimate KDE
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




    st.markdown("---")
    # Select features for the pair plot
    st.subheader("Pairwise Feature Relationships")

    # Feature selection (max 12)
    subset_features = st.multiselect(
        "Select 3-9 Features to Visualize (include 'diagnosis' for coloring)",
        options=[col for col in data.columns if col != 'id' and col != 'Unnamed: 32'],
        default=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
    )

    # Input validation
    if 'diagnosis' not in subset_features:
        st.warning("Include 'diagnosis' in your selection for coloring.")
    elif len(subset_features) < 3:
        st.warning("Select at least 2 numeric features + diagnosis to generate the matrix.")
    elif len(subset_features) > 9:
        st.warning("Please select no more than 9 features for better readability.")
    else:
        fig = px.scatter_matrix(
            data_frame=data,
            dimensions=[f for f in subset_features if f != 'diagnosis'],
            color=data['diagnosis'].replace({0: "Benign", 1: "Malignant"}),
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
    st.subheader("Boxplot Grid of Mean Features by Diagnosis")

    # Let user pick which group of features to view
    box_feature_group = st.selectbox("Select Feature Group for Boxplots", ["mean", "se", "worst"], key="box_group")

    if box_feature_group == "mean":
        feature_columns = mean_features
    elif box_feature_group == "se":
        feature_columns = se_features
    else:
        feature_columns = worst_features

    # Grid settings
    n_cols = 5
    n_rows = (len(feature_columns) + n_cols - 1) // n_cols

    # Loop through the grid and show each plot in a cell
    for row in range(n_rows):
        row_cols = st.columns(n_cols)
        for i in range(n_cols):
            idx = row * n_cols + i
            if idx < len(feature_columns):
                feature = feature_columns[idx]
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
    st.markdown("Developed by [Your Name or Team Name]")



if __name__ == "__main__":
    main()
