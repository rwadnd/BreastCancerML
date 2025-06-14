import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
from modules.nav import Navbar

# Set Streamlit page configuration for the application
st.set_page_config(
    page_title="Breast Cancer Classification App",
    layout="wide",  # Use a wide layout for better content display
    initial_sidebar_state="expanded" # Keep the sidebar expanded by default
)

# Display the navigation bar (assuming modules.nav.Navbar handles the sidebar navigation)
Navbar()

@st.cache_data # Cache the data loading function to prevent re-downloading on every rerun
def load_data():
    """
    Loads and preprocesses the Breast Cancer Wisconsin dataset from Kaggle Hub.
    This function handles data download, column dropping, and diagnosis encoding.
    """
    try:
        with st.spinner("Downloading data from Kaggle... Please wait."):
            file_path = "data.csv"
            # Load the dataset using KaggleDatasetAdapter for pandas
            data = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "uciml/breast-cancer-wisconsin-data",
                file_path,
            )
            # Drop unnecessary columns: 'Unnamed: 32' (often an empty column) and 'id'
            data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
            # Encode 'diagnosis' column: 'M' (Malignant) to 1, 'B' (Benign) to 0
            data["diagnosis"].replace(to_replace={"M": 1, "B": 0}, inplace=True)
            return data
    except Exception as e:
        # Display an error message if data loading fails
        st.error(f"Failed to load data from Kaggle: {e}")
        return None

# Load data into Streamlit's session state if it's not already present
# This ensures the data is loaded only once and is accessible across all pages
if "data" not in st.session_state:
    st.session_state.data = load_data()

# Stop the application execution if data loading failed (e.g., due to network issues)
if st.session_state.data is None:
    st.stop()

# Display the main title of the dashboard
st.title("🏥 Breast Cancer Classification Dashboard")

# Create columns for the welcome message and an illustrative image
col1, _, col2 = st.columns([4,1,2],vertical_alignment="center")

with col1:
    # Display a welcome message with a brief overview of the application's features
    st.markdown("""
    Welcome to the **Breast Cancer Diagnostic Analysis App**

    This application was developed as the graduation project assigned by **Baykar Technologies** , for **"Milli Teknoloji Akademesi, Yapay Zeka Uzmanlık Programı"**. And it allows you to:
    - 🔍 **Explore and visualize** the Breast Cancer Wisconsin dataset
    - ⚙️ **Apply preprocessing techniques** of your choice
    - ⚖️ **Compare machine learning models** interactively
    - 🩺 **Classify custom entries** and interpret predictions

    Use the **sidebar** to navigate through different sections.
    """)

    # Provide information about the dataset source with a link to Kaggle
    st.info("📚 Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set — [View on Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)")

with col2:
    # Display a relevant image (e.g., a breast cancer awareness ribbon)
    st.image("extra/ribbon.svg",width=200)

# Add a horizontal separator for visual distinction
st.markdown("---")
# Display footer information
st.markdown("Developed by **Ravad Nadam** | Powered by Streamlit 💡")
st.markdown("[Github repo](https://github.com/rwadnd/BreastCancerML) | [Email](mailto:ravad.nadam@gmail.com)")