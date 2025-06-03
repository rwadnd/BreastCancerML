import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
from modules.nav import Navbar

# Set page config
st.set_page_config(
    page_title="Breast Cancer Classification App",
    layout="wide",
    initial_sidebar_state="expanded"
)

Navbar()

@st.cache_data
def load_data():
    try:
        with st.spinner("Downloading data from Kaggle... Please wait."):
            file_path = "data.csv"
            data = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "uciml/breast-cancer-wisconsin-data",
                file_path,
            )
            data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
            data["diagnosis"].replace(to_replace={"M": 1, "B": 0}, inplace=True)
            return data
    except Exception as e:
        st.error(f"Failed to load data from Kaggle: {e}")
        return None


data = load_data()
if data is None:
    st.stop()

if "data" not in st.session_state:
    st.session_state.data = load_data()



# Title
st.title("🏥 Breast Cancer Classification Dashboard")

# Welcome Message
st.markdown("""
Welcome to the **Breast Cancer Diagnostic Analysis App** built with Streamlit.

This application allows you to:
- 🔍 **Explore and visualize** the Breast Cancer Wisconsin dataset
- ⚙️ **Apply preprocessing techniques** of your choice
- ⚖️ **Compare machine learning models** interactively
- 🩺 **Classify custom entries** and interpret predictions

Use the **sidebar** to navigate through different sections.
""")

# Optional illustration or dataset source
st.info("📚 Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set — [View on Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)")

# Optional footer
st.markdown("---")
st.markdown("Developed by **Your Name or Team Name** | Powered by Streamlit 💡")
