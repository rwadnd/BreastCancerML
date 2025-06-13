# 🩺 Breast Cancer Classification Dashboard

This interactive web application was developed as part of the **Baykar Milli Teknoloji Akademesi - Yapay Zeka Uzmanlık Programı** graduation project. It enables users to explore the **Breast Cancer Wisconsin (Diagnostic) Dataset**, apply various preprocessing techniques, compare machine learning and deep learning models, and make real-time predictions on custom entries.

🚀 **Live App**: <a href="https://breastcancerml.streamlit.app" target="_blank">breastcancerml.streamlit.app</a>

> 📚 This project includes:
> - A fully interactive **Streamlit dashboard**
> - A **Jupyter Notebook** for standalone analysis
> - A formal **research report** summarizing methodology and results

---

## 🧪 Dataset

- **Source**: [Kaggle - Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 569 samples, 30 numeric features extracted from digitized images of fine needle aspirates of breast masses
- Labels:
  - `M` = Malignant (converted to 1)
  - `B` = Benign (converted to 0)

---

## 🔍 Features

### 📊 1. Exploratory Data Analysis
- Diagnosis distribution visualized via bar and pie charts
- Correlation heatmaps (with diagnosis or among features)
- Feature grouping (mean, SE, worst)
- Interactive KDE and histograms
- PCA visualization (2D & 3D)
- Pairplot-style matrix & boxplot grid

### ⚙️ 2. Preprocessing & Model Training
- **Preprocessing options**:
  - Outlier removal (IQR, Z-Score, Winsorization)
  - Imputation (Mean, Median, KNN)
  - Feature scaling (MinMax, Standard, Robust)
  - Feature selection (SelectKBest, RFE, PCA)
  - Imbalance handling (SMOTE, undersampling)
- **Machine Learning models**: Logistic Regression, SVM, RF, XGBoost, LightGBM, and more
- **Deep Learning**: Configurable Keras Sequential model with dropout, optimizer, learning rate, etc.

### 🩺 3. Custom Prediction Interface
- Use sliders or random entry from dataset
- Classify with up to 9 ML models simultaneously
- Real-time prediction with confidence score

---

## 🛠️ Setup Instructions

### Requirements

- Python (Preferably 3.11)
- Streamlit
- scikit-learn
- TensorFlow
- plotly
- pandas, numpy
- imbalanced-learn
- kagglehub (for dataset import)

### Installation

```bash
git clone https://github.com/your-username/breast-cancer-streamlit.git
cd breast-cancer-streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> ⚠️ You must have a Kaggle API token and credentials to download the dataset via `kagglehub`. Set it up as shown in the [kagglehub documentation](https://github.com/MLH-Fellowship/kagglehub).

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## 👨‍💻 Author

Developed by **Ravad Nadam**  
Contact: <a href="https://www.linkedin.com/in/ravad-nadam/" target="_blank">LinkedIn</a> • [Email](mailto:ravad.nadam@gmail.com)



