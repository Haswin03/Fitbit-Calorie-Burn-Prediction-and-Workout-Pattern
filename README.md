# 🏋️ FITBIT: Intelligent Fitness Analytics 
### *Dual-Stream Machine Learning for Calorie Prediction & Workout Segmentation*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://fitbit-calorie-burn-prediction-and-workout-pattern-clustering.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
**FITBIT** is an end-to-end Machine Learning application designed to decode physiological data into actionable fitness insights. Utilizing the Fitbit dataset, the system features two core engines:

1.  **Calorie Predictor (Supervised):** A high-precision regression engine that estimates energy expenditure during workouts.
2.  **Segment Analyzer (Unsupervised):** A pattern-recognition engine that clusters users into distinct fitness personas based on intensity and physiology.

---

## 🚀 Live Demo
**Check out the live application here:** [https://fitbit-calorie-burn-prediction-and-workout-pattern-clustering.streamlit.app/]

---

## 📊 Performance Benchmarks

### **1. Supervised Learning (Regression)**
The goal was to achieve $R^2 \ge 0.80$ for mobile deployment suitability. The final **Tuned XGBoost** model significantly outperformed this target with laboratory-grade precision.

| Model | $R^2$ Score | MAE (Error) | Status |
| :--- | :---: | :---: | :--- |
| **XGBoost (Tuned)** | **0.9989** | **5.13 kcal** |
| **Random Forest** | 0.9982 | 3.66 kcal |
| **Decision Tree** | 0.9952 | 7.51 kcal |
| **SVR** | 0.9921 | 2.75 kcal |

### **2. Unsupervised Learning (Clustering)**
The system utilized **PCA (Principal Component Analysis)** to reduce 16-dimensional data into 2D "Fitness Universe" coordinates for intuitive user mapping.

| Algorithm | Optimal Clusters | Silhouette Score | Status |
| :--- | :---: | :---: | :--- |
| **DBSCAN** | 2 + Outliers | **0.1718** |
| **Hierarchical** | 2 | 0.1715 |
| **K-Means** | 3 | 0.1701 |

---

## 📁 Project Architecture
The project is structured for production-grade modularity, separating the research "lab" from the production "app":

```text
├── fitness_app.py           # Main Streamlit Dashboard
├── requirements.txt         # Production Dependencies
├── Fitbit_dataset.csv       # Source Workout Data
├── notebooks/               # Research & Training Lab
│   ├── project_3_ML.ipynb   # Supervised Model Training
│   └── unsupervised.ipynb   # Clustering & PCA Analysis
└── pickles/                 # Serialized Brains (17+ Models)
    ├── xgb_tuned.pkl        # Best Regression Model
    ├── kmeans_final.pkl     # Clustering Engine
    └── scaler_sup.pkl       # Feature Normalization

🛠️ Tech Stack  
Language: Python 3.10+  
Modeling: Scikit-Learn, XGBoost  
Data Ops: Pandas, NumPy  
Visualization: Matplotlib, Seaborn, PCA  
Deployment: Streamlit Cloud  
Persistence: Joblib / Pickle  

💡 Key Insights & Methodology  
Feature Engineering: Calculated BMI and Heart Rate Intensity ($HR_i$) to provide the models with deeper physiological context, drastically improving $R^2$ scores.  

Model Persistence: Utilized a centralized `pickles/` directory to manage 17+ pre-trained models, ensuring sub-second inference times in production.  

Dimensionality Reduction: Implemented PCA to visualize high-dimensional workout patterns, allowing users to see their "position" relative to 14,000+ other training sessions in a 2D space.  

📥 Local Setup & Installation  

**Clone the repository:**
```bash
git clone https://github.com/Haswin03/Fitbit-Calorie-Burn-Prediction-and-Workout-Pattern.git
cd Fitbit-Calorie-Burn-Prediction-and-Workout-Pattern
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the application:**
```bash
streamlit run fitness_app.py

👨‍💻 Author  
Ashwanth Ram  
B.E. Computer Science and Engineering  
Chennai, India  
