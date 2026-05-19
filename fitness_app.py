import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Fitbit", layout="wide")


@st.cache_resource
def load_assets():
    assets = {}
    current_dir = os.path.dirname(__file__)
    folder_path = os.path.join(current_dir, "pickles")
    
    files = {
        "rf_base": "rf_reg.pkl", "rf_tuned": "rf_tuned.pkl",
        "xgb_base": "xgb_default.pkl", "xgb_tuned": "xgb_tuned.pkl",
        "dt_base": "dt_reg.pkl", "dt_tuned": "dt_tuned.pkl",
        "svr_base": "svr_reg.pkl", "svr_tuned": "svr_tuned.pkl",
        "knn_base": "knn.pkl", "knn_tuned": "knn_tuned.pkl",
        "kmeans": "kmeans_final.pkl",
        "hc": "hc_final.pkl",
        "dbscan": "dbscan_final.pkl",
        "pca": "pca_transformer.pkl",
        "scaler_sup": "scaler_supervised.pkl",
        "scaler_unsup": "scaler_unsupervised.pkl",
        "model_cols": "model_columns.pkl",
        "unsup_ref": "unsup_reference.pkl" 
    }
    
    for key, filename in files.items():
        full_path = os.path.join(folder_path, filename)
        
        if os.path.exists(full_path):
            try:
                with open(full_path, 'rb') as f:
                    assets[key] = pickle.load(f)
            except Exception as e:
                st.sidebar.error(f"Error loading {filename}: {e}")
        else:
            assets[key] = None
            # This will show you EXACTLY where the app is looking
            st.sidebar.warning(f"Missing: {full_path}")
            
    return assets

assets = load_assets()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Calorie Predictor", "Segment Analyzer"])
st.sidebar.markdown("---")
st.sidebar.info("Developed by Ashwanth Ram D | B.E. CSE")

if page == "Home":
    st.title("Welcome to Fitbit : Calorie Prediction & Workout Clustering")
    st.markdown("""
    **Fitbit** is a dual-stream Machine Learning solution.
    
    1. **Calorie Predictor:** Estimates energy expenditure using Tuned Supervised Regression.
    2. **Segment Analyzer:** Identifies user fitness personas using Unsupervised Clustering & PCA.
    """)
    
    st.header("1. Supervised Learning: Calorie Prediction")
    st.markdown("**Target Requirement:** Achieve $R^2$ ≥ 0.80 (Suitable for Mobile Deployment)")

    supervised_data = {
        "Model": ["XGBoost", "Random Forest", "Decision Tree", "SVR", "KNN"],
        "Tuned R² Score": ["0.9989", "0.9982", "0.9952", "0.9921", "0.9470"],
        "MAE (Error)": ["5.13 kcal", "3.66 kcal", "7.51 kcal", "2.75 kcal", "28.25 kcal"]    }
    st.table(pd.DataFrame(supervised_data))

    st.header("2. Unsupervised Learning: Fitness Segmentation")
    st.markdown("**Target Requirement:** Silhouette Score ≥ 0.15")

    unsupervised_data = {
        "Clustering Model": ["DBSCAN", "Hierarchical", "K-Means"],
        "Optimal Clusters (K)": ["2 + Outliers", "2", "3"],
        "Silhouette Score": ["0.1718", "0.1715", "0.1701"]    }
    st.table(pd.DataFrame(unsupervised_data))

    st.info("""
    **System Integrity Note:** Rigorously validated With a peak R² of **0.9989**, the engine provides laboratory-grade prediction.
    """)

# SUPERVISED LEARNING (CALORIE PREDICTOR)

elif page == "Calorie Predictor":
    st.title("AI Calorie Predictor")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 15, 80, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
        fat_pct = st.number_input("Fat Percentage (%)", 5.0, 40.0, 15.0)
    with col2:
        duration = st.number_input("Session Duration (hours)", 0.1, 3.0, 1.0)
        avg_bpm = st.number_input("Average BPM", 60, 200, 130)
        workout_type = st.selectbox("Workout Type", ["HIIT", "Cardio", "Strength", "Yoga"])
        exp_level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Expert"])

    model_choice = st.selectbox("Select ML Engine:", [
        "XGBoost (Tuned) - BEST", "XGBoost (Base)", "Random Forest (Tuned)", 
        "Random Forest (Base)", "Decision Tree (Tuned)", "Decision Tree (Base)",
        "SVR (Tuned)", "SVR (Base)", "KNN (Tuned)", "KNN (Base)"
    ])
    
    model_key_map = {
        "XGBoost (Tuned) - BEST": "xgb_tuned", "XGBoost (Base)": "xgb_base",
        "Random Forest (Tuned)": "rf_tuned", "Random Forest (Base)": "rf_base",
        "Decision Tree (Tuned)": "dt_tuned", "Decision Tree (Base)": "dt_base",
        "SVR (Tuned)": "svr_tuned", "SVR (Base)": "svr_base",
        "KNN (Tuned)": "knn_tuned", "KNN (Base)": "knn_base"
    }

    if st.button("Predict Calories Burned", type="primary"):
        if assets.get(model_key_map[model_choice]) is None:
            st.error("Selected model file is missing.")
        else:
            # Prepare Input
            input_df = pd.DataFrame(0, index=[0], columns=assets['model_cols'])
            exp_map = {"Beginner": 1, "Intermediate": 2, "Expert": 3}
            
            input_df['Age'] = age
            input_df['Gender'] = 1 if gender == "Male" else 0
            input_df['Weight (kg)'] = weight
            input_df['Fat_Percentage'] = fat_pct
            input_df['Session_Duration (hours)'] = duration
            input_df['Avg_BPM'] = avg_bpm
            input_df['Experience_Level'] = exp_map[exp_level]
            
            if f"Type_{workout_type}" in input_df.columns:
                input_df[f"Type_{workout_type}"] = 1
                
            scaled_data = assets['scaler_sup'].transform(input_df)
            prediction = assets[model_key_map[model_choice]].predict(scaled_data)[0]
            st.success(f"### Estimated Burn: {prediction:.2f} kcal")


# UNSUPERVISED LEARNING (SEGMENT ANALYZER)
elif page == "Segment Analyzer":
    st.title("Workout Pattern & Segment Analyzer")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 15, 80, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
        height = st.number_input("Height (m)", 1.2, 2.5, 1.75) 
        fat_pct = st.number_input("Fat Percentage (%)", 5.0, 40.0, 15.0)
    with col2:
        duration = st.number_input("Session Duration (hours)", 0.1, 3.0, 1.0)
        avg_bpm = st.number_input("Average BPM", 60, 200, 130)
        exp_level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Expert"])
        
    cluster_engine = st.selectbox("Select Clustering Engine:", 
                                  ["K-Means (Standard Groups)", "Hierarchical (Nested Segments)", "DBSCAN (Density-Based Outliers)"])

    if st.button("Analyze User Segment", type="primary"):
        # BMI and Intensity calculation
        bmi = weight / (height ** 2)
        max_bpm = 220 - age 
        hr_i = avg_bpm / max_bpm
        
        unsup_features = [age, 1 if gender=="Male" else 0, weight, height, max_bpm, avg_bpm, 70, duration, fat_pct, 2.0, 3, {"Beginner":1, "Intermediate":2, "Expert":3}[exp_level], bmi, 3.5, hr_i, 3.5*hr_i]

        user_array = np.array([unsup_features])
        scaled_user = assets['scaler_unsup'].transform(user_array)
        user_pca_coords = assets['pca'].transform(scaled_user)

        # Logic for selection
        if cluster_engine == "K-Means (Standard Groups)":
            cluster_id = assets['kmeans'].predict(scaled_user)[0]
            engine_name = "K-Means"
        elif cluster_engine == "Hierarchical (Nested Segments)":
            combined = np.vstack([scaled_user, assets['unsup_ref']])
            cluster_id = assets['hc'].fit_predict(combined)[0]
            engine_name = "Hierarchical"
            engine_key = "hc"
        else:
            combined = np.vstack([scaled_user, assets['unsup_ref']])
            cluster_id = assets['dbscan'].fit_predict(combined)[0]
            engine_name = "DBSCAN"
            engine_key = "dbscan"
            if cluster_id == -1: st.warning("This user is an OUTLIER!")

        segment_patterns = {0: "High-Intensity Pattern", 1: "Moderate Pattern", 2: "Endurance Pattern"}
        st.success(f"### Identified: {segment_patterns.get(cluster_id, f'Segment {cluster_id}')}")

        # Visualization
        st.write(f"#### {engine_name} Segmentation Map")
        ref_data = assets['unsup_ref']
        ref_pca = assets['pca'].transform(ref_data)
        
        if cluster_engine == "K-Means (Standard Groups)":
            ref_labels = assets['kmeans'].predict(ref_data)
        else:
            ref_labels = assets[engine_key].fit_predict(ref_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=ref_pca[:, 0], y=ref_pca[:, 1], hue=ref_labels, palette='viridis', alpha=0.3, s=50, ax=ax, legend=False)
        
        if cluster_engine == "K-Means (Standard Groups)":
            centers_pca = assets['pca'].transform(assets['kmeans'].cluster_centers_)
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=300, c='black', marker='o', alpha=0.8, label='Segment Centers')

        ax.scatter(user_pca_coords[:, 0], user_pca_coords[:, 1], s=400, c='red', marker='X', edgecolor='white', linewidth=2, label='★ YOU ARE HERE', zorder=5)
        ax.set_title(f"User Position in the {engine_name} Universe")
        ax.set_xlabel("PC 1 (Workout Intensity)")
        ax.set_ylabel("PC 2 (Physiology / BMI)")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.2)
        st.pyplot(fig)