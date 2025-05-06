# 🏠 Singapore Resale Flat Price Predictor

## 📌 Problem Statement

The objective of this project is to develop a **machine learning model** and deploy it as a **user-friendly web application** that predicts the **resale prices of flats in Singapore**. The model is trained using **historical data from HDB resale transactions**, assisting potential buyers and sellers in estimating flat resale values based on key features.

---

## 💡 Motivation

Singapore's resale flat market is **highly dynamic and competitive**. Accurately estimating a flat’s resale value is a challenge due to various influencing factors such as:

- Town / Location
- Flat Type and Storey Range
- Floor Area (sqm)
- Flat Model
- Lease Commencement Date

A data-driven approach through machine learning enables smarter decision-making and helps reduce market inefficiencies for all stakeholders.

---

## 📦 Scope of the Project

The project involves the following key steps:

1. **Data Collection and Preprocessing**  
   - Source: HDB resale flat prices from 1990 to present  
   - Clean and structure the data for analysis and modeling

2. **Feature Engineering**  
   - Extract important features such as:  
     `town`, `flat_type`, `storey_range`, `floor_area_sqm`, `flat_model`, `lease_commence_date`  
   - Generate new features to improve model accuracy

3. **Model Selection and Training**  
   - Apply regression algorithms like Linear Regression, Decision Trees, Random Forests, etc.  
   - Use cross-validation and train-test split for evaluation

4. **Model Evaluation**  
   - Metrics used:  
     `MAE`, `MSE`, `RMSE`, and `R² Score`  
   - Select the best-performing model

5. **Web Application Development (Streamlit)**  
   - Build an interactive UI with form inputs for user to enter flat details  
   - Predict and display resale prices using the trained ML model

6. **Deployment (Render / Cloud Platform)**  
   - Host the app on [Render](https://render.com/) (or any cloud platform)  
   - Make it accessible via public URL

7. **Testing and Validation**  
   - Ensure application stability, input validation, and accurate predictions  

---

## 🚀 Deliverables

- ✅ A machine learning model trained on HDB resale data  
- ✅ A responsive web app built using Streamlit / Flask / Django  
- ✅ Deployed app accessible on the web via Render / cloud service  
- ✅ Clear documentation and usage instructions  
- ✅ Project report detailing end-to-end implementation  

---

## 🛠️ Tech Stack

- **Languages & Frameworks**: Python, Streamlit (or Flask/Django)
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn
- **Modeling**: Regression algorithms
- **Deployment**: Render / Cloud Hosting
- **Version Control**: Git & GitHub

---

## 📂 Project Structure (Example)

<pre lang="text"> resale-price-predictor/ │ ├── data/ # Raw and processed datasets │ ├── raw/ # Original HDB resale data files │ └── processed/ # Cleaned and feature-engineered data │ ├── notebooks/ # Jupyter notebooks for EDA and modeling │ └── 01_eda_and_modeling.ipynb │ ├── src/ # Source code for core functionality │ ├── __init__.py │ ├── data_preprocessing.py # Data cleaning & transformation │ ├── feature_engineering.py # Feature extraction & encoding │ ├── model_training.py # Model training & evaluation │ └── utils.py # Helper functions │ ├── models/ # Trained models (pickle/joblib) │ └── resale_price_model.pkl │ ├── app/ # Web application (Streamlit / Flask) │ ├── __init__.py │ ├── app.py # Main web app script │ └── templates/ # For HTML templates (Flask/Django) │ ├── deployment/ # Deployment configuration files │ ├── render.yaml # Render.com configuration │ └── Dockerfile # (Optional) Docker container setup │ ├── tests/ # Unit tests for code and model │ └── test_model.py │ ├── requirements.txt # Python dependencies ├── README.md # Project documentation ├── report.pdf # Final project report └── .gitignore # Git ignored files </pre>
