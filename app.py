import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import warnings
import joblib
warnings.filterwarnings("ignore")

def Town_Name(town_name):
    # Load the fitted encoder
    with open("town_encoding.pkl", "rb") as f:
        town_encoding = pickle.load(f)

    # Convert input to the correct format
    encoded_town = town_encoding.get(town_name, -1)  # Default to -1 if town_name is not found
    return encoded_town

def flat_type(flat):
    flat_list = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE',
                 'MULTI GENERATION', 'MULTI-GENERATION']

    # Create and fit the encoder
    encoder = OrdinalEncoder(categories=[flat_list])
    encoder.fit([[x] for x in flat_list])

    return encoder.transform([[flat]])[0][0]

def storey_range(storey):
    storey_list = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15',
                   '19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30',
                   '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10',
                   '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
                   '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51']

    encoder = LabelEncoder()
    encoder.fit(storey_list)

    return encoder.transform([storey])[0]

def flat_model(model):
    with open("flat_model_encoding.pkl", "rb") as f:
        flat_model = pickle.load(f)

    encoded_model = flat_model.get(model, -1)
    return encoded_model

st.title("Singapore Resale Flat Prices Predicting")

st.write("")

def linear_predict_price(Flat_type, Storey_range, floor_area_sqm, Lease_commence_date, year_sale, town, Flat_Model):
    pd_year = int(year_sale)
    pd_town = Town_Name(town)
    pd_flat_type = flat_type(Flat_type)
    pd_storey_range = np.log1p(storey_range(Storey_range))
    pd_floor_area_sqm = float(floor_area_sqm)
    pd_Lease_commence_date = int(Lease_commence_date)
    pd_Flat_Model = flat_model(Flat_Model)

    with open("linear_regression_model.pkl", "rb") as f:
        Linear_model = pickle.load(f)

    user_data = np.array([[pd_flat_type, pd_storey_range, pd_floor_area_sqm, pd_Lease_commence_date, pd_year, pd_town, pd_Flat_Model]])
    y_pred1 = Linear_model.predict(user_data)
    price = np.exp(y_pred1[0])
    return price

def decision_predict_price(Flat_type, Storey_range, floor_area_sqm, Lease_commence_date, year_sale, town, Flat_Model):
    pd_year = int(year_sale)
    pd_town = Town_Name(town)
    pd_flat_type = flat_type(Flat_type)
    pd_storey_range = np.log1p(storey_range(Storey_range))
    pd_floor_area_sqm = float(floor_area_sqm)
    pd_Lease_commence_date = int(Lease_commence_date)
    pd_Flat_Model = flat_model(Flat_Model)

    with open("DTR_model (1).pkl", "rb") as f:
        decision_model = pickle.load(f)

    user_data = np.array([[pd_flat_type, pd_storey_range, pd_floor_area_sqm, pd_Lease_commence_date, pd_year, pd_town, pd_Flat_Model]])
    y_pred1 = decision_model.predict(user_data)
    price = np.exp(y_pred1[0])
    return price

def xgboost_predict_price(Flat_type, Storey_range, floor_area_sqm, Lease_commence_date, year_sale, town, Flat_Model):
    pd_year = int(year_sale)
    pd_town = Town_Name(town)
    pd_flat_type = flat_type(Flat_type)
    pd_storey_range = np.log1p(storey_range(Storey_range))
    pd_floor_area_sqm = float(floor_area_sqm)
    pd_Lease_commence_date = int(Lease_commence_date)
    pd_Flat_Model = flat_model(Flat_Model)

    with open("XGB_model.pkl", "rb") as f:
        xgboost_model = pickle.load(f)

    user_data = np.array([[pd_flat_type, pd_storey_range, pd_floor_area_sqm, pd_Lease_commence_date, pd_year, pd_town, pd_Flat_Model]])
    y_pred1 = xgboost_model.predict(user_data)
    price = np.exp(y_pred1[0])
    return price

with st.sidebar:
    select = option_menu("Main Menu", ["Home", "Resale_Flat_Input", "About"])
    st.sidebar.write("")

if select == "Home":
    img = Image.open("singapore model.png")
    st.image(img)

    st.header("HDB Flats:")
    st.write('''The majority of Singaporeans live in public housing provided by the HDB.
    HDB flats can be purchased either directly from the HDB as a new unit or through the resale market from existing owners.''')

    st.header("Resale Process:")
    st.write('''In the resale market, buyers purchase flats from existing flat owners, and the transactions are facilitated through the HDB resale process.
    The process involves a series of steps, including valuation, negotiations, and the submission of necessary documents.''')

    st.header("Valuation:")
    st.write('''The HDB conducts a valuation of the flat to determine its market value. This is important for both buyers and sellers in negotiating a fair price.''')

    st.header("Eligibility Criteria:")
    st.write("Buyers and sellers in the resale market must meet certain eligibility criteria, including citizenship requirements and income ceilings.")

    st.header("Resale Levy:")
    st.write("For buyers who have previously purchased a subsidized flat from the HDB, there might be a resale levy imposed when they purchase another flat from the HDB resale market.")

    st.header("Grant Schemes:")
    st.write("There are various housing grant schemes available to eligible buyers, such as the CPF Housing Grant, which provides financial assistance for the purchase of resale flats.")

    st.header("HDB Loan and Bank Loan:")
    st.write("Buyers can choose to finance their flat purchase through an HDB loan or a bank loan. HDB loans are provided by the HDB, while bank loans are obtained from commercial banks.")

    st.header("Market Trends:")
    st.write("The resale market is influenced by various factors such as economic conditions, interest rates, and government policies. Property prices in Singapore can fluctuate based on these factors.")

    st.header("Online Platforms:")
    st.write("There are online platforms and portals where sellers can list their resale flats, and buyers can browse available options.")

elif select == "Resale_Flat_Input":
    col1, col2 = st.columns(2)

    with col1:
        year_sale = st.number_input("Year Sale", min_value=1900, max_value=2099)

        town = st.selectbox("Select the Town", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                               'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                               'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                               'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                               'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                               'TOA PAYOH', 'WOODLANDS', 'YISHUN'])

        Flat_type = st.selectbox("Select the Flat Type", ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE',
                                                          'MULTI GENERATION', 'MULTI-GENERATION'])

        Storey_range = st.selectbox("Select the storey range", ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15',
                                                                '19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30',
                                                                '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10',
                                                                '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
                                                                '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51'])

    with col2:
        floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=1)

        Lease_commence_date = st.number_input("Lease Commence Date", min_value=1900, max_value=2099)

        Flat_Model = st.selectbox("Select the Flat Model", ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                                                            'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                                                            'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                                                            'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                                                            'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'])

    if st.button("Predict Price"):
        prediction1 = linear_predict_price(Flat_type, Storey_range, floor_area_sqm, Lease_commence_date, year_sale, town, Flat_Model)
        st.write(f"Linear model prediction : {prediction1}")
        st.write("")
        prediction2 = decision_predict_price(Flat_type, Storey_range, floor_area_sqm, Lease_commence_date, year_sale, town, Flat_Model)
        st.write(f"Decision Tree model prediction : {prediction2}")
        st.write("")
        prediction5 = xgboost_predict_price(Flat_type, Storey_range, floor_area_sqm, Lease_commence_date, year_sale, town, Flat_Model)
        st.write(f"XGBoost model prediction : {prediction5}")
        st.write("")

    st.header("Sample Data:")
    st.markdown('''- I have included only necessary columns, based on their importance. For actual data, Please refer the link in about.''')
    df1=pd.read_csv("Sample_data.csv")
    st.dataframe(df1)

elif select == "About":
  st.header(":blue[Data Collection and Preprocessing:]")
  st.markdown('''- Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date (950k data), Dataset link - https://beta.data.gov.sg/collections/189/view . Preprocess the data to clean and structure it for machine learning.''')

  st.header(":blue[Feature Engineering:]")
  st.markdown('''- Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.''')
  
  st.header(":blue[Model Selection and Training:]")
  st.markdown('''- Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.''')

  st.header(":blue[Model Evaluation:]")
  st.markdown('''- Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE), R2 Score and Cross Val Score''')

  st.header(":blue[Streamlit Web Application:]")
  st.markdown('''- Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.''')

  st.header(":blue[Deployment on Render:]")
  st.markdown('''- Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.''')
  
  st.header(":blue[Testing and Validation:]")
  st.markdown('''- Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.''')
