import streamlit as st
import pandas as pd
import pickle

# Load the trained model, label encoder, and scaler
@st.cache_resource
def load_model():
    with open('dehydration_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_label_encoder():
    with open('label_encoder.pkl', 'rb') as file:
        le = pickle.load(file)
    return le

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

model = load_model()
le = load_label_encoder()
s_scaler = load_scaler()

st.title('Dehydration Status Prediction App')
st.write('Enter patient details to predict dehydration status.')

# Input fields for features
age = st.number_input('Age (years)', min_value=1, max_value=120, value=30)
gender_options = ['Female', 'Male']
gender_selected = st.selectbox('Gender', gender_options)
serum_sodium = st.number_input('Serum Sodium (mEq/L)', min_value=100.0, max_value=200.0, value=140.0, format='%.2f')
serum_potassium = st.number_input('Serum Potassium (mEq/L)', min_value=2.0, max_value=10.0, value=4.0, format='%.2f')
urine_output = st.number_input('Urine Output (mL/kg/hr)', min_value=0.0, max_value=2.0, value=0.5, format='%.3f')
bun = st.number_input('Blood Urea Nitrogen (mg/dL)', min_value=0.0, max_value=100.0, value=20.0, format='%.1f')
hematocrit = st.number_input('Hematocrit (%)', min_value=20.0, max_value=70.0, value=45.0, format='%.1f')
urine_sg = st.number_input('Urine Specific Gravity', min_value=1.000, max_value=1.050, value=1.020, format='%.4f')


if st.button('Predict Dehydration Status'):
    # Preprocess input data
    gender_encoded = 1 if gender_selected == 'Male' else 0

    input_data = pd.DataFrame([[age, serum_sodium, serum_potassium, urine_output, bun, hematocrit, urine_sg, gender_encoded]],
                              columns=['Age (years)', 'Serum Sodium (mEq/L)', 'Serum Potassium (mEq/L)',
                                       'Urine Output (mL/kg/hr)', 'Blood Urea Nitrogen (mg/dL)', 'Hematocrit (%)',
                                       'Urine Specific Gravity', 'Gender_Male'])

    # Identify numerical columns for scaling
    numerical_cols = ['Age (years)', 'Serum Sodium (mEq/L)', 'Serum Potassium (mEq/L)',
                      'Urine Output (mL/kg/hr)', 'Blood Urea Nitrogen (mg/dL)',
                      'Hematocrit (%)', 'Urine Specific Gravity']

    # Scale numerical features
    input_scaled = s_scaler.transform(input_data[numerical_cols])
    
    # Create a DataFrame for scaled numerical features
    input_scaled_df = pd.DataFrame(input_scaled, columns=numerical_cols)
    
    # Combine scaled numerical features with encoded gender
    processed_input = pd.concat([input_scaled_df, input_data[['Gender_Male']]], axis=1)


    # Make prediction
    prediction_encoded = model.predict(processed_input)
    prediction_label = le.inverse_transform(prediction_encoded)

    st.success(f'Predicted Dehydration Status: {prediction_label[0]}')
