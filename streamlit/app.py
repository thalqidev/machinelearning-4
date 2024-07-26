import pandas as pd
import numpy as np
import streamlit as st
import pickle

st.write("""
## Prediksi Kualitas Tidur Mahasiswa
""")

# Load Model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('rfe.pkl', 'rb') as file:
    rfe_selector = pickle.load(file)

with open('dt_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)

occup_lib = {
    'Accountant':0,
    'Doctor':1,
    'Engineer':2,
    'Lawyer':3,
    'Manager':4,
    'Nurse':5,
    'Sales Representative':6,
    'Salesperson':7,
    'Scientist':8,
    'Software Engineer':9,
    'Teacher':10
}

bmi_lib = {
    'Normal':0,
    'Normal Weight':1,
    'Obese':2,
    'Overweight':3
}

gen_lib = {
    'Male':1,
    'Female':2
}

sleep_dis_lib = {
    'Insomnia':0,
    'No':1,
    'Sleep Apnea':2,
    0:'Insomnia',
    1:'No',
    2:'Sleep Apnea'
}


col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Enter Age:", format="%.0f",
                          value = None)
    gender_cat = st.selectbox('Choose Gender:', 
                            options = gen_lib.keys(), 
                            index = None,
                            placeholder = "Select contact gender...")
    
    occupation_cat = st.selectbox('Choose Ocupation:', 
                               options = occup_lib.keys(), 
                               index = None,
                               placeholder = "Select contact gender...")
    
    

with col2:
    sleep_dur = st.number_input("Enter Sleep Duration (hour):",
                                format="%.1f",
                                placeholder = "Input Sleep Duration...",
                                value = None)
    sleep_quality = st.number_input("Enter Sleep Quality (1-10):",
                                    format="%.0f",
                                    placeholder = "Input Sleep Quality...",
                                    value = None)
    physical_activity_level = st.number_input("Enter Physical Activity Level (0-100):",
                                              format="%.0f",
                                              placeholder = "Input Physical Activity Level...",
                                              value = None)
    stress_level = st.number_input("Enter Stress Level (0-10):",
                                   format="%.0f",
                                   placeholder = "Input Stress Level...",
                                   value = None)
    bmi_cat = st.selectbox('Choose BMI Category:', 
                        options = bmi_lib.keys(), 
                        index = None,
                        placeholder = "Select contact gender...")

with col3:
    blood_pressure = st.text_input("Enter Blood Pressure:",
                                   placeholder = "Input Blood Pressure...")
    heart_rate = st.number_input("Enter Heart Rate:",
                                 format="%.0f",
                                 placeholder = "Input Heart Rate...",
                                 value = None)
    daily_steps = st.number_input("Enter Daily Steps:",
                                  format="%.0f",
                                  placeholder = "Input Daily Steps...",
                                  value = None)

if st.button("Predict"):
    blood_pressure = float(blood_pressure.split("/")[0])/float(blood_pressure.split("/")[1])
    gender_int = gen_lib[gender_cat]
    occupation_int = occup_lib[occupation_cat]
    bmi_int = bmi_lib[bmi_cat]

    data = np.array([[
        gender_int, age, occupation_int, sleep_dur,
        sleep_quality, physical_activity_level, stress_level, bmi_int,
        blood_pressure, heart_rate, daily_steps
    ]])

    scale_data = scaler.transform(data)
    selected_data = rfe_selector.transform(scale_data)
    predict = dt_model.predict(selected_data)
    predict = sleep_dis_lib[predict[0]]
    st.write(f"Sleep Disorder: **{predict}**")
    st.write(f"{list(selected_data)}")
    st.write(f"{list(scale_data)}")
    


