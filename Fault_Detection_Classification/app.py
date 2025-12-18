import streamlit as st
import joblib
import numpy as np

# Load models
clf = joblib.load("models/classifier.pkl")
reg_dist = joblib.load("models/reg_distance.pkl")
reg_res = joblib.load("models/reg_resistance.pkl")

st.title("Fault Detection & Localization System")

# Input fields
Va = st.number_input("Voltage Va")
Vb = st.number_input("Voltage Vb")
Vc = st.number_input("Voltage Vc")
Ia = st.number_input("Current Ia")
Ib = st.number_input("Current Ib")
Ic = st.number_input("Current Ic")

if st.button("Predict"):
    # Feature engineering
    features = np.array([
        Va, Vb, Vc, Ia, Ib, Ic,
        Va - Vb, Vb - Vc, Ia - Ib, Ib - Ic,
        Ia/Ib if Ib != 0 else 0,
        Ib/Ic if Ic != 0 else 0,
        np.mean([Va, Vb, Vc]),
        np.mean([Ia, Ib, Ic]),
        np.std([Va, Vb, Vc]),
        np.std([Ia, Ib, Ic])
    ]).reshape(1, -1)

    fault = clf.predict(features)[0]
    dist = reg_dist.predict(features)[0]
    res = reg_res.predict(features)[0]

    fault_map = {0: "Healthy", 1: "AB Fault", 2: "ABG Fault", 3: "AG Fault"}
    st.success(f"Fault Type: {fault_map[fault]}")
    st.info(f"Estimated Distance: {dist:.2f} km")
    st.info(f"Estimated Resistance: {res:.2f} Ω")
