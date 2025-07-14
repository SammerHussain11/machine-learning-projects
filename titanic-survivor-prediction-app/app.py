import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survivors Prediction")

st.write("Enter passenger details to predict survival:")

# Update input field names to match your model features
passenger_class = st.selectbox("Passenger Class (PassengerClass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibling_spouse = st.number_input("Number of Siblings/Spouses (SiblingSpouse)", min_value=0, max_value=10, value=0)
parent_child = st.number_input("Number of Parents/Children (ParentChild)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

# Convert categorical variables to numerical (example encoding)
sex_num = 1 if sex == "female" else 0
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked_num = embarked_dict[embarked]

# Prepare input for prediction (order/features as per your model)
input_features = np.array([[passenger_class, sex_num, age, sibling_spouse, parent_child, fare, embarked_num]])

if st.button("Predict Survival"):
    prediction = model.predict(input_features)
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    st.success(f"Prediction: {result}")