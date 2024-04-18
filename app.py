import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict(data):
    clf = joblib.load("DecisionTree_Model.sav")
    return clf.predict(data)

# Function to map classes to images
def class_to_image(class_name):
    if class_name == "minor":
        return "images/minor.jpg"  
    elif class_name == "moderate":
        return "images/moderate.jpg"  
    elif class_name == "serious":
        return "images/serious.jpg"  
    elif class_name == "fatal":
        return "images/fatal.jpeg"  

st.title('Classifying Road Accident Severity')
st.markdown('Model to classify road accidents severity into \
     (Minor, Moderate, Serious, Fatal) based on their Temperature(F), \
     Visibility(mi), Pressure(in), Precipitation(in), Humidity(%), \
     Wind chill(F), and Wind speed(mph).')

st.header("Road Accidents Features")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.text("Wind Characteristics")
    wind_c = st.slider('Wind chill(F)', 33.0, 97.0, 50.0)
    wind_s = st.slider('Wind speed(mph)', 0.0, 33.0, 15.0)

with col2:
    st.text("Weather Characteristics")
    humid = st.slider('Humidity(%)', 1.0, 100.0, 50.0)
    prec = st.slider('Precipitation(in)', 0.0, 2.2, 1.0)

with col3:
    st.text("Environment Characteristics")
    temp = st.slider('Temperature(F)', 40.0, 102.0, 70.0)
    press = st.slider('Pressure(in)', 29.52, 30.51, 30.0)

with col4:
    st.text("Visibility Characteristic")
    visib = st.slider('Visibility(mi)', 0.0, 10.0, 5.0)

st.text('')
if st.button("Predict Accident Severity"):
    result = predict(
        np.array([[temp, visib, press, prec, humid, wind_c, wind_s]]))
    st.text(result[0])

    # Display the image/icon corresponding to the predicted class
    image_path = class_to_image(result[0].split("-")[1])
    st.image(image_path, use_column_width=True)


st.text('')
