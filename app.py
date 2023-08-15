import streamlit as st
from joblib import load
import numpy as np

model = load("iris_model.joblib")

st.title("Iris Flower")


sepal_length = st.slider("Sepal Length: ", 4.3, 7.9, 5.4)
sepal_width = st.slider("Sepal width: ", 2.0, 4.4, 3.4)
petal_length = st.slider("petal Length: ", 1.0, 6.9, 1.3)
petal_width = st.slider("petal width: ", 0.1, 2.5, 0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button('Predict'):
    prediction = model.predict(features)

    st.write('The flower is ', prediction[0])


## Run app 
## Cd to this path
## streamlit run app.py