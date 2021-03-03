import streamlit as st
import pandas as pd
import pickle
from PIL import Image
# model = pickle.load(open('IRIS-model.pkl', 'rb'))
import pickle
model = pickle.load( open( "save.p", "rb" ) )

st.header("Iris Classification:")
st.set_option('deprecation.showImageFormat', False)
image = Image.open('image.png')
st.image(image, use_column_width=True,format='PNG')
st.write("Please insert values, to get Iris class prediction")

SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
PetalWidthCm = st.slider('PetalWidthCm:', 0.0, 2.0)
data = {'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm}

features = pd.DataFrame(data, index=[0])

pred_proba = model.predict_proba(features)
#or
prediction = model.predict(features)

st.subheader('Prediction Percentages:') 
st.write('**Probablity of Iris Class being Iris-setosa is ( in % )**:',pred_proba[0][0]*100)
st.write('**Probablity of Isis Class being Iris-versicolor is ( in % )**:',pred_proba[0][1]*100)
st.write('**Probablity of Isis Class being Iris-virginica ( in % )**:',pred_proba[0][2]*100)


##    run folowing command in terminal 

# cd /home/sumon/Data_Science/APP/Simple-Iris-Classifier-Web-App-using-Streamlit-Youtube-main/my
# streamlit run app21.py

