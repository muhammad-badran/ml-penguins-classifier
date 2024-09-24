import streamlit as st
import pandas as pd

st.title('Machine Learning App to Classify Penguins')

st.info('This is a mahcine learning app that builds a model to predict the spieces of penguins.')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/muhammad-badran/ml-penguins-classifier/refs/heads/main/penguins.csv')
  df

  st.write('The inputs features (X)')
  X = df.drop('species', axis=1)
  X

  st.write('The output vector (y)')
  y = df.species
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.sidebar:
  st.write('Input Feature')
  island = st.selectbox('Island:', ('Biscoe', 'Dream', 'Torgersen'))
  gender = st.selectbox('Sex:', ('male', 'female'))
  bill_length_mm = st.slider('Bill Length (mm):', 32.1, 69.6, 43.9)
  bill_depth_mm = st.slider('Bill Depth (mm):', 13.1,  21.5, 17.2)
  flipper_length_mm = st.slider('Flipper Length (mm):', 172.0 , 231.0, 201.0)
  body_mass_g = st.slider('Body Mass (g):', 2700.0, 6300.0, 4207.0) 

input_data = {'island': island,
              'bill_enght_mm': bill_length_mm,
              'bill_depth_mm': bill_depth_mm,
              'flipper_length_mm': flipper_length_mm,
              'body_mass_g': body_mass_g }
              
              
