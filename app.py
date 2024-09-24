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

