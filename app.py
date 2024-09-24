import streamlit as st
import pandas as pd

st.title('Machine Learning App to Classify Penguins')

st.info('This is a mahcine learning app that builds a model to predict the spieces of penguins.')

df = pd.read_csv('https://raw.githubusercontent.com/muhammad-badran/ml-penguins-classifier/refs/heads/main/penguins.csv')
df

st.write('Hello world!')
