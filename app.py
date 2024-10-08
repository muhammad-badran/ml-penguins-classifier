import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Machine Learning App: Penguins Species Prediction')

st.info('This is a mahcine learning app that builds a model to predict the spieces of penguins.')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/muhammad-badran/ml-penguins-classifier/refs/heads/main/penguins.csv')
  df

  st.write('The inputs features (X)')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('The output vector (y)')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
  
with st.sidebar:
  st.header('Input Feature')
  island = st.selectbox('Island:', ('Biscoe', 'Dream', 'Torgersen'))
  gender = st.selectbox('Sex:', ('male', 'female'))
  bill_length_mm = st.slider('Bill Length (mm):', 32.1, 69.6, 43.9)
  bill_depth_mm = st.slider('Bill Depth (mm):', 13.1,  21.5, 17.2)
  flipper_length_mm = st.slider('Flipper Length (mm):', 172.0 , 231.0, 201.0)
  body_mass_g = st.slider('Body Mass (g):', 2700.0, 6300.0, 4207.0) 

# Create a DataFrame for the input features
input_data = {'island': island,
              'bill_length_mm': bill_length_mm,
              'bill_depth_mm': bill_depth_mm,
              'flipper_length_mm': flipper_length_mm,
              'body_mass_g': body_mass_g,
              'gender': gender
             }
              
input_penguin = pd.DataFrame(input_data, index=[0])
input_penguins = pd.concat([input_penguin, X_raw], axis=0)

with st.expander('Input Features'):
  st.write('**Input Penguin**')
  input_penguin

# Data Preparation
# Encode categorical data in X
encode = ['island', 'gender']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

# Encode y
target_mapper = {'Adelie': 0,
                'Chinstrap': 1,
                'Gentoo': 2
                }

def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('**Data Preparation**'):
  st.write('**Encode X (input penguin)**')
  input_row = df_penguins[:1]
  input_row
  st.write('**Combined Data**')
  df_penguins
  st.write('**Encoded X**')
  df_penguins[1:]
  st.write('**Encoded y**')
  y

# Model Training and Inference
clf = RandomForestClassifier()
clf.fit(df_penguins[1:],y)

pred = clf.predict(input_row) # gives the predicted class
prediction_proba = clf.predict_proba(input_row) # gives the probability of each class

df_result = pd.DataFrame(prediction_proba)
df_result.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_result = df_result.rename(columns={'0':'Adelie', '1': 'Chinstrap', '2':'Gentoo'})

# Display the predicted speices
st.subheader('Predict Species')
st.dataframe(df_result,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               )
             }, hide_index=True)
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[pred][0]))

sp = df['species'].value_counts()
sp

