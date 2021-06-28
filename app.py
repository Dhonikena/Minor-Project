import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import joblib
from PIL import Image


st.title('Image Classification Deployment')
st.text('Upload the Image')

model = joblib.load('img_model')

uploaded_file = st.file_uploader("Choose an image..." , type="jpg" )
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image')

  if st.button('PREDICT'):
    CATEGORIES = ['polar bear species','penguin species']
    st.write('Result...')
    flat_data = []   
    img = np.array(img)
    img_resized = resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.write(f'PREDICTED OUTPUT : {y_out}') 
    r = model.predict_proba(flat_data)
    for index, item in enumerate(CATEGORIES):
      st.write(f'{item} : { r[0][index]*100}')



 
