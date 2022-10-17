from PIL import Image
import numpy as np
import base64
from io import BytesIO
import json
import random
import cv2 as cv
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import streamlit as st

# st.title('Face Recognition System')
st.markdown("<h1 style='text-align: center; color: #7FCDCD;'>Face Recognition System</h1>", unsafe_allow_html=True)
# run = st.button(label="Run Video")
# stop = st.button(label="Stop Video")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    pass
with col2:
    run = st.button(label="Turn on Video!")
with col4:
    stop = st.button(label="Turn off Video!")
with col5:
    pass
with col3:
    pass

FRAME_WINDOW = st.image([],channels='BGR')

model = load_model(r'Face Recognition using Keras\facefeatures_model1.h5')

# Loading the cascades

face_cascade = cv.CascadeClassifier('Face Recognition using Keras\haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Face Recognition with the webcam
video_capture = cv.VideoCapture(0)

while (stop):
    video_capture.release()

while run:
    _, frame = video_capture.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')

        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        name="None matching"
        if(pred[0][0]>0.8):
            name='Aniket'
        if(pred[0][1]>0.8):
            name='Kushagra'
        if(pred[0][2]>0.8 ):
            name='Sushovan'
        
        cv.putText(frame,name, (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv.putText(frame,"No face found", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    # cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    FRAME_WINDOW.image(frame)

cv.destroyAllWindows()
