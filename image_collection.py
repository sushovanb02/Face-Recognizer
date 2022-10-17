import numpy as np
import cv2 as cv

face_classifier = cv.CascadeClassifier('Face-Recognizer-Keras\haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_classifier.detectMultiScale(img,1.3,5)
    if faces is ():
        return None
    
    # Cropping all the faces found
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50,x:x+w+50]

    return cropped_face

# Initialize webcam
cap = cv.VideoCapture(0)
count = 0

# Collecting 300 samples of my face from webcam input
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv.resize(face_extractor(frame),(400,400))

        # Saving file in Images directory with a specified name
        file_name_path = 'Face-Recognizer-Keras/Datasets/Train/Sushovan' + str(count) + '.jpg'
        cv.imwrite(file_name_path,face)

        # Put count on images and display live count
        cv.putText(face,str(count),(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv.imshow('Face Cropper',face)
    
    else:
        print("Face Not Found")
        pass

    if cv.waitKey(1) == 13 or count == 300:
        break

cap.release()
cv.destroyAllWindows()
print("Collecting Samples Complete")
