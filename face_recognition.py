from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model,load_model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

# Resizing all images

IMAGE_SIZE = [224,224]

train_path = 'Face Recognition using Keras\Datasets\Train'
test_path = 'Face Recognition using Keras\Datasets\Test'

# adding preprocessing layer to the front of VGG

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Not training existing weights

for layer in vgg.layers:
    layer.trainable = False

# Useful for getting number of classes

folders = glob('Face Recognition using Keras/Datasets/*')

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)

# Creating a model object

model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

# Telling the model what cost and optimization methood to use

model.compile(loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Face Recognition using Keras\Datasets\Train', target_size=(224,224), batch_size=32, class_mode= 'categorical')
test_set = test_datagen.flow_from_directory('Face Recognition using Keras\Datasets\Test', target_size=(224,224), batch_size=32, class_mode= 'categorical')

# Fitting the model

r = model.fit(training_set, validation_data=test_set, epochs=5, steps_per_epoch=len(training_set), validation_steps=len(test_set))

# Loss

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# Accuracies

plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model.save('facefeatures_model1.h5')