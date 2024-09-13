import tensorflow as tf
import scipy
#import keras_preprocessing
#from keras_preprocessing import image
#from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "Treinamento/"
training_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    batch_size = 32,
    class_mode='categorical'
)

VALIDATION_DIR = "Validar/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    batch_size = 32,
    class_mode='categorical'
)

model = tf.keras.models.Sequential([ #Definicao de rede neural
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dense(3, activation='softmax') #3 Neuronios pq existem 3 classes
    tf.keras.layers.Dense(2, activation='softmax') #2 Neuronios pq existem 2 classes
])

model.compile(loss = 'categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(train_generator, epochs = 35,
                              validation_data = validation_generator,
                              verbose = 1)

model.save('model.keras')