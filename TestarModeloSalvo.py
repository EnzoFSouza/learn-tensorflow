import tensorflow
import numpy as np
from keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.models import load_model

#Modelo para predizer numeros
modelo_numeros = load_model('model.keras')
print(modelo_numeros.predict([9]))

#Modelo para predizer imagens
modelo_imagens = load_model('modelImagem.keras')

#carregando imagem
img = image.load_img("ValidarBorracha.jpeg", target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
imagens = np.vstack([x])

classes = modelo_imagens.predict(imagens, batch_size = 10)
print(classes)
if (classes[0][0] < classes[0][1]) and (classes[0][2] < classes[0][1]):
  print("Borracha")

elif (classes[0][0] < classes[0][2]) and (classes[0][1] < classes[0][2]):
  print("Lapis")
  
else:
  print("Apontador")