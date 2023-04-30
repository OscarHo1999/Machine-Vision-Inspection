import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet import preprocess_input


model = load_model("trainmodel.h5")

path = "diamondlogo.png"
img = load_img(path,target_size=(256,256))

i=img_to_array(img)/255
#i=preprocess_input(i)

input_arr=np.array([i])
input_arr.shape

pred = np.argmax(model.predict(input_arr)) #print index value
pred1 = np.max(model.predict(input_arr)) #print sum value
prediction_percentage = pred1*100
prediction_percentage = "{:.4f}".format(prediction_percentage)
print(str(prediction_percentage)+"%")

if pred==0:
    print("bad")
else:
    print("good")

plt.imshow(input_arr[0])
plt.title("input image")
plt.axis = False
plt.show()

