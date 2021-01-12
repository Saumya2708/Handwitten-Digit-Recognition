import tensorflow as tf
from tensorflow.keras.datasets import mnist

(X_train,y_train),(X_test,y_test)=mnist.load_data()

import matplotlib.pyplot as plt

print(y_train[35986])

plt.imshow(X_train[35986],cmap='Greys')

X_test.shape

X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)

input_shape=(28,28,1)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255
X_test/=255

X_train.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

model=Sequential()

model.add(Convolution2D(32,kernel_size=(3,3),input_shape=(28,28,1), activation='relu'))
model.add(Convolution2D(64,(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.summary()

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))

model.summary()

model.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')

model.fit(x=X_train,y=y_train,epochs=10)

model.evaluate(X_test,y_test)

img_in=36
plt.imshow(X_test[img_in].reshape(28,28),cmap='Greys')

pred=model.predict(X_test[img_in].reshape(1,28,28,1))

print(pred.argmax())