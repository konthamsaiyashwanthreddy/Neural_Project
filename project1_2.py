from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D,BatchNormalization

batch_size = 32
num_classes = 10
#epochs = 1600
data_augmentation = True

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""CASE 1 - relu,softmax,rmsprop,50 epochs"""

model = Sequential()

model.add(Conv2D(32, (3, 3),kernel_initializer='random_normal',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),kernel_initializer='random_normal',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),kernel_initializer='random_normal',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3),kernel_initializer='random_normal',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3),kernel_initializer='random_normal',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(GlobalMaxPooling2D())
model.add(Dense(500))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

#opt = keras.optimizers.RMSProp(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer="RMSProp",
              metrics=['accuracy'])

m1=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=50,
              validation_data=(x_test, y_test),
              shuffle=True)

import matplotlib.pyplot as plt
import seaborn as sb

import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(10, 6))
sb.set_style("whitegrid")

# Check if 'accuracy' or 'acc' key is present in the history dictionary
if 'accuracy' in m1.history:
    plt.plot(m1.history['accuracy'], color="#E74C3C", marker='o', label='Training Accuracy')
if 'val_accuracy' in m1.history:
    plt.plot(m1.history['val_accuracy'], color='#641E16', marker='h', label='Validation Accuracy')

plt.title('Accuracy comparison between Validation and Train Data set', fontsize=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# Add legend only if there are labeled elements in the plot
if 'accuracy' in m1.history or 'val_accuracy' in m1.history:
    plt.legend()

plt.show()

plt.figure(figsize=(10,6))
sb.set_style("whitegrid")
plt.plot(m1.history['loss'],color="Purple",marker='o')
plt.plot(m1.history['val_loss'],color='Orange',marker='h')
plt.title('Loss comparison between Validation and Train Data set',fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

"""CASE 2 - relu,softmax,adam,epochs-100"""

model2 = Sequential()

model2.add(Conv2D(32, (3, 3),kernel_initializer='random_uniform',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(32, (3, 3),kernel_initializer='random_uniform',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(32, (3, 3),kernel_initializer='random_uniform',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(48, (3, 3),kernel_initializer='random_uniform',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(48, (3, 3),kernel_initializer='random_uniform',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(48, (3, 3),kernel_initializer='random_uniform',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(64, (3, 3),kernel_initializer='random_uniform',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(128, (3, 3),kernel_initializer='random_uniform',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(GlobalMaxPooling2D())
model2.add(Dropout(0.25))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))
model2.summary()

opt = keras.optimizers.Adamax(lr=0.001)
model2.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

m2=model2.fit(x_train, y_train,
              batch_size=500,
              epochs=100,
              validation_data=(x_test, y_test),
              shuffle=True)

plt.figure(figsize=(10, 6))
sb.set_style("whitegrid")

# Check if 'accuracy' key is present in the history dictionary
if 'accuracy' in m2.history:
    plt.plot(m2.history['accuracy'], color="#E74C3C", marker='o')
if 'val_accuracy' in m2.history:
    plt.plot(m2.history['val_accuracy'], color='#641E16', marker='h')

plt.title('Accuracy comparison between Validation and Train Data set', fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

plt.figure(figsize=(10, 6))
sb.set_style("whitegrid")
plt.plot(m2.history['loss'],color="#E74C3C",marker='o')
plt.plot(m2.history['val_loss'],color='#641E16',marker='h')
plt.title('Loss comparison between Validation and Train Data set',fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

"""CASE 3-tanh,softmax,epochs-100,adamax"""

model3 = Sequential()

model3.add(Conv2D(32, (3, 3),kernel_initializer='glorot_uniform',
                 input_shape=x_train.shape[1:]))
model3.add(Activation('tanh'))
model3.add(Conv2D(32, (3, 3),kernel_initializer='glorot_uniform',
                 input_shape=x_train.shape[1:]))
model3.add(Activation('tanh'))
model3.add(Conv2D(32, (3, 3),kernel_initializer='glorot_uniform',
                 input_shape=x_train.shape[1:]))
model3.add(Activation('tanh'))
model3.add(Conv2D(48, (3, 3),kernel_initializer='glorot_uniform',
                 input_shape=x_train.shape[1:]))
model3.add(Activation('tanh'))
model3.add(Conv2D(48, (3, 3),kernel_initializer='glorot_uniform',
                 input_shape=x_train.shape[1:]))
model3.add(Activation('tanh'))
model3.add(Conv2D(48, (3, 3),kernel_initializer='glorot_uniform',
                 input_shape=x_train.shape[1:]))
model3.add(Activation('tanh'))
model3.add(Conv2D(48, (3, 3),kernel_initializer='glorot_uniform',
                 input_shape=x_train.shape[1:]))
model3.add(Activation('tanh'))
model3.add(GlobalMaxPooling2D())
model3.add(Dense(500))
model3.add(Dropout(0.25))
model3.add(Dense(num_classes))
model3.add(Activation('softmax'))
model3.summary()

#opt = keras.optimizers.Adam(lr=0.0001)
model3.compile(loss='categorical_crossentropy',
              optimizer="Adamax",
              metrics=['accuracy'])

m3=model3.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=150,
              validation_data=(x_test, y_test),
              shuffle=True)

plt.figure(figsize=(10, 6))
sb.set_style("whitegrid")

# Check the keys in the history object
print(m3.history.keys())

# Plot accuracy using the correct key
plt.plot(m3.history['accuracy'], color="#E74C3C", marker='o')
plt.plot(m3.history['val_accuracy'], color='#641E16', marker='h')

plt.title('Accuracy comparison between Validation and Train Data set', fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

plt.figure(figsize=(10,6))
sb.set_style("whitegrid")
plt.plot(m3.history['loss'],color="#E74C3C",marker='o')
plt.plot(m3.history['val_loss'],color='#641E16',marker='h')
plt.title('Loss comparison between Validation and Train Data set',fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

"""Case 4: ReLu,softmax,epochs-50,adamax"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adamax

# Assuming x_train.shape[1:] is (width, height, channels) and num_classes is defined

model4 = Sequential()

model4.add(Conv2D(32, (3, 3), kernel_initializer='uniform', input_shape=x_train.shape[1:]))
model4.add(Activation('relu'))
model4.add(Conv2D(32, (3, 3), kernel_initializer='uniform'))
model4.add(Activation('relu'))
model4.add(Conv2D(32, (3, 3), kernel_initializer='uniform'))
model4.add(Activation('relu'))

model4.add(MaxPooling2D((2, 2)))
model4.add(Dropout(0.25))

model4.add(Conv2D(48, (3, 3), kernel_initializer='uniform'))
model4.add(Activation('relu'))
model4.add(Conv2D(48, (3, 3), kernel_initializer='uniform'))
model4.add(Activation('relu'))
model4.add(Conv2D(48, (3, 3), kernel_initializer='uniform'))
model4.add(Activation('relu'))

model4.add(MaxPooling2D((2, 2)))
model4.add(Dropout(0.25))

model4.add(Conv2D(48, (3, 3), kernel_initializer='uniform'))
model4.add(Activation('relu'))

model4.add(GlobalMaxPooling2D())
model4.add(Dense(500))
model4.add(Dropout(0.25))
model4.add(Dense(num_classes))
model4.add(Activation('softmax'))

# Summary
model4.summary()

# Compile the model
model4.compile(optimizer=Adamax(),  # Using Adamax optimizer
               loss='categorical_crossentropy',
               metrics=['accuracy'])

#opt = keras.optimizers.Adam(lr=0.0001)
model4.compile(loss='categorical_crossentropy',
              optimizer="Adamax",
              metrics=['accuracy'])

m4=model4.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=50,
              validation_data=(x_test, y_test),
              shuffle=True)

plt.figure(figsize=(10, 6))
sb.set_style("whitegrid")

# Check the keys in the history object
print(m4.history.keys())

# Plot accuracy using the correct key
plt.plot(m4.history['accuracy'], color="#E74C3C", marker='o')
plt.plot(m4.history['val_accuracy'], color='#641E16', marker='h')

plt.title('Accuracy comparison between Validation and Train Data set', fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

plt.figure(figsize=(10,6))
sb.set_style("whitegrid")
plt.plot(m4.history['loss'],color="#E74C3C",marker='o')
plt.plot(m4.history['val_loss'],color='#641E16',marker='h')
plt.title('Loss comparison between Validation and Train Data set',fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

"""case 5: ReLu,softmax,epochs-50,adamax"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adamax

# Assuming x_train.shape[1:] is (width, height, channels) and num_classes is defined

model5 = Sequential()

model5.add(Conv2D(32, (3, 3), kernel_initializer='glorot_uniform', input_shape=x_train.shape[1:]))
model5.add(Activation('relu'))
model5.add(Conv2D(32, (3, 3), kernel_initializer='glorot_uniform'))
model5.add(Activation('relu'))
model5.add(Conv2D(32, (3, 3), kernel_initializer='glorot_uniform'))
model5.add(Activation('relu'))

model5.add(MaxPooling2D((2, 2)))
model5.add(Dropout(0.25))

model5.add(Conv2D(48, (3, 3), kernel_initializer='glorot_uniform'))
model5.add(Activation('relu'))
model5.add(Conv2D(48, (3, 3), kernel_initializer='glorot_uniform'))
model5.add(Activation('relu'))
model5.add(Conv2D(48, (3, 3), kernel_initializer='glorot_uniform'))
model5.add(Activation('relu'))

model5.add(MaxPooling2D((2, 2)))
model5.add(Dropout(0.25))



model5.add(GlobalMaxPooling2D())
model5.add(Dense(500))
model5.add(Dropout(0.25))
model5.add(Dense(num_classes))
model5.add(Activation('softmax'))

# Summary
model5.summary()

# Compile the model
model5.compile(optimizer=Adamax(),  # Using Adamax optimizer
               loss='categorical_crossentropy',
               metrics=['accuracy'])

#opt = keras.optimizers.Adam(lr=0.001)
model5.compile(loss='categorical_crossentropy',
              optimizer="Adamax",
              metrics=['accuracy'])

m5=model5.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=50,
              validation_data=(x_test, y_test),
              shuffle=True)

import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(10, 6))
sb.set_style("whitegrid")
plt.plot(m5.history['accuracy'], color="#E74C3C", marker='o', label='Train')
plt.plot(m5.history['val_accuracy'], color='#641E16', marker='h', label='Validation')
plt.title('Accuracy comparison between Validation and Train Data set', fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(10,6))
sb.set_style("whitegrid")
plt.plot(m5.history['loss'],color="#E74C3C",marker='o')
plt.plot(m5.history['val_loss'],color='#641E16',marker='h')
plt.title('Loss comparison between Validation and Train Data set',fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()
