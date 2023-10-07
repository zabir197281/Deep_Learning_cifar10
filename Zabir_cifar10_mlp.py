import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.regularizers import l2
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam

# Function to read data from pickel file

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Reading the training data from pickle file

data_batch_1=unpickle("data_batch_1")
data_batch_2=unpickle("data_batch_2")
data_batch_3=unpickle("data_batch_3")
data_batch_4=unpickle("data_batch_4")
data_batch_5=unpickle("data_batch_5")

# Concating all the training data

data_x=list(data_batch_1["data"])+list(data_batch_2["data"])+list(data_batch_3["data"])+list(data_batch_4["data"])+list(data_batch_5["data"])
lavel_y=list(data_batch_1["labels"])+list(data_batch_2["labels"])+list(data_batch_3["labels"])+list(data_batch_4["labels"])+list(data_batch_5["labels"])

data_x=np.array(data_x)
lavel_y=np.array(lavel_y)

# Normalizing the training image vector array to improve the accuricy

min_vals = np.min(data_x, axis=1, keepdims=True)
max_vals = np.max(data_x, axis=1, keepdims=True)
normalized_data_x = (data_x - min_vals) / (max_vals - min_vals)

# Reading the test  data from pickle file

test_data=unpickle("test_batch")
x_test=np.array(test_data["data"])
y_test=np.array(test_data["labels"])

# Normalizing the testing image vector array to compute the accuricy correctly

test_min_vals = np.min(x_test, axis=1, keepdims=True)
test_max_vals = np.max(x_test, axis=1, keepdims=True)
normalized_x_test = (x_test - test_min_vals) / (test_max_vals - test_min_vals)

# Building the  the model

model=Sequential()
""" 
In the first layer I first used five neurons and the activation function Sigmoid
but The sigmoid function squashes its input, which can be any real number, into a
range between 0 and 1. This makes it suitable for problems where the output needs
to represent a probability or likelihood so here the ReLU activation function was 
used where it  outputs the input value if it is positive, and if it's negative,
it outputs zero. I aslo inceased and decrised  the number of nuroens and layers 
to see for which parameters the accuriey is better.
"""
#model.add(Dense(5, input_dim=3072,activation='sigmoid')) 
model.add(Dense(512, input_dim=3072,activation='relu')) 
"""
I used Batch Normalization to improve the training stability, convergence speed, 
and overall performance of neural networks
"""

model.add(BatchNormalization())
#model.add(Dense(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
"""
I used Dropout to reduce overfitting and 
improve the generalization of the model
"""
model.add(Dropout(0.5))
#model.add(Dense(32, activation='relu'))
model.add(Dense(128, activation='softmax'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

"""
In the last layer I used the accivation function softmax because this function is used 
get outputs for catagorical data but in the question our professor instructed us to use 
sigmoid function in the last layer. So I increased another layer and in the last layer 
I used the sigmoid activation function but the accuricy was almost similar.
"""
# model.add(Dense(10, activation='softmax'))
model.add(Dense(10, activation='sigmoid'))


"""
First I used the learning rate 0.05 to compile the model but the accuricy was not so good.
Than I played with the parametter of learning rate and according to my trials with the learning 
rate 0.001 I got the best accuracy . 
"""


model.compile(optimizer=Adam(learning_rate= 0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
I also played with the parameters of epochs and batch_size
first I used the epochs 10 and batch size 64 after many trail and 
error I found that for epoc 30 and batch size 100 we get quite good 
results.
"""


hist= model.fit(normalized_data_x, lavel_y, epochs=30, batch_size=100, validation_split=0.2)

print()
print()
print()
print()

# Calculation test and train accuracy 

train_loss, train_acc = model.evaluate(normalized_data_x, lavel_y)
print(f'Train accuracy: {train_acc}')

test_loss, test_acc = model.evaluate(normalized_x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Plothing loss and accuricy figure 

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

##################################

plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()





