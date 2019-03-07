'''
Problem:
    There is a dataset called MNIST Fasion. 
    There are images and corresponding labels. 
    labels are in integer from 0 to 9. 
    make a prediction by taking an image means
    tell us whether it is a boot or trouser 
    and tell the accuracy of the prediction
    plot image, labels, accuracy
    
'''
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images.shape # (60000, 28, 28) 

len(train_labels) # 60,0000 train labels

train_labels # train labels are in between 0 to 9


test_images.shape # 10,000 test images, 28 by 28 pixel

len(test_labels) # 10,000 test labels

test_labels # array of 0 to 9


plt.figure()
plt.imshow(train_images[1])
plt.grid(False)
plt.colorbar()
plt.show()


# We scale these values to a range of 0 to 1 before feeding to the neural 
# network model. For this, we divide the values by 255. It's important that the 
# training set and the testing set are preprocessed in the same way:

train_images = train_images / 255.0
test_images = test_images / 255.0


# Display the first 25 images from the training set and 
# display the class name below each image. 
# Verify that the data is in the correct format and 
# we're ready to build and train the network.

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()
    
    
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
    
'''    
Compile the model

Before the model is ready for training, it needs a few more settings. These are 
added during the model's compile step:

    Loss function —This measures how accurate the model is during training. 
    We want to minimize this function to "steer" the model in the right direction.
    
    Optimizer —This is how the model is updated based on the data it sees and 
    its loss function.
    
    Metrics —Used to monitor the training and testing steps. The following example 
    uses accuracy, the fraction of the images that are correctly classified.
'''
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    


'''
Train the model

Training the neural network model requires the following steps:

    Feed the training data to the model—in this example, the train_images and 
    train_labels arrays.
    The model learns to associate images and labels.
    We ask the model to make predictions about a test set—in this example, 
    the test_images array. We verify that the predictions match the labels from 
    the test_labels array.

To start training, call the model.fit method—the model is "fit" to the training data:
'''    
model.fit(train_images, train_labels, epochs=5)

# As the model trains, the loss and accuracy metrics are displayed. 
# This model reaches an accuracy of about 0.88 (or 88%) on the training data.    

'''
Evaluate accuracy

Next, compare how the model performs on the test dataset:
'''
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

'''
It turns out, the accuracy on the test dataset is a little less than the accuracy 
on the training dataset. This gap between training accuracy and test accuracy 
is an example of overfitting. Overfitting is when a machine learning model 
performs worse on new data than on their training data.     
'''

'''
Make predictions

With the model trained, we can use it to make predictions about some images.
'''
predictions = model.predict(test_images)

'''
Here, the model has predicted the label for each image in the testing set. 
Let's take a look at the first prediction:
'''
predictions[1]
predictions[0]

'''
A prediction is an array of 10 numbers. These describe the "confidence" of the 
model that the image corresponds to each of the 10 different articles of clothing. 
We can see which label has the highest confidence value:
'''

np.argmax(predictions[0])
np.argmax(predictions[1])

'''
So the model is most confident that this image is an ankle boot, or class_names[9]. 
And we can check the test label to see this is correct:
'''
test_labels[0] # 9 = Ankle Boot
test_labels[1]  # 2 = Trousers


# We can graph this to look at the full set of 10 channels    
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 1
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
  

i = 2
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()


'''
Finally, use the trained model to make a prediction about a single image.
'''
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape) # (1, 28, 28)


# Now predict a single image
predictions_single = model.predict(img)

print(predictions_single) 


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# model.predict returns a list of lists, one for each image in the batch of data. 
# Grab the predictions for our (only) image in the batch:

np.argmax(predictions_single[0]) # 9

