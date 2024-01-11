# Importing necessary libraries for data manipulation, visualization, and deep learning.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mnist import MNIST
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from scipy.io import loadmat
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# Function to load MNIST dataset from a specified path.
def load_mnist_data():
    mndata = MNIST('C:\\fara\\MNIST_Project\\MNIST_data')
    images, labels = mndata.load_training()
    return images, labels

# Function to display the first 10 images and their labels from the dataset.
def show_images(images, labels):
    images_np = np.array(images)
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images_np[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.title("Label: %s" % labels[i])
        plt.axis('off')
    plt.show()

# Load MNIST data and display the images.
images, labels = load_mnist_data()
show_images(images, labels)

# Normalizing image data to have values between 0 and 1.
images_np = np.array(images) / 255.0
labels_np = np.array(labels)

# Splitting the dataset into training, validation, and test sets.
X_train, X_test, y_train, y_test = train_test_split(images_np, labels_np, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Saving the split data into numpy files for later use.
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Printing the size of each dataset to the console.
print(f'Training set: {X_train.shape[0]} examples')
print(f'Validation set: {X_val.shape[0]} examples')
print(f'Test set: {X_test.shape[0]} examples')

# Loading the CIFAR-10 dataset and normalizing the images.
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
X_train_cifar = X_train_cifar.astype('float32') / 255.0
X_test_cifar = X_test_cifar.astype('float32') / 255.0

# One-hot encoding the labels for the CIFAR-10 dataset.
y_train_cifar = to_categorical(y_train_cifar, 10)
y_test_cifar = to_categorical(y_test_cifar, 10)
# Displaying the first 9 images from the CIFAR-10 training dataset.
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train_cifar[i])
plt.show()

# Splitting the CIFAR-10 dataset into training and validation sets.
X_train_cifar, X_val_cifar, y_train_cifar, y_val_cifar = train_test_split(
    X_train_cifar, y_train_cifar, test_size=0.2, random_state=42
)

# Printing the size of the CIFAR-10 training and validation sets.
print(f'CIFAR-10 Training set: {X_train_cifar.shape[0]} examples')
print(f'CIFAR-10 Validation set: {X_val_cifar.shape[0]} examples')
print(f'CIFAR-10 Test set: {X_test_cifar.shape[0]} examples')

# Ensuring that the labels are integers between 0 and 9.
assert np.all((y_train_cifar >= 0) & (y_train_cifar < 10))
assert np.all((y_val_cifar >= 0) & (y_val_cifar < 10))

# Loading the SVHN (Street View House Numbers) dataset.
train = loadmat('C:/fara/MNIST_Project/SVHN_data/train_32x32.mat')
test = loadmat('C:/fara/MNIST_Project/SVHN_data/test_32x32.mat')
X_train_svhn = train['X']
y_train_svhn = train['y']
X_test_svhn = test['X']
y_test_svhn = test['y']

# Normalizing the SVHN images and adjusting their dimensions.
X_train_svhn = np.transpose(X_train_svhn, (3, 0, 1, 2)).astype('float32') / 255.0
X_test_svhn = np.transpose(X_test_svhn, (3, 0, 1, 2)).astype('float32') / 255.0

# Adjusting labels (10 is mapped to 0).
y_train_svhn[y_train_svhn == 10] = 0
y_test_svhn[y_test_svhn == 10] = 0

# Converting labels to one-hot encoding.
y_train_svhn = y_train_svhn.flatten()  # Ensuring labels are one-dimensional.
y_test_svhn = y_test_svhn.flatten()
y_train_svhn_one_hot = to_categorical(y_train_svhn, 10)
y_test_svhn_one_hot = to_categorical(y_test_svhn, 10)

# Displaying the first 9 images from the SVHN training dataset.
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train_svhn[i])
plt.show()

# Splitting the SVHN dataset into training, validation, and test sets.
X_train_svhn, X_val_svhn, y_train_svhn_one_hot, y_val_svhn_one_hot = train_test_split(X_train_svhn, y_train_svhn_one_hot, test_size=0.2, random_state=42)
X_val_svhn, X_test_svhn, y_val_svhn_one_hot, y_test_svhn_one_hot = train_test_split(X_val_svhn, y_val_svhn_one_hot, test_size=0.5, random_state=42)

# Printing the size of the SVHN training, validation, and test sets.
print(f'SVHN Training set: {X_train_svhn.shape[0]} examples')
print(f'SVHN Validation set: {X_val_svhn.shape[0]} examples')
print(f'SVHN Test set: {X_test_svhn.shape[0]} examples')

# Define a sequential model for image classification.
model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))  # Setting the input layer size.
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # First convolutional layer.
model.add(layers.MaxPooling2D((2, 2)))  # First max pooling layer.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer.
model.add(layers.MaxPooling2D((2, 2)))  # Second max pooling layer.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Third convolutional layer.
model.add(layers.Flatten())  # Flattening the output for the dense layer.
model.add(layers.Dense(64, activation='relu'))  # Dense layer with 64 neurons.
model.add(layers.Dense(10, activation='softmax'))  # Output layer with softmax activation for classification.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use this loss function for one-hot encoded labels.
              metrics=['accuracy'])
model.summary()  # Display the model architecture.

# Loading data from Numpy files.
X_train_mnist = np.load('X_train.npy')
y_train_mnist = np.load('y_train.npy')
X_val_mnist = np.load('X_val.npy')
y_val_mnist = np.load('y_val.npy')

# Reshaping the MNIST data to the required dimensions.
X_train_mnist_reshaped = X_train.reshape(-1, 28, 28, 1)
X_val_mnist_reshaped = X_val.reshape(-1, 28, 28, 1)

# Function to resize MNIST images to 32x32 and repeat them across three color channels.
def resize_mnist_images(images):
    images_tensor = tf.convert_to_tensor(images)
    images_resized = tf.image.resize(images_tensor, [32, 32])
    images_resized = tf.repeat(images_resized, 3, axis=-1)
    return images_resized.numpy()

X_train_mnist_resized = resize_mnist_images(X_train_mnist_reshaped)
X_val_mnist_resized = resize_mnist_images(X_val_mnist_reshaped)

# Converting labels to one-hot encoding.
y_train_one_hot = to_categorical(y_train_mnist, num_classes=10)
y_val_one_hot = to_categorical(y_val_mnist, num_classes=10)

# Training the model on the MNIST dataset.
history = model.fit(
    X_train_mnist_resized, 
    y_train_one_hot, 
    epochs=10, 
    batch_size=64, 
    validation_data=(X_val_mnist_resized, y_val_one_hot)
)

# Plotting training and validation accuracy for MNIST.
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('MNIST Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting training and validation loss for MNIST.
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('MNIST Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Saving the trained model.
model.save('mnist_model.h5')
# Creating a MNIST model with Dropout layers to reduce overfitting.
model_mnist_with_dropout = models.Sequential()
model_mnist_with_dropout.add(layers.Input(shape=(28, 28, 1)))  # Input size for MNIST data.
model_mnist_with_dropout.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_mnist_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_mnist_with_dropout.add(layers.Dropout(0.5))  # Adding a Dropout layer to prevent overfitting.
model_mnist_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_mnist_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_mnist_with_dropout.add(layers.Dropout(0.5))  # Another Dropout layer.
model_mnist_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_mnist_with_dropout.add(layers.Flatten())
model_mnist_with_dropout.add(layers.Dense(64, activation='relu'))
model_mnist_with_dropout.add(layers.Dropout(0.5))  # Final Dropout layer.
model_mnist_with_dropout.add(layers.Dense(10, activation='softmax'))
model_mnist_with_dropout.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the MNIST model with Dropout layers.
history_with_dropout = model_mnist_with_dropout.fit(
    X_train_mnist_reshaped, 
    y_train_mnist, 
    epochs=10, 
    batch_size=64, 
    validation_data=(X_val_mnist_reshaped, y_val_mnist)
)

# Visualizing training and validation accuracy with Dropout layers.
plt.plot(history_with_dropout.history['accuracy'], label='Train Accuracy (With Dropout)')
plt.plot(history_with_dropout.history['val_accuracy'], label='Validation Accuracy (With Dropout)')
plt.title('MNIST Training and Validation Accuracy with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualizing training and validation loss with Dropout layers.
plt.plot(history_with_dropout.history['loss'], label='Train Loss (With Dropout)')
plt.plot(history_with_dropout.history['val_loss'], label='Validation Loss (With Dropout)')
plt.title('MNIST Training and Validation Loss with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Displaying the structure of the MNIST model with Dropout.
model_mnist_with_dropout.summary()

# Creating a model for CIFAR-10 dataset.
model_cifar = models.Sequential()
model_cifar.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_cifar.add(layers.MaxPooling2D((2, 2)))
model_cifar.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cifar.add(layers.MaxPooling2D((2, 2)))
model_cifar.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cifar.add(layers.Flatten())
model_cifar.add(layers.Dense(64, activation='relu'))
model_cifar.add(layers.Dense(10, activation='softmax'))
model_cifar.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Using ImageDataGenerator for data augmentation to improve training.
datagen = ImageDataGenerator(rotation_range=20, 
                             width_shift_range=0.2, 
                             height_shift_range=0.2, 
                             horizontal_flip=True)
datagen.fit(X_train_cifar)

# Training the CIFAR-10 model with augmented data.
history_cifar = model_cifar.fit(
    datagen.flow(X_train_cifar, y_train_cifar, batch_size=64),
    epochs=20, validation_data=(X_val_cifar, y_val_cifar))

# Visualizing training and validation accuracy for CIFAR-10.
plt.plot(history_cifar.history['accuracy'], label='Train Accuracy')
plt.plot(history_cifar.history['val_accuracy'], label='Validation Accuracy')
plt.title('CIFAR-10 Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualizing training and validation loss for CIFAR-10.
plt.plot(history_cifar.history['loss'], label='Train Loss')
plt.plot(history_cifar.history['val_loss'], label='Validation Loss')
plt.title('CIFAR-10 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Saving the CIFAR-10 model.
model_cifar.save('cifar10_model.keras')
# Creating a CIFAR-10 model with Dropout layers.
model_cifar_with_dropout = models.Sequential()
model_cifar_with_dropout.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_cifar_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_cifar_with_dropout.add(layers.Dropout(0.5))  # Adding a Dropout layer to prevent overfitting.
model_cifar_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cifar_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_cifar_with_dropout.add(layers.Dropout(0.5))  # Another Dropout layer.
model_cifar_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cifar_with_dropout.add(layers.Flatten())
model_cifar_with_dropout.add(layers.Dense(64, activation='relu'))
model_cifar_with_dropout.add(layers.Dropout(0.5))  # Final Dropout layer.
model_cifar_with_dropout.add(layers.Dense(10, activation='softmax'))
model_cifar_with_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preparing data augmentation.
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train_cifar)

# Training the CIFAR-10 model with Dropout layers.
history_cifar_with_dropout = model_cifar_with_dropout.fit(
    datagen.flow(X_train_cifar, y_train_cifar, batch_size=64),
    epochs=20, validation_data=(X_test_cifar, y_test_cifar)
)

# Visualizing training and validation accuracy with Dropout layers.
plt.plot(history_cifar_with_dropout.history['accuracy'], label='Train Accuracy (With Dropout)')
plt.plot(history_cifar_with_dropout.history['val_accuracy'], label='Validation Accuracy (With Dropout)')
plt.title('CIFAR-10 Training and Validation Accuracy with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualizing training and validation loss with Dropout layers.
plt.plot(history_cifar_with_dropout.history['loss'], label='Train Loss (With Dropout)')
plt.plot(history_cifar_with_dropout.history['val_loss'], label='Validation Loss (With Dropout)')
plt.title('CIFAR-10 Training and Validation Loss with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Displaying the structure of the CIFAR-10 model with Dropout.
model_cifar_with_dropout.summary()

# Creating a model for the SVHN (Street View House Numbers) dataset.
model_svhn = models.Sequential()
model_svhn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_svhn.add(layers.MaxPooling2D((2, 2)))
model_svhn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_svhn.add(layers.MaxPooling2D((2, 2)))
model_svhn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_svhn.add(layers.Flatten())
model_svhn.add(layers.Dense(64, activation='relu'))
model_svhn.add(layers.Dense(10, activation='softmax'))
model_svhn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preparing data augmentation for SVHN.
datagen_svhn = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen_svhn.fit(X_train_svhn)

# Training the model for SVHN.
history_svhn = model_svhn.fit(
    datagen_svhn.flow(X_train_svhn, y_train_svhn_one_hot, batch_size=64),
    epochs=20,validation_data=(X_val_svhn, y_val_svhn_one_hot))

# Saving the SVHN model.
model_svhn.save('svhn_model.h5')

# Visualizing training and validation accuracy for SVHN.
plt.plot(history_svhn.history['accuracy'], label='Train Accuracy')
plt.plot(history_svhn.history['val_accuracy'], label='Validation Accuracy')
plt.title('SVHN Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualizing training and validation loss for SVHN.
plt.plot(history_svhn.history['loss'], label='Train Loss')
plt.plot(history_svhn.history['val_loss'], label='Validation Loss')
plt.title('SVHN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Transposing SVHN data to the correct format.
X_train_svhn = np.transpose(train['X'], (3, 0, 1, 2)).astype('float32') / 255.0
X_test_svhn = np.transpose(test['X'], (3, 0, 1, 2)).astype('float32') / 255.0

# Converting labels to one-hot encoding for SVHN dataset.
y_train_svhn = to_categorical(train['y'].flatten() % 10, 10)
y_test_svhn = to_categorical(test['y'].flatten() % 10, 10)

# Creating an SVHN model with Dropout layers to reduce overfitting.
model_svhn_with_dropout = models.Sequential()
model_svhn_with_dropout.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_svhn_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_svhn_with_dropout.add(layers.Dropout(0.5))  # Adding a Dropout layer.
model_svhn_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_svhn_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_svhn_with_dropout.add(layers.Dropout(0.5))  # Another Dropout layer.
model_svhn_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_svhn_with_dropout.add(layers.Flatten())
model_svhn_with_dropout.add(layers.Dense(64, activation='relu'))
model_svhn_with_dropout.add(layers.Dropout(0.5))  # Final Dropout layer.
model_svhn_with_dropout.add(layers.Dense(10, activation='softmax'))
model_svhn_with_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preparing data augmentation for the SVHN dataset.
datagen_svhn = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen_svhn.fit(X_train_svhn)

# Training the SVHN model with Dropout layers.
history_svhn_with_dropout = model_svhn_with_dropout.fit(
    datagen_svhn.flow(X_train_svhn, y_train_svhn, batch_size=64),
    epochs=20, validation_data=(X_test_svhn, y_test_svhn)
)

# Visualizing training and validation accuracy for the SVHN model with Dropout layers.
plt.plot(history_svhn_with_dropout.history['accuracy'], label='Train Accuracy (With Dropout)')
plt.plot(history_svhn_with_dropout.history['val_accuracy'], label='Validation Accuracy (With Dropout)')
plt.title('SVHN Training and Validation Accuracy with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualizing training and validation loss for the SVHN model with Dropout layers.
plt.plot(history_svhn_with_dropout.history['loss'], label='Train Loss (With Dropout)')
plt.plot(history_svhn_with_dropout.history['val_loss'], label='Validation Loss (With Dropout)')
plt.title('SVHN Training and Validation Loss with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Displaying the structure of the SVHN model with Dropout.
model_svhn_with_dropout.summary()