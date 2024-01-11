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

def load_mnist_data():
    mndata = MNIST('C:\\fara\\MNIST_Project\\MNIST_data')
    images, labels = mndata.load_training()
    return images, labels

def show_images(images, labels):
    images_np = np.array(images)
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images_np[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.title("Label: %s" % labels[i])
        plt.axis('off')
    plt.show()

images, labels = load_mnist_data()
show_images(images, labels)

images_np = np.array(images) / 255.0
labels_np = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images_np, labels_np, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print(f'Training set: {X_train.shape[0]} examples')
print(f'Validation set: {X_val.shape[0]} examples')
print(f'Test set: {X_test.shape[0]} examples')

(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
X_train_cifar = X_train_cifar.astype('float32') / 255.0
X_test_cifar = X_test_cifar.astype('float32') / 255.0
y_train_cifar = to_categorical(y_train_cifar, 10)
y_test_cifar = to_categorical(y_test_cifar, 10)

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train_cifar[i])
plt.show()

# Разделение данных на обучающую и валидационную выборки
X_train_cifar, X_val_cifar, y_train_cifar, y_val_cifar = train_test_split(
    X_train_cifar, y_train_cifar, test_size=0.2, random_state=42
)

print(f'CIFAR-10 Training set: {X_train_cifar.shape[0]} examples')
print(f'CIFAR-10 Validation set: {X_val_cifar.shape[0]} examples')
print(f'CIFAR-10 Test set: {X_test_cifar.shape[0]} examples')

# Проверяем, что метки - это целые числа от 0 до 9
assert np.all((y_train_cifar >= 0) & (y_train_cifar < 10))
assert np.all((y_val_cifar >= 0) & (y_val_cifar < 10))

train = loadmat('C:/fara/MNIST_Project/SVHN_data/train_32x32.mat')
test = loadmat('C:/fara/MNIST_Project/SVHN_data/test_32x32.mat')
X_train_svhn = train['X']
y_train_svhn = train['y']
X_test_svhn = test['X']
y_test_svhn = test['y']
X_train_svhn = np.transpose(X_train_svhn, (3, 0, 1, 2)).astype('float32') / 255.0
X_test_svhn = np.transpose(X_test_svhn, (3, 0, 1, 2)).astype('float32') / 255.0
y_train_svhn[y_train_svhn == 10] = 0
y_test_svhn[y_test_svhn == 10] = 0

# Преобразование меток в one-hot encoding
y_train_svhn = y_train_svhn.flatten()  # Убедитесь, что метки одномерны
y_test_svhn = y_test_svhn.flatten()
y_train_svhn_one_hot = to_categorical(y_train_svhn, 10)
y_test_svhn_one_hot = to_categorical(y_test_svhn, 10)

y_train_svhn = to_categorical(y_train_svhn, 10)
y_test_svhn = to_categorical(y_test_svhn, 10)
y_train_svhn = to_categorical(y_train_svhn, 10)
y_test_svhn = to_categorical(y_test_svhn, 10)

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train_svhn[i])
plt.show()

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train_svhn, X_val_svhn, y_train_svhn_one_hot, y_val_svhn_one_hot = train_test_split(X_train_svhn, y_train_svhn_one_hot, test_size=0.2, random_state=42)
X_val_svhn, X_test_svhn, y_val_svhn_one_hot, y_test_svhn_one_hot = train_test_split(X_val_svhn, y_val_svhn_one_hot, test_size=0.5, random_state=42)

print(f'SVHN Training set: {X_train_svhn.shape[0]} examples')
print(f'SVHN Validation set: {X_val_svhn.shape[0]} examples')
print(f'SVHN Test set: {X_test_svhn.shape[0]} examples')

#тут мы уже подготовили и загрузили разнообразные наборы данных
#дальше реализация и оптимизация базовой архитектуры CNN

# Создание модели MNIST
model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))  # Устанавливаем размер входного слоя
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Используйте эту функцию потерь для one-hot меток
              metrics=['accuracy'])
model.summary()

#ниже будет 6 шаг 3-го пункта

# Загрузка данных из Numpy файлов
X_train_mnist = np.load('X_train.npy')
y_train_mnist = np.load('y_train.npy')
X_val_mnist = np.load('X_val.npy')
y_val_mnist = np.load('y_val.npy')

# Преобразование размерности MNIST данных
X_train_mnist_reshaped = X_train.reshape(-1, 28, 28, 1)
X_val_mnist_reshaped = X_val.reshape(-1, 28, 28, 1)

def resize_mnist_images(images):
    images_tensor = tf.convert_to_tensor(images)
    images_resized = tf.image.resize(images_tensor, [32, 32])
    images_resized = tf.repeat(images_resized, 3, axis=-1)
    return images_resized.numpy()

X_train_mnist_resized = resize_mnist_images(X_train_mnist_reshaped)
X_val_mnist_resized = resize_mnist_images(X_val_mnist_reshaped)

# Изменение размера и добавление каналов к изображениям MNIST
X_train_mnist_resized = tf.image.resize(X_train_mnist_reshaped, [32, 32])
X_train_mnist_resized = tf.repeat(X_train_mnist_resized, 3, axis=-1)
X_val_mnist_resized = tf.image.resize(X_val_mnist_reshaped, [32, 32])
X_val_mnist_resized = tf.repeat(X_val_mnist_resized, 3, axis=-1)

y_train_one_hot = to_categorical(y_train_mnist, num_classes=10)
y_val_one_hot = to_categorical(y_val_mnist, num_classes=10)

# Обучение модели для MNIST
history = model.fit(
    X_train_mnist_resized, 
    y_train_one_hot, 
    epochs=10, 
    batch_size=64, 
    validation_data=(X_val_mnist_resized, y_val_one_hot)
)

# Визуализация точности обучения и валидации для MNIST
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('MNIST Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Визуализация потерь обучения и валидации для MNIST
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('MNIST Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Сохранение модели
model.save('mnist_model.h5')

# Создание модели MNIST с Dropout
model_mnist_with_dropout = models.Sequential()
model_mnist_with_dropout.add(layers.Input(shape=(28, 28, 1)))  # Размер входных данных для MNIST
model_mnist_with_dropout.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_mnist_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_mnist_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_mnist_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_mnist_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_mnist_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_mnist_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_mnist_with_dropout.add(layers.Flatten())
model_mnist_with_dropout.add(layers.Dense(64, activation='relu'))
model_mnist_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_mnist_with_dropout.add(layers.Dense(10, activation='softmax'))
model_mnist_with_dropout.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели MNIST с Dropout
history_with_dropout = model_mnist_with_dropout.fit(
    X_train_mnist_reshaped, 
    y_train_mnist, 
    epochs=10, 
    batch_size=64, 
    validation_data=(X_val_mnist_reshaped, y_val_mnist)
)

# Визуализация точности обучения и валидации для MNIST с Dropout
plt.plot(history_with_dropout.history['accuracy'], label='Train Accuracy (With Dropout)')
plt.plot(history_with_dropout.history['val_accuracy'], label='Validation Accuracy (With Dropout)')
plt.title('MNIST Training and Validation Accuracy with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Визуализация потерь обучения и валидации для MNIST с Dropout
plt.plot(history_with_dropout.history['loss'], label='Train Loss (With Dropout)')
plt.plot(history_with_dropout.history['val_loss'], label='Validation Loss (With Dropout)')
plt.title('MNIST Training and Validation Loss with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Вывод структуры модели
model_mnist_with_dropout.summary()


# Создание модели для CIFAR-10
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
datagen = ImageDataGenerator(rotation_range=20, 
                             width_shift_range=0.2, 
                             height_shift_range=0.2, 
                             horizontal_flip=True)
datagen.fit(X_train_cifar)
history_cifar = model_cifar.fit(
    datagen.flow(X_train_cifar, y_train_cifar, batch_size=64),
    epochs=20, validation_data=(X_val_cifar, y_val_cifar))

# Визуализация точности обучения и валидации
plt.plot(history_cifar.history['accuracy'], label='Train Accuracy')
plt.plot(history_cifar.history['val_accuracy'], label='Validation Accuracy')
plt.title('CIFAR-10 Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Визуализация потерь обучения и валидации
plt.plot(history_cifar.history['loss'], label='Train Loss')
plt.plot(history_cifar.history['val_loss'], label='Validation Loss')
plt.title('CIFAR-10 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
model_cifar.save('cifar10_model.keras')

# Создание модели CIFAR-10 с Dropout
model_cifar_with_dropout = models.Sequential()
model_cifar_with_dropout.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_cifar_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_cifar_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_cifar_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cifar_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_cifar_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_cifar_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cifar_with_dropout.add(layers.Flatten())
model_cifar_with_dropout.add(layers.Dense(64, activation='relu'))
model_cifar_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_cifar_with_dropout.add(layers.Dense(10, activation='softmax'))
model_cifar_with_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Подготовка аугментации данных
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train_cifar)

# Обучение модели CIFAR-10 с Dropout
history_cifar_with_dropout = model_cifar_with_dropout.fit(
    datagen.flow(X_train_cifar, y_train_cifar, batch_size=64),
    epochs=20, validation_data=(X_test_cifar, y_test_cifar)
)

# Визуализация точности обучения и валидации
plt.plot(history_cifar_with_dropout.history['accuracy'], label='Train Accuracy (With Dropout)')
plt.plot(history_cifar_with_dropout.history['val_accuracy'], label='Validation Accuracy (With Dropout)')
plt.title('CIFAR-10 Training and Validation Accuracy with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Визуализация потерь обучения и валидации
plt.plot(history_cifar_with_dropout.history['loss'], label='Train Loss (With Dropout)')
plt.plot(history_cifar_with_dropout.history['val_loss'], label='Validation Loss (With Dropout)')
plt.title('CIFAR-10 Training and Validation Loss with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Вывод структуры модели
model_cifar_with_dropout.summary()

#обучение для SVHN
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
datagen_svhn = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen_svhn.fit(X_train_svhn)
history_svhn = model_svhn.fit(
    datagen_svhn.flow(X_train_svhn, y_train_svhn_one_hot, batch_size=64),
    epochs=20,validation_data=(X_val_svhn, y_val_svhn_one_hot))
model_svhn.save('svhn_model.h5')

plt.plot(history_svhn.history['accuracy'], label='Train Accuracy')
plt.plot(history_svhn.history['val_accuracy'], label='Validation Accuracy')
plt.title('SVHN Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_svhn.history['loss'], label='Train Loss')
plt.plot(history_svhn.history['val_loss'], label='Validation Loss')
plt.title('SVHN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Транспонирование данных SVHN
X_train_svhn = np.transpose(train['X'], (3, 0, 1, 2)).astype('float32') / 255.0
X_test_svhn = np.transpose(test['X'], (3, 0, 1, 2)).astype('float32') / 255.0

# Преобразование меток в one-hot encoding
y_train_svhn = to_categorical(train['y'].flatten() % 10, 10)
y_test_svhn = to_categorical(test['y'].flatten() % 10, 10)

# Создание модели SVHN с Dropout
model_svhn_with_dropout = models.Sequential()
model_svhn_with_dropout.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_svhn_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_svhn_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_svhn_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_svhn_with_dropout.add(layers.MaxPooling2D((2, 2)))
model_svhn_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_svhn_with_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_svhn_with_dropout.add(layers.Flatten())
model_svhn_with_dropout.add(layers.Dense(64, activation='relu'))
model_svhn_with_dropout.add(layers.Dropout(0.5))  # Добавление слоя Dropout
model_svhn_with_dropout.add(layers.Dense(10, activation='softmax'))
model_svhn_with_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Подготовка аугментации данных
datagen_svhn = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen_svhn.fit(X_train_svhn)

# Обучение модели SVHN с Dropout
history_svhn_with_dropout = model_svhn_with_dropout.fit(
    datagen_svhn.flow(X_train_svhn, y_train_svhn, batch_size=64),
    epochs=20, validation_data=(X_test_svhn, y_test_svhn)
)

# Визуализация точности обучения и валидации
plt.plot(history_svhn_with_dropout.history['accuracy'], label='Train Accuracy (With Dropout)')
plt.plot(history_svhn_with_dropout.history['val_accuracy'], label='Validation Accuracy (With Dropout)')
plt.title('SVHN Training and Validation Accuracy with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Визуализация потерь обучения и валидации
plt.plot(history_svhn_with_dropout.history['loss'], label='Train Loss (With Dropout)')
plt.plot(history_svhn_with_dropout.history['val_loss'], label='Validation Loss (With Dropout)')
plt.title('SVHN Training and Validation Loss with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Вывод структуры модели
model_svhn_with_dropout.summary()
