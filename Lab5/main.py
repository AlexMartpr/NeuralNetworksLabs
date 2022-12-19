import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np

from keras import datasets, layers, models

epochs = 40

LENET = False
ALEXNET = False
SIMPLE = False

def LeNet():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    #Greyscale необходим, так как на вход в LeNet поступает сэмпл с одним цветовым каналом
    #В cifar10 их 3
    def rgb2gray(rgb):
        """Convert from color image (RGB) to grayscale.
        Source: opencv.org
        grayscale = 0.299*red + 0.587*green + 0.114*blue
        Argument:
            rgb (tensor): rgb image
        Return:
            (tensor): grayscale image
        """
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    img_rows = train_images.shape[1]
    img_cols = train_images.shape[2]

    img_rows_test = test_images.shape[1]
    img_cols_test = test_images.shape[2]

    train_images = rgb2gray(train_images)
    test_images = rgb2gray(test_images)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_images_grey = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images_grey = test_images.reshape(test_images.shape[0], img_rows_test, img_cols_test, 1)

    x_val = test_images_grey 
    y_val = test_labels 
    x_train = train_images_grey 
    y_train = train_labels

    model = models.Sequential()
    model.add(layers.Conv2D(6, 5, activation='relu', input_shape=(32, 32, 1), padding='same'))
    model.add(layers.AveragePooling2D(strides=2))
    model.add(layers.Conv2D(16, 5, activation='relu'))
    model.add(layers.AveragePooling2D(strides=2))    
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

    history = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        batch_size=64,
        validation_data=(x_val, y_val),
        validation_freq=1, 
        verbose=1)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(top=1)
    plt.legend(loc='lower right')
    plt.savefig('LeNet2.png')
    plt.close()


def AlexNet():

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # def process_images(image, label):
    #     # Normalize images to have a mean of 0 and standard deviation of 1
    #     image = tf.image.per_image_standardization(image)
    #     # Resize images from 32x32 to 277x277
    #     image = tf.image.resize(image, (227,227))
    #     return image, label

    x_val = test_images 
    y_val = test_labels 
    x_train = train_images 
    y_train = train_labels

    # train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()    
    # val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()
    
    # train_ds = (train_ds
    #               .map(process_images)
    #             #   .shuffle(buffer_size=train_ds_size)
    #               .batch(batch_size=32, drop_remainder=True))

    # val_ds = (val_ds
    #               .map(process_images)
    #             #   .shuffle(buffer_size=val_ds_size)
    #               .batch(batch_size=32, drop_remainder=True))

    #В оригинальной AlexNet сеть содержит намного больше параметров и input_shape=(227, 227, 3)
    #Если у вас мощная машинка, то можете сделать оригинальную сеть
    model = models.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), strides=(4,4), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Conv2D(filters=60, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Conv2D(filters=60, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.Conv2D(filters=30, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.Conv2D(filters=20, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=64,
          validation_data=(x_val, y_val),
          validation_freq=1,
          verbose=1)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(top=1)
    plt.legend(loc='lower right')
    plt.savefig('AlexNet2.png')
    plt.close()


def simple_model():

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))        
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()
        
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(
        train_images, train_labels, 
        epochs=epochs, 
        batch_size=64, 
        validation_data=(test_images, test_labels),
        validation_freq=1, 
        verbose=1)
        
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(top=1)
    plt.legend(loc='lower right')
    plt.savefig('simple1.png')
    plt.close()

def main():

    if SIMPLE:
        simple_model()

    if LENET:
        LeNet()

    if ALEXNET:
        AlexNet()

if __name__ == '__main__':
    main()
