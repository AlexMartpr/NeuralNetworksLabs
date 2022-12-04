import json

import numpy as np
import cv2

from sklearn.datasets import load_digits #dataset 8x8 digits
from sklearn.model_selection import train_test_split
from keras.datasets import mnist #mnist dataset 28x28 digits

from NeuralNetwork.nn import NN

TRAIN = True
MNIST = True
TEST = False
MY_TEST = True

test_images = [
    '1.png', '11.png', 
    '2.png', '22.png', 
    '3.png', '33.png', 
    '4.png', '44.png', 
    '5.png', '55.png', 
    '6.png', '66.png',
    '7.png', '77.png',
    '8.png', '88.png',
    '9.png', '99.png']

def save_weights(nn, file_name):
    with open(file_name, 'w') as f:
        data = []
        for w in nn.w:
            if isinstance(w, np.ndarray):
                w = w.tolist()
            data.append(w)
        json.dump(data, f)

def read_weights(file_name):
    data = []
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except Exception as e:
        # return []
        print(e)
        exit(-1)

    if len(data) == 0:
        print('Empty weights')

    return data

def read_test_img():
    data, res = [], []

    for i in test_images:
        img = cv2.imread('test/' + i)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image1 = 255 - gray_image
        n = 64
        if MNIST:
            n = 784
        image2 = np.reshape(image1, n)
        data.append(image2)
        res.append(int(i[0]))

    return data, res

def main():
    train_data, test_data, train_labels, test_labels = [], [], [], []
    file_weights = ''
    epochs = 0
    if MNIST:
        file_weights = 'weights1.json'
        if TRAIN:
            epochs = 10
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x = np.concatenate((x_train, x_test))
            y = np.concatenate((y_train, y_test))
            mnist_res = train_test_split(x, y, train_size=0.6, test_size=0.2, random_state=1)
            train_data, test_data, train_labels, test_labels = mnist_res
    else:
        file_weights = 'weights.json'
        if TRAIN:
            epochs = 100
            digits = load_digits()
            res = train_test_split(digits.data, digits.target, train_size=0.8, test_size=0.2, random_state=1)
            train_data, test_data, train_labels, test_labels = res

    w = read_weights(file_weights)
    flag = len(w) == 0
    # w = []
    # flag = True
    nn = NN(784, 10)
    if flag:
        print('SHIT')
        nn.generate_weights()
    else:
        nn.w = w  

    if TRAIN:
        count, correct = 0, 0
        for i in range(epochs):
            for (test, res) in zip(train_data, train_labels):
                output = [0 for _ in range(10)]
                output[res] = 1
                
                if MNIST:
                    test = np.reshape(test, 784)

                test = nn.normalize_input(test)
                _, res_nn = nn.train(test, output)

                if res_nn == res:
                    correct += 1

                count += 1
            print(f'Epoch = {i}, corrects = {correct}, all = {count}')
            print(f'Accuracy = {correct / count * 100}')

            correct = 0
            count = 0

        save_weights(nn, file_weights)

    if TEST:
        idx = 0
        correct = 0
        for (test, res) in zip(test_data, test_labels):
            print(f'Test №{idx + 1}')
            output = [0 for _ in range(10)]
            output[res] = 1

            if MNIST:
                test = np.reshape(test, 784)
            
            test = nn.normalize_input(test)
            _, res_nn = nn.predict(test)

            if res_nn == res:
                correct += 1
            print(f'Actual = {res}, prediction = {res_nn}')
            idx += 1
        

        print(f'Tests N={len(test_data)}, corrects={correct}')

    if MY_TEST:
        my_data, my_res = read_test_img()
        idx = 0
        for (test, res) in zip(my_data, my_res):
            print(f'My test №{idx + 1}')
            output = [0 for _ in range(10)]
            output[res] = 1

            test = nn.normalize_input(test)
            _, res_nn = nn.predict(test)

            idx+=1

            print(f'Actual = {res}, prediction = {res_nn}')

if __name__ == '__main__':
    main()