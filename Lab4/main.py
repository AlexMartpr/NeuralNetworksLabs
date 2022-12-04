import json

import numpy as np
import cv2

import tools.Utils as utils
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from rbf import RBF


TRAIN = True
MNIST = True
TEST = False
MY_TEST = False

if MNIST:
    from keras.datasets import mnist #mnist dataset 28x28 digits
else:
    from sklearn.datasets import load_digits #dataset 8x8 digits

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
            epochs = 100
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x = np.concatenate((x_train, x_test))
            y = np.concatenate((y_train, y_test))
            mnist_res = train_test_split(x, y, train_size=0.1, test_size=0.2, random_state=1)
            train_data, test_data, train_labels, test_labels = mnist_res
    else:
        file_weights = 'weights.json'
        if TRAIN:
            epochs = 1000
            digits = load_digits()
            res = train_test_split(digits.data, digits.target, train_size=0.8, test_size=0.2, random_state=1)
            train_data, test_data, train_labels, test_labels = res


    inp_tr = utils.Preprocess.normalize_whole_input(train_data, MNIST=MNIST)
    inp_tr = np.array(inp_tr)
    print(len(inp_tr))
    output_tr = utils.Preprocess.get_normalize_output_for_whole_ds(train_labels)
    output_tr = np.array(output_tr)
    # inp_te = utils.Prep.normalize_whole_input(test_data)
    # output_te = utils.Prep.get_normalize_output_for_whole_ds(test_labels)

    if TRAIN:
        ks = [1, 2, 8, 16, 64, 100, 120]
        # ks = [24, 48, 64, 128, 1000]
        accs = []
        for k in ks:
            nn = RBF(k=k)
            acc = nn.train(inp_tr, output_tr)
            print(acc)
            accs.append(acc)

        for (k, acc) in zip(ks, accs):
            print(f'K = {k} Acc = {acc}')

        plt.plot(ks, accs, 'o', color='blue')

        plt.xlabel('K')
        plt.ylabel('Acc')

        plt.legend()
        plt.savefig('pic.png')
        plt.close()
       


    # if TEST:
    #     idx = 0
    #     correct = 0
    #     for (test, res) in zip(test_data, test_labels):
    #         print(f'Test №{idx + 1}')
    #         output = [0 for _ in range(10)]
    #         output[res] = 1

    #         if MNIST:
    #             test = np.reshape(test, 784)
            
    #         test = nn.normalize_input(test)
    #         _, res_nn = nn.predict(test)

    #         if res_nn == res:
    #             correct += 1
    #         print(f'Actual = {res}, prediction = {res_nn}')
    #         idx += 1
        

    #     print(f'Tests N={len(test_data)}, corrects={correct}')

    # if MY_TEST:
    #     my_data, my_res = read_test_img()
    #     idx = 0
    #     for (test, res) in zip(my_data, my_res):
    #         print(f'My test №{idx + 1}')
    #         output = [0 for _ in range(10)]
    #         output[res] = 1

    #         test = nn.normalize_input(test)
    #         _, res_nn = nn.predict(test)

    #         idx+=1

    #         print(f'Actual = {res}, prediction = {res_nn}')

if __name__ == '__main__':
    main()