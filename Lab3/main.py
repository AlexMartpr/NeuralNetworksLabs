import json

import numpy as np
import cv2

import tools.Utils as utils
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tools.NeuralNetwork.nn import NN

TRAIN = True
MNIST = False
TEST = False
MY_TEST = False

TASK1 = True
TASK2 = True
TASK3 = True

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

    if TRAIN:
        sr = {}
        digits = load_digits()
        if TASK1:
        #Task1:
            #20%
            sr['task1'] = {'p': [], 'res': [], 'losses': [], 'acc': [], 'epochs': np.arange(1, epochs + 1)}
            perc = (0.2, 0.4, 0.6, 0.8, 0.95)
            for p in perc:
                nn = NN(N=64, OUT=10, epochs=epochs)
                nn.generate_weights()
                res = train_test_split(digits.data, digits.target, train_size=p, test_size=1-p, random_state=1)
                train_data, test_data, train_labels, test_labels = res

                inp_tr = utils.Preprocess.normalize_whole_input(train_data, MNIST=MNIST)
                inp_tr = np.array(inp_tr)
                output_tr = utils.Preprocess.get_normalize_output_for_whole_ds(train_labels)
                output_tr = np.array(output_tr)
                
                rt_1 = nn.train(input=inp_tr, output=output_tr)
                sr['task1']['p'].append(p * 100)
                sr['task1']['losses'].append(rt_1[0])
                sr['task1']['acc'].append(rt_1[2])
        
            for i in range(len(perc)):
                plt.plot(sr['task1']['epochs'], sr['task1']['losses'][i], label=f'{perc[i] * 100}%')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plt.legend()
            plt.savefig('task1-losses.png')
            plt.close()

            for i in range(len(perc)):
                plt.plot(sr['task1']['epochs'], sr['task1']['acc'][i], label=f'{perc[i] * 100}%')

            plt.xlabel('Epochs')
            plt.ylabel('Acc')

            plt.legend()
            plt.savefig('task1-acc.png')
            plt.close()


        if TASK2:
            res = train_test_split(digits.data, digits.target, train_size=0.8, test_size=0.2, random_state=1)
            train_data, test_data, train_labels, test_labels = res

            inp_tr = utils.Preprocess.normalize_whole_input(train_data, MNIST=MNIST)
            inp_tr = np.array(inp_tr)
            output_tr = utils.Preprocess.get_normalize_output_for_whole_ds(train_labels)
            output_tr = np.array(output_tr)
            #Task2
            sr['task2'] = {'p': [], 'res': [], 'losses': [], 'acc': [], 'epochs': np.arange(1, epochs + 1)}
            l = ([], [16], [16, 8], [8, 8, 8])
            for i in l:
                nn = NN(N=64, OUT=10, layers=i, epochs=epochs)
                nn.generate_weights()
                
                rt_1 = nn.train(input=inp_tr, output=output_tr)
                sr['task2']['losses'].append(rt_1[0])
                sr['task2']['acc'].append(rt_1[2])

        
            for i in range(len(l)):
                plt.plot(sr['task2']['epochs'], sr['task2']['losses'][i], label=f'{" ".join(str(x) for x in l[i])}')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plt.legend()
            plt.savefig('task2-losses.png')
            plt.close()

            for i in range(len(l)):
                plt.plot(sr['task2']['epochs'], sr['task2']['acc'][i], label=f'{" ".join(str(x) for x in l[i])}')

            plt.xlabel('Epochs')
            plt.ylabel('Acc')

            plt.legend()
            plt.savefig('task2-acc.png')
            plt.close()

        if TASK3:
            res = train_test_split(digits.data, digits.target, train_size=0.8, test_size=0.2, random_state=1)
            train_data, test_data, train_labels, test_labels = res

            inp_tr = utils.Preprocess.normalize_whole_input(train_data, MNIST=MNIST)
            inp_tr = np.array(inp_tr)
            output_tr = utils.Preprocess.get_normalize_output_for_whole_ds(train_labels)
            output_tr = np.array(output_tr)

            R = (None,1,  2, 3)
            sr['task3'] = {'p': [], 'res': [], 'losses': [], 'acc': [], 'epochs': np.arange(1, epochs + 1)}
            for r in R:
                nn = NN(N=64, OUT=10, layers=[16], epochs=epochs)
                nn.generate_weights()
                
                rt_1 = nn.train(input=inp_tr, output=output_tr, regular=r)
                sr['task3']['losses'].append(rt_1[0])
                sr['task3']['acc'].append(rt_1[2])

            for i in range(len(R)):
                label = 'no'
                if i == 1:
                    label = 'L1'
                elif i == 2:
                    label = 'L2'
                elif i == 3:
                    label = 'L3'
                plt.plot(sr['task3']['epochs'], sr['task3']['losses'][i], label=label)

            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plt.legend()
            plt.savefig('task3-losses.png')
            plt.close()

            for i in range(len(R)):
                label = 'no'
                if i == 1:
                    label = 'L1'
                elif i == 2:
                    label = 'L2'
                elif i == 3:
                    label = 'Dropout'
                plt.plot(sr['task3']['epochs'], sr['task3']['acc'][i], label=label)

            plt.xlabel('Epochs')
            plt.ylabel('Acc')

            plt.legend()
            plt.savefig('task3-acc.png')
            plt.close()

if __name__ == '__main__':
    main()