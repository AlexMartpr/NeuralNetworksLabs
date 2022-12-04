from copy import deepcopy
import json

import numpy as np
import cv2

import matplotlib.pyplot as plt

import tools.Utils as utils
import tools.Optimizers as optims

from sklearn.model_selection import train_test_split

from tools.NeuralNetwork.nn import NN


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
    optims_res = {}
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
            mnist_res = train_test_split(x, y, train_size=0.3, test_size=0.2, random_state=1)
            train_data, test_data, train_labels, test_labels = mnist_res
    else:
        file_weights = 'weights.json'
        if TRAIN:
            epochs = 1000
            digits = load_digits()
            res = train_test_split(digits.data, digits.target, train_size=0.8, test_size=0.2, random_state=1)
            train_data, test_data, train_labels, test_labels = res

    nn = NN(784, 10, epochs=epochs)

    inp_tr = utils.Preprocess.normalize_whole_input(train_data, MNIST=MNIST)
    inp_tr = np.array(inp_tr)
    output_tr = utils.Preprocess.get_normalize_output_for_whole_ds(train_labels)
    output_tr = np.array(output_tr)
    # inp_te = utils.Prep.normalize_whole_input(test_data)
    # output_te = utils.Prep.get_normalize_output_for_whole_ds(test_labels)

    if TRAIN:
        
        # nn.optimizer = optims.Batch(nn)
        # rt = nn.optimizer.run(inp_tr, output_tr)
        # optims_res['batch'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}
        # exit(-1)
        
        # nn.optimizer = optims.MiniBatch(nn)
        # rt = nn.optimizer.run(inp_tr, output_tr)
        # optims_res['mini_batch'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}
        
        nn.optimizer = optims.SGD(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['sgd'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

        nn.optimizer = optims.Momentum(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['momentum'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

        nn.optimizer = optims.NAG(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['nag'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

        nn.optimizer = optims.Adagrad(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['adagrad'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

        nn.optimizer = optims.Adadelta(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['adadelta'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

        nn.optimizer = optims.RMSProp(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['rmsprop'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

        nn.optimizer = optims.Adam(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['adam'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

        nn.optimizer = optims.Adamax(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['adamax'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

        nn.optimizer = optims.Nadam(nn)
        rt = nn.optimizer.run(inp_tr, output_tr)
        optims_res['nadam'] = {'cost': deepcopy(rt[0]), 'epochs': deepcopy(rt[1]), 'accs': deepcopy(rt[2])}

    plt.figure(figsize=(15, 15))

    # plt.plot(optims_res['batch']['epochs'], optims_res['batch']['cost'], label='Batch')
    # plt.plot(optims_res['mini_batch']['epochs'], optims_res['mini_batch']['cost'], label='MiniBatch')
    plt.plot(optims_res['sgd']['epochs'], optims_res['sgd']['cost'], label='SGD')
    plt.plot(optims_res['momentum']['epochs'], optims_res['momentum']['cost'], label='Momentum')
    plt.plot(optims_res['nag']['epochs'], optims_res['nag']['cost'], label='NAG')
    plt.plot(optims_res['adagrad']['epochs'], optims_res['adagrad']['cost'], label='Adagrad')
    plt.plot(optims_res['adadelta']['epochs'], optims_res['adadelta']['cost'], label='Adadelta')
    plt.plot(optims_res['rmsprop']['epochs'], optims_res['rmsprop']['cost'], label='RMSProp')
    plt.plot(optims_res['adam']['epochs'], optims_res['adam']['cost'], label='Adam')
    plt.plot(optims_res['adamax']['epochs'], optims_res['adamax']['cost'], label='Adamax')
    plt.plot(optims_res['nadam']['epochs'], optims_res['nadam']['cost'], label='Nadam')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()
    plt.savefig('pic1-no-batch.png')
    plt.close()
    

    plt.figure(figsize=(15, 15))
    # plt.plot(optims_res['batch']['epochs'], optims_res['batch']['accs'], label='Batch')
    # plt.plot(optims_res['mini_batch']['epochs'], optims_res['mini_batch']['accs'], label='MiniBatch')
    plt.plot(optims_res['sgd']['epochs'], optims_res['sgd']['accs'], label='SGD')
    plt.plot(optims_res['momentum']['epochs'], optims_res['momentum']['accs'], label='Momentum')
    plt.plot(optims_res['nag']['epochs'], optims_res['nag']['accs'], label='NAG')
    plt.plot(optims_res['adagrad']['epochs'], optims_res['adagrad']['accs'], label='Adagrad')
    plt.plot(optims_res['adadelta']['epochs'], optims_res['adadelta']['accs'], label='Adadelta')
    plt.plot(optims_res['rmsprop']['epochs'], optims_res['rmsprop']['accs'], label='RMSProp')
    plt.plot(optims_res['adam']['epochs'], optims_res['adam']['accs'], label='Adam')
    plt.plot(optims_res['adamax']['epochs'], optims_res['adamax']['accs'], label='Adamax')
    plt.plot(optims_res['nadam']['epochs'], optims_res['nadam']['accs'], label='Nadam')

    plt.xlabel('Epochs')
    plt.ylabel('Acc')

    plt.legend()
    plt.savefig('pic2-no-batch.png')
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

if __name__ == '__main__':
    main()