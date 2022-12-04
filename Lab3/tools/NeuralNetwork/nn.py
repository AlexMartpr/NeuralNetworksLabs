import random
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

class NN:
    def __init__(self, N, OUT, layers=[], optimizer=None, epochs=100, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.N = N
        self.OUT = OUT
        self.neurons = layers
        self.layers = []
        self.optimizer = optimizer
        self.lambd = 0.0001
        self.prob = 0.5
        self.active_layers = [] 
        self.w = []
        self.b = []
        
    def generate_weights(self):
        i = 0
        # For hidden layers
        for n in (self.neurons):
            self.layers.append({
                'w': [],
                'b': [],
            })
            if i == 0:
                self.layers[i]['w'] = np.random.rand(n, self.N)
            else:
                self.layers[i]['w'] = np.random.rand(n, self.neurons[i - 1])

            self.layers[i]['b'] = np.random.rand(n)
        
            i += 1
        
        #Output layer
        self.layers.append({
            'w': [],
            'b': [],
        })

        if i == 0:
            self.neurons.append(self.OUT)
            self.layers[i]['w'] = np.random.rand(self.OUT, self.N)
        else:
            self.layers[i]['w'] = np.random.rand(self.OUT, self.neurons[i - 1])

        self.layers[i]['b'] = np.random.randn(self.OUT)
    
                

    def prediction(self, input, l):
        # print(len(l['w']) == 0) 
        final_inputs = l['w'] @ input + l['b']
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs
    
    def train(self, input, output, regular=None):
        koef = 1 / (1 - self.prob)
        epochs_list = []
        cost_list = []
        acc_list = []

        count = len(input)
        for e in range(self.epochs):
            correct = 0
            losses = []
            active = []
            new_layers_m = []
            new_l = []

            #Dropout
            if regular == 3:
                ll = len(self.layers) - 1

                for l in range(ll):
                    # print(l)
                    active.append([])
                    for j in range(len(self.layers[l]['w'])):
                        check = random.choice(np.arange(1,3))
                        if check != 1:
                            active[l].append(j)

                for i, a in enumerate(active):
                    new_layers_m.append([])
                    new_l.append({
                            'w': [],
                            'b': [],
                        })
                    for k in range(len(self.layers[i]['w'])):
                        if k in a:
                            new_layers_m[i].append(1)
                            new_l[i]['w'].append(self.layers[i]['w'][k])
                            new_l[i]['b'].append(self.layers[i]['b'][k])
                        else:
                            new_layers_m[i].append(0)
                            new_l[i]['w'].append(np.zeros_like(self.layers[i]['w'][k]))
                            new_l[i]['b'].append(np.zeros_like(self.layers[i]['b'][k]))
                
            #Normalize input and output before call this method!!!
            for (test, res) in zip(input, output):
                predicts = []
                deltas = [[] for _ in range(len(self.layers))]

                #Feedforward
                i = 0
                for i, l in enumerate(self.layers):
                    pred = None
                    # print(i)
                    if i == 0:
                        if regular == 3:
                            pred = self.prediction(test.T, l) * koef
                            pred *= np.array(new_layers_m[i]) 
                        else:
                            pred = self.prediction(test.T, l)
                    else:
                        if regular == 3:
                            if i != len(self.layers) - 1:
                                pred = self.prediction(predicts[i - 1].T, l) * koef
                                pred *= np.array(new_layers_m[i])
                            else:
                                pred = self.prediction(predicts[i - 1].T, l)
                        else:
                            pred = self.prediction(predicts[i - 1].T, l)
                    
                    predicts.append(pred)

                res_nn = np.argmax(predicts[i])
                loss = predicts[i] - res
                loss_out = loss * self.sigmoid_prime(predicts[i])
                # print(len(loss))
                deltas[i] = loss_out
                # if regular == 3:
                #         deltas[i] *= np.array(new_layers[i])

                #Backpropogation
                for i in range(len(self.layers) - 2, -1, -1):
                    deltas[i] = deltas[i + 1] @ self.layers[i + 1]['w'] * self.sigmoid_prime(predicts[i])
                    # if regular == 3:
                    #     print(active[i])
                    #     print(deltas[i])
                    #     deltas[i] *= np.array(new_layers_m[i])
                    #     print(deltas[i])

                #Update
                for i in range(len(self.layers) - 1, -1, -1):
                    dw = None

                    if i == len(self.layers) - 1 and i != 0:
                        dw = np.kron(deltas[i], predicts[i - 1]).reshape(self.OUT, len(predicts[i - 1]))
                    elif i == 0:
                        dw = np.kron(deltas[i], test).reshape(self.neurons[i], len(test))
                    else:
                        dw = np.kron(deltas[i], predicts[i - 1]).reshape(self.neurons[i], len(predicts[i - 1]))

                    if regular == 1:
                        self.layers[i]['w'] -= (self.lr * dw + self.lr * self.lambd * np.sign(self.layers[i]['w']))
                    elif regular == 2:
                        self.layers[i]['w'] -= self.lr * dw + self.lr * self.lambd * self.layers[i]['w']
                    elif regular == 3:
                        if i != len(self.layers) - 1:
                            for k in active[i]:
                                dw[k] = np.zeros_like(dw[k])
                            # print(active[i])
                            # # print(dw[active[i]])
                            # exit(-1)
                        dw *= koef
                        self.layers[i]['w'] -= self.lr * dw
                    else:
                        self.layers[i]['w'] -= self.lr * dw

                    self.layers[i]['b'] -= self.lr * deltas[i]
                    

                cost = np.mean(1/2 * np.square(loss))
                losses.append(cost)

                if res_nn == np.argmax(res):
                    correct += 1

            print('MLP with SGD')
            print(f'Epoch = {e + 1}, corrects = {correct}, all = {count}')
            print(f'Accuracy = {correct / count * 100}')
            print(f'Loss = {np.mean(losses)}\n')

            cost_list.append(np.mean(losses))
            epochs_list.append(e)
            acc_list.append(correct/count)

        return cost_list, epochs_list, acc_list

    def predict(self, input, output):
        corrects = 0 
        count = input.shape[0]
        #Feedforward
        for (test, res) in zip(input, output):
            i = 0
            predicts = []
            for i, l in enumerate(self.layers):
                pred = None
                
                if i == 0:
                    pred = self.prediction(test.T, l)
                else:
                    pred = self.prediction(predicts[i - 1].T, l)
                
                predicts.append(pred)

            res_nn = np.argmax(predicts[i])
            _res = np.argmax(res)

            if res_nn == _res:
                corrects += 1

        return corrects / count 

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))