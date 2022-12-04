import math
import numpy as np

class NN:
    def __init__(self, N, layers, optimizer=None, epochs=100, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.N = N
        self.layers = layers
        self.optimizer = optimizer
        self.w = []
        self.b = []
        
    def generate_weights(self):
        for _ in range(self.layers):
            self.w.append(np.random.randn(self.N) / math.sqrt(self.N))
            self.b.append(np.random.randn())

    def predict(self, input):
        final_inputs = [((self.w[i] @ input)  + self.b[i]) / self.N for i in range(self.layers)]
        final_outputs = [self.sigmoid(x) for x in final_inputs]
        
        return final_outputs, np.argmax(final_outputs)

        
    def normalize_input(self, input):
        return [ (1 if input[i] > 0 else 0) for i in range(len(input))]
    
    #TODO: add backpropogation for testing accuracy
    def train(self, input, output):
        final_outputs, res = self.predict(input)
        errors = [((output[i] - final_outputs[i])) for i in range(len(output))]
        
        for l in range(self.layers):
            self.w[l] += self.lr * errors[l] * input
            self.b[l] += self.lr * errors[l]

        return final_outputs, res

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))