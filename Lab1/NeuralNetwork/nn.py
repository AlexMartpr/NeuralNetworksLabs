import math
import numpy as np

class NN:
    def __init__(self, N, layers, lr=0.05):
        self.lr = lr
        self.layers = layers
        self.N = N
        self.w = []
        
    def generate_weights(self):
        for _ in range(self.layers):
            self.w.append(np.random.randn(self.N) / math.sqrt(self.N))

    def predict(self, input):
        final_inputs = [np.dot(self.w[i], input) / self.N for i in range(len(self.w))]
        final_outputs = [self.sigmoid(x) for x in final_inputs]
        
        return final_outputs, np.argmax(final_outputs)

        
    def normalize_input(self, input):
        return [ (1 if input[i] > 0 else 0) for i in range(len(input))]
    
    #TODO: add backpropogation for testing accuracy
    def train(self, input, output):
        final_outputs, res = self.predict(input)
        errors = [((output[i] - final_outputs[i])) for i in range(len(output))]
        
        for l in range(self.layers):
            for i in range(self.N):
                self.w[l][i] += errors[l] * input[i] * self.lr

        return final_outputs, res

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))