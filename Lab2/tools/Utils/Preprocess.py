import numpy as np

class Preprocess:
   
    @staticmethod
    def normalize_input(input):
        return [ (1 if input[i] > 0 else 0) for i in range(len(input))]

    @staticmethod
    def normalize_whole_input(input, MNIST=False):
        if MNIST:
            rt = np.zeros((input.shape[0], 784))
            for i in range(input.shape[0]):
                rt[i] = np.reshape(input[i], 784)
            rt /= 255
            return rt
        return [[(1 if input[k][i] > 0 else 0) for i in range(len(input[k]))] for k in range(len(input))]

    @staticmethod
    def get_normalize_output(res, k=10):
        output = np.zeros(k)
        output[res] = 1
        return output

    @staticmethod
    def get_normalize_output_for_whole_ds(res, k=10):
        n = len(res)
        output = np.zeros((n, k))
        for i in range(n):
            output[i][res[i]] = 1

        return output