import numpy as np

class Batch:
    def __init__(self, model) -> None:
        self.model = model

    def prediction(self, input):
        m = len(input[0])
        final_inputs = np.zeros((self.model.layers, m))
        for i in range(self.model.layers):
            for j in range(m):
                final_inputs[i][j] = (self.model.w[i] @ input[:, j] + self.model.b[i])
            # final_inputs[i] /= (self.model.N + 1)

        final_inputs = final_inputs.T
    
        # final_inputs = np.array([((self.model.w[i] @ input)  + self.model.b[i]) / self.model.N for i in range(self.model.layers)]).T

        final_outputs = np.array([self.model.sigmoid(x) for x in final_inputs])
        
        return final_outputs, np.argmax(final_outputs, axis=1)

    def validate(self, pred, res):
        acc = 0

        for i in range(len(res)):
            if pred[i] == np.argmax(res[i]):
                acc += 1

        return acc


    def run(self, input, output):
        self.model.generate_weights()
        
        epochs_list = []
        cost_list = []
        acc_list = []

        input_t = input.T
        k = len(output[0])
        m = input.shape[0]

        for e in range(self.model.epochs):
            # losses = []
            acc = 0
            #Prediction for whole dataset
            pred, res = self.prediction(input=input_t)
            loss = np.array([output[i] - pred[i] for i in range(m)])
            # loss_t = loss.T() 

            l = []
            #Compute gradients
            for i in range(m):
                l.append([input[i] * loss[i][p] for p in range(k)])
                 
            w_grad = 1/m * np.sum(l, axis=0)
            b_grad = 1/m * np.sum(loss, axis=0)

            #Update weights and bias
            for l in range(self.model.layers):
                self.model.w[l] += self.model.lr * w_grad[l]
                self.model.b[l] += self.model.lr * b_grad[l]

            acc = self.validate(res, output)

            cost = np.mean([1/2 * np.square(loss[i]) for i in range(len(loss))])
            # losses.append(cost)

            print('Batch')
            print(f'Epoch = {e + 1}, corrects = {acc}, all = {m}')
            print(f'Accuracy = {acc / m * 100}')
            print(f'Loss = {cost}\n')

            cost_list.append(cost)
            epochs_list.append(e)
            acc_list.append(acc/m)


        return cost_list, epochs_list, acc_list