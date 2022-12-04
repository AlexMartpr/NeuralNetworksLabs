import numpy as np

class SGD:
    def __init__(self, model) -> None:
        self.model = model

    def prediction(self, input):
        final_inputs = np.array([((self.model.w[i] @ input)  + self.model.b[i]) / self.model.N for i in range(self.model.layers)])
        final_outputs = np.array([self.model.sigmoid(x) for x in final_inputs])
        
        return final_outputs, np.argmax(final_outputs)

    def run(self, input, output):
        self.model.generate_weights()
        
        epochs_list = []
        cost_list = []
        acc_list = []

        count = len(input)
        for e in range(self.model.epochs + 1):
            correct = 0
            losses = []
            #Normalize input and output before call this method!!!
            for (test, res) in zip(input, output):
                pred, res_nn = self.prediction(test)
                loss = res - pred

                w_grad = np.array([test * loss[i] for i in range(len(loss)) ])
                b_grad = loss

                for l in range(self.model.layers):
                    self.model.w[l] += self.model.lr * w_grad[l]
                    self.model.b[l] += self.model.lr * b_grad[l]

                cost = np.mean(1/2 * np.square(loss))
                losses.append(cost)

                if res_nn == np.argmax(res):
                    correct += 1

            print('SGD')
            print(f'Epoch = {e + 1}, corrects = {correct}, all = {count}')
            print(f'Accuracy = {correct / count * 100}')
            print(f'Loss = {np.mean(losses)}\n')

            cost_list.append(np.mean(losses))
            epochs_list.append(e)
            acc_list.append(correct/count)

        return cost_list, epochs_list, acc_list   

