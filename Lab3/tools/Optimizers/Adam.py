import numpy as np
from sklearn.utils import shuffle

class Adam:
    def __init__(self, model, b1=0.9, b2=0.999) -> None:
        self.model = model
        self.b1 = b1
        self.b2 = b2 

    def prediction(self, input):
        m = len(input[0])
        final_inputs = np.zeros((self.model.layers, m))
        for i in range(self.model.layers):
            for j in range(m):
                final_inputs[i][j] = (self.model.w[i] @ input[:, j] + self.model.b[i])
            # final_inputs[i] /= self.model.N

        final_inputs = final_inputs.T
        final_outputs = np.array([self.model.sigmoid(x) for x in final_inputs])
        
        res = None
        # print(len(final_outputs))
        if (final_outputs.ndim == 1):
            res = np.argmax(final_outputs)
        else:
            res = np.argmax(final_outputs, axis=1)
        
        return final_outputs, res

    def make_mini_batches(self, input, output, batch_size):
        mini_batches = []
        s1, s2 = shuffle(input, output, random_state=0)
        s1 = np.array(s1)
        s2 = np.array(s2)
        # print(s1[0:10, :])
        n_minibatches = s1.shape[0] // batch_size
        # print(n_minibatches)
        i = 0

        for i in range(n_minibatches + 1):
            x_mini = s1[i * batch_size: (i + 1) * batch_size, :]
            y_mini = s2[i * batch_size: (i + 1) * batch_size, :]
            # print(x_mini)
            # print(y_mini)
            if (len(x_mini) != 0):
                mini_batches.append((x_mini, y_mini))

        if n_minibatches % batch_size != 0:
            x_mini = s1[i * batch_size: s1.shape[0], :]
            y_mini = s2[i * batch_size: s1.shape[0], :]
            if (len(x_mini) != 0):
                mini_batches.append((x_mini, y_mini))

        return mini_batches

    def validate(self, pred, res):
        acc = 0
        if (isinstance(pred, np.ndarray)):
            for i in range(len(res)):
                if pred[i] == np.argmax(res[i]):
                    acc += 1
        else:
            if pred == np.argmax(res):
                acc += 1

        return acc

    def run(self, input, output):
        self.model.generate_weights()
        
        epochs_list = []
        cost_list = []
        acc_list = []

        k = len(output[0])
        count = len(input)
        V_w = np.zeros_like(self.model.w)
        M_w = np.zeros_like(self.model.w)
        V_b = np.zeros_like(self.model.b)
        M_b = np.zeros_like(self.model.b)
        eps = 1e-10

        for e in range(self.model.epochs):
            # losses = []
            losses = []
            acc = 0
            # w_grad, b_grad = np.zeros_like(self.model.w), np.zeros_like(self.model.b)
            mini_batches = self.make_mini_batches(input, output, 20)
            t = e + 1 # t > 0 !!!
            for mini_batch in mini_batches:
                mb_input, mb_output = mini_batch
                #Prediction for whole dataset
                mb_input_t = mb_input.T
                m = len(mb_input)
                # print(mb_input)
                # exit(-1)
                pred, res = self.prediction(input=mb_input_t)
                # print(pred)
                # exit(-1)
                loss = np.array([mb_output[i] - pred[i] for i in range(m)])
                # print(loss)
                # exit(-1)
                # loss_t = loss.T() 

                l = []
                #Compute gradients
                for i in range(m):
                    l.append([mb_input[i] * loss[i][p] for p in range(k)])
                    
                w_grad = 1/m * np.sum(l, axis=0)
                b_grad = 1/m * np.sum(loss, axis=0)

                #Update weights and bias
                for l in range(self.model.layers):
                    M_w[l] = self.b1 * M_w[l] + (1 - self.b1) * w_grad[l]
                    M_b[l] = self.b1 * M_b[l] + (1 - self.b1) * b_grad[l]

                    V_w[l] = self.b2 * V_w[l] + (1 - self.b2) * (w_grad[l] ** 2)
                    V_b[l] = self.b2 * V_b[l] + (1 - self.b2) * (b_grad[l] ** 2)

                    m_h_w = M_w[l] / (1 - self.b1 ** t)
                    m_h_b = M_b[l] / (1 - self.b1 ** t)
                    v_h_w = V_w[l] / (1 - self.b2 ** t)
                    v_h_b = V_b[l] / (1 - self.b2 ** t)

                    self.model.w[l] += self.model.lr * m_h_w / (np.sqrt(v_h_w) + eps)
                    # print(self.model.w[l])
                    self.model.b[l] += self.model.lr * m_h_b / (np.sqrt(v_h_b) + eps)

                acc += self.validate(res, mb_output)

                cost = np.mean([1/2 * np.square(loss[i]) for i in range(len(loss))])
                losses.append(cost)

            
            print('Adam')
            print(f'Epoch = {e + 1}, corrects = {acc}, all = {count}')
            print(f'Accuracy = {acc / count * 100}')
            print(f'Loss = {np.mean(losses)}\n')
            
            cost_list.append(np.mean(losses))
            epochs_list.append(e)
            acc_list.append(acc/count)

        return cost_list, epochs_list, acc_list