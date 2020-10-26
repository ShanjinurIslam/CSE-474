# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)


# %%
def generate(filename):
    df = pd.read_csv(filename, delimiter='\s+', header=None)
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    num_classes = len(np.unique(y))
    N = len(X)

    Y = np.zeros((N, num_classes))

    for i in range(N):
        Y[i][y[i]-1] = 1

    return X, Y


# %%
X_train, Y_train = generate('trainNN.txt')
X_test, Y_test = generate('testNN.txt')


# %%
class L_Layer_NN():
    def __init__(self, k, lr=0.25, max_epoch=1000):
        self.k = k
        self.L = len(self.k)-1
        self.weights = []
        self.lr = lr
        self.max_epoch = max_epoch

    def init_weights(self):
        for i in range(1, len(self.k)):
            self.weights.append(np.random.uniform(
                0, 1, (self.k[i], self.k[i-1]+1)))

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def feed_forward(self, x_i):
        self.y = []
        self.v = []

        self.y.append(x_i)

        for r in range(self.L):
            v_r = np.dot(self.weights[r], self.y[r])
            self.v.append(v_r)
            y_r = self.sigmoid(v_r)
            y_r = np.insert(y_r, 0, values=1, axis=0)
            self.y.append(y_r)

    def backpropagation(self, y_true):
        self.delta = [0]*self.L
        self.y_hat = self.y[-1]
        self.delta[self.L-1] = np.multiply(self.y_hat[1:] -
                                           y_true, self.derivative(self.v[self.L-1]))

        for r in reversed(range(0, self.L-1)):
            e_r = np.dot(self.delta[r+1], self.weights[r+1][:, 1:])
            self.delta[r] = np.multiply(e_r, self.derivative(self.v[r]))

    def update_weights(self):
        self.y = self.y[:-1]
        for r in range(self.L):
            self.weights[r] = self.weights[r] - self.lr * \
                ((self.delta[r].reshape(-1, 1))*self.y[r])

    def calculate_cost(self, y_hat, y_true):
        return 0.5*np.sum((y_hat-y_true)**2, axis=0)

    def fit(self, X, Y):
        self.init_weights()  # initialize weights
        X = np.insert(X, 0, values=1, axis=1)  # appending 1 to all x vectors

        for epoch in range(self.max_epoch):
            total_cost = 0
            for (x_i, y_i) in zip(X, Y):
                self.feed_forward(x_i)
                self.backpropagation(y_i)
                self.update_weights()
                total_cost += self.calculate_cost(self.y_hat[1:], y_i)
            print("Epoch: ", epoch, total_cost)
            if total_cost < 5:
                break
        return

    def predict(self, X, Y):
        correct = 0
        sample_number = 0

        X = np.insert(X, 0, values=1, axis=1)

        for (x_i, y_i) in zip(X, Y):
            self.feed_forward(x_i)
            sample_number += 1

            predicted_class = np.argmax(self.y[-1][1:])
            actual_class = np.argmax(y_i)

            if predicted_class == actual_class:
                correct += 1
            else:
                print(sample_number, x_i, y_i,
                      actual_class+1, predicted_class+1)

        return correct/len(X)


# %%
min_max_scaler = MinMaxScaler()  # min max scaler
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

input_neurons = X_train.shape[1]
output_neurons = Y_train.shape[1]

model = L_Layer_NN([input_neurons, 5, 3, 4, output_neurons], lr=1)
model.fit(X_train, Y_train)


# %%
model.predict(X_train, Y_train)


# %%
model.predict(X_test, Y_test)
