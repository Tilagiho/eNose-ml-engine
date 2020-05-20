import torch
import torch.optim as optim
import torch.nn as nn
import math

class MultiLayerPerceptron(nn.Module):
    def __init__(self, nFuncs, classList, isInputAbsolute = False , name ="MultiLayerPerceptron", mean=[], variance=[], nHiddenLayers=0, nHiddenNeuronPerLayer=0, loss_func=None):
        super (MultiLayerPerceptron, self).__init__()

        # list of class names
        self.classList = ','.join(list(classList))

        # N inputs, M outputs
        self.N = nFuncs
        self.M = len(classList)

        # meta information:
        # name
        self.name = name

        # should input be absolute? (or relative to base vector)
        self.isInputAbsolute = isInputAbsolute

        # info about normalisation
        self.mean = list(mean)
        self.variance = list(variance)

        # get variables for hidden layers
        if nHiddenNeuronPerLayer == 0:
            self.nHiddenNeuronPerLayer = math.ceil((self.N + self.M) / 2)
        else:
            self.nHiddenNeuronPerLayer = nHiddenNeuronPerLayer

        self.nHiddenLayers = nHiddenLayers

        # define network
        self.networkFunctions = nn.ModuleList()

        # no hidden layers:
        # fully connected linear net
        if self.nHiddenLayers <= 0:
            self.networkFunctions.append(nn.Linear(in_features=self.N, out_features=self.M, bias=True))
        # with hidden layers:
        # multi layer perceptron with relu activation
        else:
            # input layer -> hidden layer
            self.networkFunctions.append(nn.Linear(in_features=self.N, out_features=self.nHiddenNeuronPerLayer, bias=True))
            self.networkFunctions.append(nn.ReLU())

            # hidden layer -> hidden layer
            for i in range(1, self.nHiddenLayers):
                self.networkFunctions.append(nn.Linear(in_features=self.nHiddenNeuronPerLayer, out_features=self.nHiddenNeuronPerLayer, bias=True))
                self.networkFunctions.append(nn.ReLU())

            # hidden layer -> output layer
            self.networkFunctions.append(nn.Linear(in_features=self.nHiddenNeuronPerLayer, out_features=self.M, bias=True))

        # loss function
        self.loss_func = loss_func

    def forward(self, vector):
        for func in self.networkFunctions:
            vector = func(vector)

        return vector

    # loss function
    # output: network output of batch
    # target: target class of batch
    def loss(self, output, target):
        return self.loss_func(output, target)

    # step:
    # forward batch through network, calculate loss & and update weigths
    def step(self, loss):
        # reset gradients
        self.optimizer.zero_grad()

        # calc derivatives
        loss.backward()

        # update weights
        self.optimizer.step()

    # get_scores:
    # return vector of predicted probabilities of the classes
    def get_scores(self, X):
        output = self.forward(X)
        return nn.functional.softmax(output, dim=1)

    def get_class(self, vector):
        return torch.max(vector)[-1]

    # predict:
    # returns numpy array of predicted classes for data
    def predict(self, X):
        output = self.forward(X)
        y_pred = output.argmax(1)

        return y_pred
