import torch
import torch.optim as optim
import torch.nn as nn
import math


# constants:
# learning rate

class OneHiddenNetwork(nn.Module):
    def __init__(self, nFuncs, classList, isInputAbsolute = False , name ="LinearNet", mean=[], variance=[],alpha=0.01, nHidden=0):
        super (OneHiddenNetwork, self).__init__()

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

        # get n of hidden layers
        if nHidden == 0:
            self.nHidden = math.ceil((self.N + self.M) / 2)
        else:
            self.nHidden = nHidden

        # define network
        self.linear1 = nn.Linear(in_features=self.N, out_features=self.nHidden, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=self.nHidden, out_features=self.M, bias=True)

        # criterion + optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, vector):
        vector = self.linear1(vector)
        vector = self.relu(vector)
        vector = self.linear2(vector)
        return vector

    # loss function
    # output: network output of batch
    # target: target class of batch
    def loss(self, output, target):
        return self.criterion(output, target)

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
