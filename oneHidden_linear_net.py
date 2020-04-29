import torch
import torch.optim as optim
import torch.nn as nn


# constants:
nHidden = 7
# learning rate

class LinearNetwork(nn.Module):
    def __init__(self, nFuncs, nClasses, alpha=0.01):
        super (LinearNetwork, self).__init__()

        self.nInputs = nFuncs
        self.nOutputs = nClasses
        self.name = "Neural net, 1 hidden layer without activation"


        # define network
        self.linear1 = nn.Linear(in_features=nFuncs, out_features=nHidden, bias=True)
        self.linear2 = nn.Linear(in_features=nHidden, out_features=nClasses, bias=True)

        # criterion + optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=alpha)

    def forward(self, vector):
        vector = self.linear1(vector)
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
