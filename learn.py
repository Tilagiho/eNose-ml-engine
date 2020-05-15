from funcdataset import *
from linear_classifier import *
from oneHidden_reluAct_net import *

import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader

import time
import pickle

# constants
n_epochs = 100
learning_rate = 0.03
test_step = 2   # each test_step epochs the test stats are calculated

training_data = 'data/dataset_0403_full'  # string or list of strings
calculateFuncVectors = True
convertToRelativeVectors = True
normaliseData = True
balanceDatasets=True
# model_type = 'linear'
model_type = 'nn'
nHidden = 8

# definitions
def save_result(dataset, model, fig, output_name="", meta_comment=""):
    # get info
    if output_name == "":
        output_name = input("Enter name of predicted data")
    if meta_comment == "":
        meta_comment = input("Enter comment for meta data")

    # get confusion matrices
    train_pred = model.predict(torch.from_numpy(dataset.train_set).float())
    train_confusion_matrix = metrics.confusion_matrix(train_pred, dataset.train_classes)
    train_accuracy = train_confusion_matrix.trace() / train_confusion_matrix.sum()

    test_pred = model.predict(torch.from_numpy(dataset.test_set).float())
    test_confusion_matrix = metrics.confusion_matrix(test_pred, dataset.test_classes)
    test_accuracy = test_confusion_matrix.trace() / test_confusion_matrix.sum()

    # predict full data
    y_probs = model.get_scores(torch.from_numpy(dataset.full_data).float()).data.float().numpy()

    # store predicted classes in dataset
    dataset.set_detected_probs(y_probs)
    dataset.save(output_name)

    # create meta data
    metadata = output_name + ':\n'
    if meta_comment != "":
        metadata += "# " + meta_comment + '\n\n'
    metadata += 'classes:\n'
    metadata += ', '.join(dataset.label_encoder.classes_) + '\n\n'
    metadata += 'confusion matrix:\n'
    metadata += 'training data:\n'
    metadata += str(train_confusion_matrix) + '\n'
    metadata += 'accuracy = ' + str(train_accuracy) + '\n\n'
    metadata += 'test data:\n'
    metadata += str(test_confusion_matrix) + '\n'
    metadata += 'accuracy = ' + str(test_accuracy) + '\n\n'


    metadata += 'data used:\n'
    if (dataset.test_dirs != []):
        metadata += 'training:\n'
        metadata += str(dataset.train_dirs)
        metadata += '\ntest:\n'
        metadata += str(dataset.test_dirs)
    else:
        metadata += '\n'.join(dataset.get_filenames())

    # write meta data
    with open('predicted/' + output_name + '/metadata.txt', 'w+') as metafile:
        metafile.write(metadata)

    # save model
    model.name = output_name
    torch.jit.script(model).save('predicted/' + output_name + '/model.pt')
    # get_traced_model(model).save('predicted/' + output_name + '/model.pt')
    # pickle.dump(model, open('predicted/' + output_name + '/model.sav', 'wb'))

    # save plot
    fig.savefig("predicted/" + output_name + "/" + output_name + "_loss+acc.png")

def get_traced_model(model):
    example = torch.rand(1, model.nInputs)
    return torch.jit.trace(model, example)

def train_model(dataset, dataloader, net, n_epochs, learning_rate):
    if n_epochs > 0:
        # create plot
        fig, (loss_plot, accuracy_plot) = plt.subplots(2, 1, figsize=(11, 6))  # a figure with a 2x2 grid of Axes
        optim_string = ""

        if  isinstance(net.optimizer, optim.Adam):
            optim_string = 'Adam'
        elif isinstance(net.optimizer, optim.SGD):
            optime_string = "SGB (lr={})".format(learning_rate)
        fig.suptitle('{}, Optimiser: {}'.format(net.name, optim_string), fontsize=16)

        loss_plot.set_xlim(-1, n_epochs+1)
        loss_plot.set_ylim(0, 2)
        loss_plot.set_ylabel("loss")
        accuracy_plot.set_ylabel("accuracy")
        accuracy_plot.set_xlim(-1,n_epochs+1)
        accuracy_plot.set_ylim(0,1)
        accuracy_plot.set_xlabel("epoch")

        # prepare lists fór axes
        train_output = net.forward(torch.from_numpy(dataset.train_set).float())
        test_output = net.forward(torch.from_numpy(dataset.test_set).float())
        train_pred = train_output.argmax(1)
        test_pred = test_output.argmax(1)

        training_loss = net.loss(train_output, dataset.train_classes).detach().numpy().item()
        test_loss = net.loss(test_output, dataset.test_classes).detach().numpy().item()
        training_accuracy = (train_pred == dataset.train_classes).float().sum() / dataset.train_set.shape[0]
        test_accuracy = (test_pred == dataset.test_classes).float().sum() / dataset.test_set.shape[0]

        training_losses = [training_loss]
        test_losses = [test_loss]
        training_accuracies = [training_accuracy]
        test_accuracies = [test_accuracy]

        # add plots to axes
        train_loss_line, = loss_plot.plot(training_losses, [0], 'r-', label='training loss', zorder=10)
        test_loss_line, = loss_plot.plot(test_losses, [0], 'b-', label='test loss', zorder=0)
        loss_plot.legend()
        train_accuracy_line, = accuracy_plot.plot(training_accuracies, [0], 'r-', label='training accuracy', zorder=10)
        test_accuracy_line, = accuracy_plot.plot(test_accuracies, [0], 'b-', label='test accuracy', zorder=0)
        accuracy_plot.legend(loc=4)

        # print stats before first epoch
        print("[0]\ttraining:\tloss = {}\taccuracy = {}".format(training_loss, training_accuracy))
        print("\t\ttest:\tloss = {}\taccuracy = {}".format(test_loss, test_accuracy))

    # training loop
    for epoch in range(n_epochs):
        # reset
        running_loss = torch.zeros((1,1))   # culmulated loss of current epoch
        running_correct = 0                 # number of correctly predicted training samples in current epoch

        for i, batch in enumerate(dataloader):
            # get batch
            input_array, labels = batch

            # calculate output of batch & calculate loss
            output = net.forward(input_array.float())
            loss = net.loss(output, labels)

            # update running stats
            running_loss += loss * labels.shape[0]  # -> running_loss is sum of all losses after each epoch
            running_correct += (output.argmax(1) == labels).float().sum()

            # update weights
            net.step(loss)

        # training set statistics:
        # calculate losses
        training_loss = running_loss.detach().numpy().item() / dataset.train_set.shape[0]   # use running_loss

        # calculate accuracies
        training_accuracy = running_correct / dataset.train_set.shape[0]  # use running_correct

        # print stats
        print("[{:d}]\ttraining:\tloss = {}\taccuracy = {}".format(epoch + 1, training_loss, training_accuracy))

        # test set statistics:
        if epoch % test_step == 0:
            # forward output & predict classes
            test_output = net.forward(torch.from_numpy(dataset.test_set).float())
            test_pred = test_output.argmax(1)

            # calculate losses
            test_loss = net.loss(test_output, dataset.test_classes).detach().numpy().item()

            # calculate accuracies
            test_accuracy = (test_pred == dataset.test_classes).float().sum() / dataset.test_set.shape[0]

            # print stats
            print("\t\ttest:\tloss = {}\taccuracy = {}".format(test_loss, test_accuracy))

        # append lists
        training_losses.append(training_loss)
        test_losses.append(test_loss)
        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)

        # update graphs
        train_loss_line.set_xdata(range(len(training_losses)))
        test_loss_line.set_xdata(range(len(training_losses)))
        train_accuracy_line.set_xdata(range(len(training_losses)))
        test_accuracy_line.set_xdata(range(len(training_losses)))

        train_loss_line.set_ydata(training_losses)
        test_loss_line.set_ydata(test_losses)
        train_accuracy_line.set_ydata(training_accuracies)
        test_accuracy_line.set_ydata(test_accuracies)

        plt.draw()
        # plt.pause(1e-17)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(1e-17)

    if n_epochs > 0:
        # scale loss plot
        train_max = np.max(training_losses)
        test_max = np.max(test_losses)
        loss_plot.set_ylim(0, 1.1*np.max([train_max, test_max]))
    return (net, fig)

def ask_for_save():
    ans = input("Save predicted data? (y/n)")

    while ans.lower() != 'y' and ans.lower() != 'n':
        ans = input()

    return ans.lower() == 'y'

def loo_cross(dataset, startindex=0, name=''):
    if name == '':
        name = input("Base name for loo cross validation: ")

    for index in range(startindex, len(dataset.directory_data_dict.keys())):
        # split dataset
        dataset.setLooSplit(index, normaliseData=normaliseData, balanceDatasets=balanceDatasets)
        dataloader = DataLoader(dataset, batch_size=4, num_workers=3, shuffle=True)

        # define model
        if model_type == 'linear':
            net = LinearNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= name + "_" + str(index),
                                alpha=learning_rate,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative)
        elif model_type == 'nn':
            net = OneHiddenNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= name + "_" + str(index),
                                alpha=learning_rate,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative,
                                nHidden=nHidden)
        else:
            raise ValueError

        # train
        (model, train_figure) = train_model(dataset, dataloader, net, n_epochs, learning_rate)

        # save
        save_result(dataset, model, train_figure, name + "_" + str(index),
                    "Leave one out cross validation")

        plt.close(train_figure)

def tt_cross(dataset, startindex=0,n_splits=14, name=''):
    if name == '':
        name = input("Base name for kfold tt cross validation: ")

    dataset.prepareTTSplit(n_splits=n_splits)

    for index in range(startindex, n_splits):
        # split dataset
        dataset.setTTSplit(index, normaliseData=normaliseData, balanceDatasets=balanceDatasets)
        dataloader = DataLoader(dataset, batch_size=4, num_workers=3, shuffle=True)

        # define model
        if model_type == 'linear':
            net = LinearNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= name + "_" + str(index),
                                alpha=learning_rate,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative)
        elif model_type == 'nn':
            net = OneHiddenNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= name + "_" + str(index),
                                alpha=learning_rate,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative,
                                nHidden=nHidden)
        else:
            raise ValueError
        # train
        (model, train_figure) = train_model(dataset, dataloader, net, n_epochs, learning_rate)

        # save
        save_result(dataset, model, train_figure, name + "_" + str(index),
                    "kfold tt split cross validation")

        plt.close(train_figure)


# load dataset
dataset = FuncDataset(data_dir=training_data, convertToRelativeVectors=convertToRelativeVectors, calculateFuncVectors=calculateFuncVectors)