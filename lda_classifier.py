import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pickle
import funcdataset_old

# load data
train_dataset = funcdataset_old.FuncDataset("data/train")
test_dataset = funcdataset_old.FuncDataset("data/test")

# create numeric labels
le = LabelEncoder()
le.fit(train_dataset.training_classes)

# get training data and the corresponding numeric labels
X = train_dataset.training_data
y = le.transform(train_dataset.training_classes)

# fit linear classifier
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

# predict classes for classified vectors in test data
# -> for evaluation of results
y_test_true = le.transform(test_dataset.training_classes)
y_test_pred = clf.predict(test_dataset.training_data)

# predict classes for all vectors in test data
pred_classes = le.inverse_transform(clf.predict(test_dataset.full_data))

# set predicted data in dataset & save
test_dataset.set_detected_classes(pred_classes)

print(clf.coef_)

print("Save predicted data? (y/n)")
ans = input()

while ans.lower() != 'y' and ans.lower() != 'n':
    ans = input()

if ans.lower() == 'y':
    # get info
    print("Enter name of predicted data")
    output_name = input()
    print("Enter comment for meta data")
    meta_comment = input()

    # save predicted data
    test_dataset.save(output_name)

    # create meta data
    metadata = output_name + ':\n'
    if meta_comment != "":
        metadata += "# " + meta_comment + '\n\n'
    metadata += 'classes:\n'
    metadata += ', '.join(le.classes_) + '\n\n'
    metadata += 'confusion matrix:\n'
    metadata += str(confusion_matrix(y_test_true, y_test_pred)) + ' (absolute)\n'
    metadata += str(confusion_matrix(y_test_true, y_test_pred, normalize='true')) + '(relative)\n\n'
    metadata += 'data used:\n'
    metadata += '\n'.join(train_dataset.filenames)

    # write meta data
    with open('predicted/' + output_name + '/metadata.txt', 'w+') as metafile:
        metafile.write(metadata)

    # save model
    pickle.dump(clf, open('predicted/' + output_name + '/model', 'wb'))

# load model
#model = pickle.load(open('predicted/' + output_name + '/model', 'rb'))