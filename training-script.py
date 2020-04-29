import argparse

parser = argparse.ArgumentParser(
    description=
    'Train ML models based on the dataset in SOURCE.\n'
    'SOURCE is expected to contain subdirectories, each directory containing measurements belonging together.')

# flags
parser.add_argument('-nonorm', action='store_true',
                    help='do not normalise dataset based on training set')
parser.add_argument('-nobal', action='store_true',
                    help='do not balance training & test set')
parser.add_argument('-abs', action='store_true',
                    help='use absolute vectors as input, relative vectors are used by default')
parser.add_argument('-func', action='store_true',
                    help='use functionalisation vectors as input, measurement vector is used by default')

# parameters
parser.add_argument('-model', metavar='MODEL',
                     help='select type of model to be trained')
parser.add_argument('-hlw', type=int,
                     help='hidden layer width: use HLW neurons in each hidden layer')
parser.add_argument('-hld', type=int,
                     help='hidden layer depth: use HLD hidden layers')
parser.add_argument('-batchsize', type=int,
                     help='set batch size')

# source directory
parser.add_argument('SOURCE')

cli_input = parser.parse_args()

print(cli_input.func)