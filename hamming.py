import numpy as np
import argparse
from scipy.spatial.distance import hamming


def parse_args():
    program_examples = '''Example of use:
    python hamming.py --path=test_4.csv --data=samples_vector.csv
    python hamming.py --path=vector.csv -i=150 --distance=4
    '''
    default_training_set = "samples_vector.csv"

    parser = argparse.ArgumentParser(
        description='Hamming net',
        epilog=program_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--data', type=str, help='Path to the training data set (*.csv)', default=default_training_set)
    parser.add_argument('-m', '--distance', type=int, default=1, help='Max Hamming distance')
    parser.add_argument('-i', '--iteration', type=int, help='Max count of iterations', default=40)
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the vector (*.csv)')
    return parser.parse_args()


def result_class(y):
    _eps = 0.005
    y_max = np.max(y)
    for i in range(len(y)):
        if abs(y[i] - y_max) < _eps:
            return i + 1
    return 0


class HammingNet:
    def __init__(self, max_distance=1, max_iter=40):
        self.max_distance = max_distance
        self.max_iter = max_iter

        self.classes = 0
        self.vector_length = 0
        self.weights = np.array([])
        self.E = np.array([])
        self.threshold = 1

    def activation(self, y):
        _activation = y.copy()
        _activation[_activation < 0] = 0
        _activation[_activation > self.threshold] = self.threshold
        return _activation

    def train(self, X):
        shape = X.shape
        self.classes = shape[0]
        self.vector_length = shape[1]
        self.threshold = self.vector_length / 2.0
        self.weights = X.copy() / 2
        epsilon = 0.05  # TODO: change to ( 0; 1/K ]
        self.E = np.full((self.classes, self.classes), -epsilon)
        np.fill_diagonal(self.E, 1)

    def predict(self, x):
        previous_y = self.weights.dot(x) + self.threshold  # y1
        y2 = previous_y.copy()
        iterations = 1
        distance = self.max_distance + 1

        while distance > self.max_distance and iterations < self.max_iter:
            y2 = self.activation(self.E.dot(previous_y))
            distance = hamming(previous_y, y2)
            previous_y = y2.copy()
            iterations += 1
        print("iterations =", iterations)
        return result_class(y2)


args_res = parse_args()
training_data = args_res.data

X = np.genfromtxt(training_data, delimiter=',')
network = HammingNet(max_distance=args_res.distance, max_iter=args_res.iteration)
print("size of train data set =", X.shape)
network.train(X)

path_to_vector = args_res.path
x = np.genfromtxt(path_to_vector, delimiter=',')
print("predicted class =", network.predict(x))
