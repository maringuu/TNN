import numpy as np
import pandas as pd

learning_rate = 0.8

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class State:
    def __init__(self, in_dim, out_dim):
        """
        Constructor for State.
        The class represents a two layer (no hidden layers) densly connected neural Network.

        :param in_dim: The amount of input neurons.
        :param out_dim: The amount of output neurons.
        """
        # State for backpropagation
        self._out = None
        self._deriv = None
        self._delta_w = None

        self._in_dim = in_dim
        self._out_dim = out_dim
        # Weights for the output layer
        # +1 for bias
        # The weights shall be between -0.5 and 0.5
        self._weights = np.random.rand(self._out_dim, self._in_dim + 1) - 0.5

    # TODO this is useless. Remove. Removing this should also reduce allocations by a lot
    def reset(self):
        self._out = None
        self._deriv = None
        self._delta_w = None

    def forward(self, input):
        """
        This function does the forward propagation.

        :param input: The input for the NN. A numpy array of dimension ``out_dim`` as given in the constructor.
        :return: The outout of the NN. A numpy array of dimension ``out_dim`` as given in the constructor.
        """
        input_with_bias = np.concatenate((input, np.ones(1)))
        # Needed for backward
        self._input = input_with_bias
        net_sum = np.dot(self._weights, input_with_bias)
        self._out = sigmoid(net_sum)
        self._deriv = self._out * (1 - self._out)

        return self._out

    def backward(self, expected):
        """
        This function does the backward propagation.
        The loss function is the mean squared error.

        :param: The expected output for the last input given to ``forward``.
        """
        in_dim_with_bias = self._in_dim + 1
        self._delta_w = np.zeros((self._out_dim, in_dim_with_bias))
        # Update the weights of perceptron
        # TODO use np.indices instead
        for i in np.arange(self._out_dim):
            self._delta_w[i,:] = learning_rate * (expected[i] - self._out[i]) * self._deriv[i] * self._input

    def update(self):
        """
        Updates the weights of the neurons.
        Must be called after calling ``forward`` and ``backward``.
        """
        self._weights += self._delta_w
        self.reset()


def main():
    path = "data/PA-A_training_data_01.txt"
    df = pd.read_csv(
        path,
        sep=" ",
        comment="#",
        names=["in0", "in1", "out0"],
    )

    s = State(2, 1)
    iterations = 5000

for _ in range(iterations):
    # We just repeditly sample from the training set
    sample = df.sample()
    input = np.array(sample[["in0", "in1"]]).squeeze()
    output = np.array(sample["out0"])

    s.forward(input)
    s.backward(output)
    s.update()


    return s

def little_mans_statistic(state):
    print(s.forward(np.array([0,0])))
    print(s.forward(np.array([0,1])))
    print(s.forward(np.array([0,0])))
    print(s.forward(np.array([0,0])))
