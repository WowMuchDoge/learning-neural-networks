import math
import numpy as np
import numpy.typing as npt

class NueralNet:
    # Generates weights randomly; sizes in order from input layer to output layer 
    def __init__(self, layer_sizes: list[int], learning_rate) -> None:
        # List of adjacency matrices representing all the 
        # connections from the current neurons to the next

        if len(layer_sizes) <= 1:
            raise Exception("Cannot create a neural network with a single layer.")

        self.connections = []

        random_generator = np.random.default_rng()

        for i in range(len(layer_sizes) - 1):
            # Each row is a neuron and each column of each row represents a weight to the column
            # neuron or input activation.
            self.connections.append(random_generator.random((layer_sizes[i + 1], layer_sizes[i] + 1)) / 3)
            # print(self.connections[-1])

        self.learning_rate = learning_rate

    # Activation function
    def sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def sigmoid_prime(self, x: float) -> float:
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # Uses mean squared error for the cost function
    def cost(self, observed: float, expected: float) -> float:
        return (observed - expected) ** 2

    def cost_prime(self, observed: float, expected: float) -> float:
        return 2 * (observed - expected)

    # Input activation is a list of numbers that represent each input neuron activation
    def feed_foward(self, activation: npt.NDArray, current_layer: int = 0) -> npt.NDArray:

        if current_layer == len(self.connections):
            return activation

        rows, cols = self.connections[current_layer].shape

        if len(activation) != cols - 1:
            raise Exception("Activation length does not match neurons for that layer.")

        current_activations: npt.NDArray = np.ones(rows)

        # Loops over each output neuron for adjacency matrix
        for row in range(rows):
            # Multiply the activations by the weights for that neuron and then add the bias
            neuron_sum = np.sum(activation * self.connections[current_layer][row][:-1]) + self.connections[current_layer][row][-1]
            current_activations[row] = self.sigmoid(neuron_sum)

        return self.feed_foward(current_activations, current_layer + 1)

    def record_feed_forward(self, activation: npt.NDArray, layer_activation: list[list[float]], depth: int = 0):
        layer_activation.append(activation.tolist())
        if depth == len(self.connections):
            return

        rows, cols = self.connections[depth].shape

        if cols - 1 != len(activation):
            raise Exception("Activation length does not match neurons for that layer.")

        cur_activation = np.empty(rows)

        for row in range(rows):
            neuron_sum = np.sum(activation * self.connections[depth][row][:-1]) + self.connections[depth][row][-1]
            cur_activation[row] = self.sigmoid(neuron_sum)

        self.record_feed_forward(cur_activation, layer_activation, depth + 1)

    # `full_activation` is the list of neurons, each outer list represents a layer and each inner represents activation of a neuron
    def calculate_gradient(self, gradient: list[npt.NDArray], cur_neuron: int, expected_activation: float, 
                                     full_activation: list[list[float]], depth: int = 0, cur_influence: float = 0.0) -> None:

        # The last layer is input neurons which we do not want to recurse over
        if depth == len(full_activation) - 1:
            return

        cur_activation = full_activation[-1 - depth][cur_neuron]

        z = None

        z = -math.log(1 / cur_activation - 1) # Sigmoid function is reversable, though reversed it is technically not a function
        cur_weights = self.connections[-1 - depth][cur_neuron][:-1] # Excludes bias

        if depth == 0:
            cur_influence = self.cost_prime(cur_activation, expected_activation)

        # This is the bias influence, there is only one per neuron so we don't have to loop over it
        gradient[-1 - depth][cur_neuron][-1] += self.sigmoid_prime(z) * cur_influence

        for index, weight in enumerate(cur_weights):
            prev_activation = full_activation[-1 - 1 - depth][index]

            weight_influence = prev_activation * self.sigmoid_prime(z) * cur_influence 
            activation_influence = weight * self.sigmoid_prime(z) * cur_influence

            gradient[-1 - depth][cur_neuron][index] += weight_influence
            self.calculate_gradient(gradient, index, 0.0, full_activation, depth + 1, activation_influence)

    # Just stochastic gradient descent, no mini batch (for now)
    def back_propagate(self, training_data: npt.NDArray, expected_outputs: npt.NDArray) -> None:
        if training_data.ndim != 2:
            raise Exception("Training data must be stored in a two dimensional array.")

        if expected_outputs.ndim != 2:
            raise Exception("Expected outputs must be stored in a two dimensional array.")

        # Each row is new training and output example 
        _, training_width = training_data.shape
        _, output_width = expected_outputs.shape

        if training_width != self.connections[0].shape[1] - 1:
            raise Exception("Training data dimensions do not match that of the neural network")
        if output_width != self.connections[-1].shape[0]:
            raise Exception("Expected output dimensions do not match that of the neural network")

        last_gradient = [] 

        for training_datum, expected_output in zip(training_data, expected_outputs):
            # Basically the connections but represents how much each weight influences the cost vector
            gradient = []

            # Same logic as the constructor, just constructing an empty gradient instead of the actual weights and biases
            for i in range(len(self.connections)):
                gradient.append(np.zeros(self.connections[i].shape))

            layer_activation = []
            self.record_feed_forward(training_datum, layer_activation)

            for index, expected in enumerate(expected_output):
                self.calculate_gradient(gradient, index, expected, layer_activation)

            for i in range(len(gradient)):
                if len(last_gradient) < len(gradient):
                    last_gradient.append(gradient[i].copy())
                else:
                    last_gradient[i] = (last_gradient[i] + gradient[i]) / 2

        for i in range(len(last_gradient)):
            self.connections[i] += -self.learning_rate * last_gradient[i]

        cost = np.zeros(expected_outputs.shape[1])

        for training_datum, expected_output in zip(training_data, expected_outputs):
            cost = cost + (self.feed_foward(training_datum) - expected_output) ** 2

        print("Cost after backprop is", np.average(cost / expected_outputs.shape[0]))