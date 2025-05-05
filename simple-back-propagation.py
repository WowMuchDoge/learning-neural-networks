# Essentially a linked list of neurons for testing back propagation

import math

learning_rate = 0.2

def sigmoid(value: float) -> float:
    return 1 / (1 + math.e ** -value)

def sigmoid_prime(point: float) -> float:
    return sigmoid(point) * (1 - sigmoid(point))

def means_squared_error(calculated: float, expected: float) -> float:
    return (calculated - expected) ** 2

def means_squared_error_prime(calculated: float, expected: float) -> float:
    return 2 * (calculated - expected)

class Neuron:
    def __init__(self, is_input: bool, weight: float, bias: float):
        self.is_input = is_input
        self.weight = weight
        self.bias = bias

    def set_input_neuron(self, neuron: 'Neuron'):
        self.input = neuron

    def set_input_activation(self, activation: float) -> None:
        self.activation = activation

    def adjust_parameters(self, new_weight: float, new_bias: float) -> None:
        self.weight = new_weight
        self.bias = new_bias

    def activate(self) -> float:
        if self.is_input:
            if self.activation != None:
                return self.activation

            raise Exception("Activation on input neuron is not configured")

        if self.input == None:
            raise Exception("No input neuron on a hidden or output layer")

        return sigmoid(self.weight * self.input.activate() + self.bias)

    def calculate_gradient(self, gradient: list[tuple[float, float]], depth: int, expected: float, cur_activation_influence: float) -> None:
        if self.is_input: # Indicates that this is an input neuron that has no activation or weights
            return
        elif depth == 0: # Indicates that this is the ouput neuron
            # All calculating via influence on cost

            print(f"Cost for output neuron is {means_squared_error(self.activate(), expected)}")

            input_activation = self.input.activate()
            z = self.weight * input_activation + self.bias

            weight_influence = input_activation * sigmoid_prime(z) * means_squared_error_prime(sigmoid(z), expected)
            bias_influence = sigmoid_prime(z) * means_squared_error_prime(sigmoid(z), expected)

            gradient.append((weight_influence, bias_influence))

            self.input.calculate_gradient(gradient, depth + 1, expected, self.weight * sigmoid_prime(z) * means_squared_error_prime(sigmoid(z), expected))
        else:
            input_activation = self.input.activate()
            z = self.weight * input_activation + self.bias

            weight_influence = input_activation * sigmoid_prime(z) * cur_activation_influence
            bias_influence = sigmoid_prime(z) * cur_activation_influence

            gradient.append((weight_influence, bias_influence))

            self.input.calculate_gradient(gradient, depth + 1, expected, self.weight * sigmoid_prime(z) * cur_activation_influence)

    def back_propagate(self, gradient_vector: list[tuple[float, float]]) -> None:
        if self.is_input:
            if len(gradient_vector) > 0:
                raise Exception("Too many elements in gradient vector.")
            return
        elif len(gradient_vector) == 0:
            raise Exception("Not enough elements in gradient vector.")

        weight_change = -learning_rate * gradient_vector[0][0]
        bias_change = -learning_rate * gradient_vector[0][1]

        self.weight += weight_change
        self.bias += bias_change

input_neuron = Neuron(True, 0.0, 0.0)
input_neuron.set_input_activation(0.5)

hidden_neuron = Neuron(False, 0.3, 0.2)
hidden_neuron.set_input_neuron(input_neuron)

output_neuron = Neuron(False, 0.1, 0.04)
output_neuron.set_input_neuron(hidden_neuron)

for _ in range(10):
    gradient = []
    output_neuron.calculate_gradient(gradient, 0, 1.0, 0.0)
    output_neuron.back_propagate(gradient)
