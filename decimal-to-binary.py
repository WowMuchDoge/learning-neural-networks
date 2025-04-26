def approx_sigmoid(x: float) -> float:
    # Not for efficiency, just I am too lazy to find the constant e 
    return x / (2 + 2 * abs(x)) + 0.5

class BinaryNeuron:
    def __init__(self, weights: list[float], threshold: float) -> None:
        self.weights = weights
        self.threshold = threshold

    def calculate_activation(self, activation: list[float]) -> bool:
        if len(activation) != len(self.weights):
            print("Number of weights does not match number of inputs!")
            return False 

        sum: float = 0
        for i in range(len(activation)):
            sum += activation[i] * self.weights[i]

        # We have 10 digits representing our input
        return approx_sigmoid(sum) > self.threshold

def bin_len(n: int):
    it = 0
    p = 1

    while n >= p:
        p *= 2
        it += 1

    return it 

def generate_binary_nums(n: int, cur: list[int], final: list[list[int]]) -> None:
    if len(cur) == n:
        final.append(cur)
        return

    # Pushing the digit in front to force the number to start with the LSB (where B is bit in this case)
    generate_binary_nums(n, [0] + cur, final)
    generate_binary_nums(n, [1] + cur, final)

nums = []
generate_binary_nums(bin_len(9), [], nums)
nums = nums[0:10]

weights = []

# Basically going through the binary numbers and seeing which decimal number activates which bit, eg 
# 2 wouldn't activate the LSB but it would the second
for i in range(4):
    tmp = []
    for num in nums:
        tmp.append(num[i])

    weights.append(tmp)

neurons = [BinaryNeuron(weight, 0.5) for weight in weights]

number = [0] * 10
number[9] = 1

for i in range(4):
    print(neurons[i].calculate_activation(number))
    print("\n")
