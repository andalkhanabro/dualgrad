from engine import Value 
import random

# a unit to abstract out a single neuron's computation 

class Neuron:

    # make a neuron that has n weights (inputs, a bias b)
    def __init__(self, nin):

        self.w = [Value(random.uniform(-1,1)) for i in range(nin)] # list comp, initialize weights randomly 
        self.b = Value(random.uniform(-1,1))

    # allows neuron n to be callable like n(x), for an x input of dimension (1 x nin)
    def __call__(self, x):

        activation = sum([w_i * x_i for w_i, x_i in zip(self.w, x)], self.b).tanh()
        return activation

    # return a flattened list of all weights w_1 .. w_n for this neuron, and the bias. x excluded 
    def parameters(self) -> list[Value]:

        return self.w + [self.b] # just returns the tunable parameters for this neuron 

    


    

    










