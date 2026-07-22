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


# a layer of neurons stacked atop eachother, sharing an input (from the previous layer, or raw input to NN)

class Layer:

    def __init__(self, nin, nout):

        self.neurons = [Neuron(nin) for _ in range(nout)] # make nout neurons of nin weights per neuron

    def __call__(self, x):

        # pass x through each neuron, and output a list of n(x) where n(x) is output from each neuron 

        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    # a flattened list of parameters for this layer 

    def parameters(self) -> list[Value]:

        params = [p for n in self.neurons for p in n.parameters()]
        return params


class MLP:

    # a multi-layer perceptron; feed forward neural network of fully connected hidden layers 

    def __init__(self, nin, nouts): 

        sizes = [nin] + nouts # 
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]


    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):

        params = [p for l in self.layers for p in l.parameters()]
        return params 

    



               

    



    










