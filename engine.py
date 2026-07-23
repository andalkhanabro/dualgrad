'''a Value container for a scalar unit in a computational graph'''

import math

class Value:

    def __init__(self, data, _op='', children=(), label = ''):

        self.data = data 
        self.grad = 0 # gradient wrt final object
        self._op = _op # which operation made me?
        self._prev = set(children) # which operands were used to make me?
        self.label = label # what is my variable name? useful for debugging/viz etc 

        # for backprop 

        self._backward = lambda: None # a callable entity

    # override the default print method for a Value obj (make it human readable)
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # overload the + operator, define what it means to add two Value objects 
    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other) # if adding raw numbers to values 

        out = self.data + other.data 
        val = Value(data=out, _op='+', children=(self, other))

        def _backward():
            # this already has access to self, other and out when self + other = out // in this operation 

            # atp, val.grad would be filled in by the backward pass 

            self.grad += 1 * val.grad
            other.grad += 1 * val.grad 

        val._backward = _backward

        return val 
    
    def __radd__(self, other):
        # to handle operations like <int/float> + Value() (in this order)
        return self + other
    
    def __rmul__(self, other):
        # to handle operations like <int/float> * Value() (in this order)
        return self * other
    
    # overload the * operator, define what it means to multiply two Value objects 
    def __mul__(self, other):
    
        other = other if isinstance(other, Value) else Value(other) # if multiplyinh raw numbers to values 
        out = self.data * other.data
        val = Value(data=out, _op='*', children=(self, other))

        def _backward():

            self.grad += other.data * val.grad 
            other.grad += self.data * val.grad 

        val._backward = _backward

        return val
    
    # a tanh transform for non-linearity stuff later (x to tanh(x))
    def tanh(self):

        # smth like x->tanh(x)

        val = Value(math.tanh(self.data), _op = 'tanh', children = (self,)) 

        def _backward():

            self.grad += (1 - (val.data)**2) * val.grad # d/dx(tanh(x)) =  1 - tanh(x)^2 

        val._backward = _backward

        return val
    
    # to be able to exponentiate inputs (i.e x -> e^x)
    def exp(self):

        val = Value(math.exp(self.data), _op = 'exp', children = (self,))

        def _backward():

            self.grad += val.data * val.grad# d/dx(e^x) = e^x

        val._backward = _backward

        return val
    
    def __pow__(self, other):
        # what does it mean to raise our obj to smth? so smth like:
        # Value(3)**2 would be parsed as:
        #Value.__pow__(Value(3), 2)

        # only implementing val*num rn, not actual variable exponents 

        assert isinstance(other, (int, float)) # checks if other has int type 

        val = Value((self.data)**other, _op = 'pow', children=(self,))

        def _backward():
            # propagate gradient from x^k to x 
            # d/dx(x^k) = kx^(k-1) [local derivative]

            self.grad += other * (self.data)**(other - 1) * val.grad # global derivative 

        val._backward = _backward # TODO: this 

        return val 
    
    def __truediv__(self, other):
        # what does it mean to divide a/b when a,b Value objects 

        # a / b === a * b**(-1)
        
        return self * (other)**(-1)
    
    def __neg__(self):

        # what does -x mean when we do -x for x = Value() obj?
        # i.e, what does it mean to negate a Value() object 

        return self * (-1)
    

    def __sub__(self, other):

        return self + (-other)
    
    def __rtruediv__(self, other):
        # for the reversed order, i.e if a/b fails for k/Val cases 

        # call would be int.__truediv__(k, Val) [not implemented]
        # so gets rerouted to Val.__rtruediv(Val, k) # reflected.. 

        return other * self**(-1)
    
    def __rsub__(self, other):

        # for cases like k - Val, k integer -> call: int.__sub__(k, Val)
        # not implemented, get rerouted: Val.__rsub__(Val, k) // need this 
        # this needs to return smth equivalent to k - Val in ORIGINAL call
        # and we'll get here by a reflected arg order, so Val = self, and k = other 
        
        return other + (-self) # easier way to write this, since __neg__ supported for Val()
    
    # NOTE: __r*__() is called only when for some a <op> b, the left operand does not know how
    # how to <op> between it and b. then the call becomes: type(b).__r*__(b, a) 
    # NOTE (better): for `a <op> b`, if a can't handle b, Python calls b.__r*__(a) — b is self, a is other.

    def backward(self):

        # now, toposort the graph from self to figure out dependencies. self has children.. and so on..

        visited = set()
        order = []

        def toposort(node):

            if node in visited:
                return 
            visited.add(node)

            for child in node._prev:
                toposort(child)
            
            order.append(node)
        
        toposort(self) # this makes the order. and now, for the actual order, reverse it 

        order.reverse() # in place

        seed = 1.0 

        self.grad = seed # this is how we're initializing it anyway 

        for n in order:
            n._backward()

        # a toposort with a post order guarantees that:
        # every node is appended after all its children in order[] (e.g in A->B and A->C), order = [BCA]
        # so in reversed(order), every child is appended AFTER its parent node (this is what we want) --> order = [ABC] or [ACB], both valid topological orders 
        # so every node.grad is fully calculated before we pass it to its children and propagate the gradient 