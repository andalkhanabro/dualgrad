
'''a Value container for a scalar unit in a computational graph'''

class Value:

    def __init__(self, data, _op='', children=(), label = ''):
        self.data = data 
        self.grad = 0 # gradient wrt final object
        self._op = _op # which operation made me?
        self._prev = set(children) # which operands were used to make me?
        self.label = label # what is my variable name? useful for debugging/viz etc 

    # override the default print method for a Value obj (make it human readable)
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # overload the + operator, define what it means to add two Value objects 
    def __add__(self, other):
        out = self.data + other.data 
        val = Value(out)
        val._prev = (self, other)
        val._op = '+'
        return val
    
    # overload the * operator, define what it means to multiply two Value objects 
    def __mul__(self, other):
        out = self.data * other.data
        val = Value(out)
        val._prev = (self, other)
        val._op = '*'
        return val