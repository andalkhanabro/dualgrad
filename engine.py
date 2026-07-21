'''a Value container for a scalar unit in a computational graph'''

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


        
    
            


