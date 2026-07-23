'''a unit of computation for forward-mode autodiff. 
<a,b> represents a dual number, where a is the value, b is the derivative wrt some input variable'''

import math

class DualValue:

    # makes a dual value <a,b> with value a, and derivative b
    # .dual = 0 by default to allow constant Dual Values <c, 0>
    def __init__(self, val, dual=0):

        self.val = val 
        self.dual = dual 

    # how to add two dual value objects 
    def __add__(self, other):

        # then values add additively, and derivative of a sum (u(x) + v(x)) = u'(x) + v'(x)

        # if other is a constant c, then <c,0> is its DualValue representation 
        other = DualValue(other, 0) if not isinstance(other, DualValue) else other

        return DualValue(self.val + other.val, self.dual + other.dual)

    def __radd__(self, other):

        return self + other # __add__() already handles this case after the reflected arguments 
        # say we do something like 2 + DV(), then it would try: int.__(2, DV) -> Value.__radd__(DV, 2), so self = DV 

    # how to multiply two dual value objects
    def __mul__(self, other):

        # if other is a constant c, then <c,0> is its DualValue representation 
        other = DualValue(other, 0) if not isinstance(other, DualValue) else other

         # if u(x), v(x), then value is simply the product u(x)v(x), but derivative of the product is u'(x)v(x) + v'(x)u(x) 
        return DualValue(self.val * other.val, self.val * other.dual + other.val * self.dual)

    # same logic as implemented in __radd__(); asymmetry is handled directly 
    def __rmul__(self, other):
        return self * other 

    def tanh(self):

        # say we have some <x, 1> 
        # now what? 

        # CRUX OF UNDERSTANDING 

        # so if we do: f(<x, 1>), we're saying we had some x, with x' = 1, since x is the variable of differentiation 
        # so new val is just f(x.val) = f(x) but derivative is POTENTIALLY a composition. say we instead had:
        # <u(x), u'(x)> for some value of x, since none of this is symbolic, so: 
        # arbitary f would make it:
        # f(<u(x), u'(x)>) = f(u(x)) = f(u.val) AND (f(u(x)))' = f'(u(x)) * u'(x), and u(x) = u.val and u'(x) = u.dual, so
        # for arbitary <a,b>, we get: f(<a,b>) = <f(a), f'(a) * b> where form of f'() depends on the transform f being applied

        tanh_val = math.tanh(self.val)
        # logic: tanh(<u(x), u'(x)>) = <tanh(u(x)), (1- tanh(u(x))^2) * u'(x)> 
        return DualValue(tanh_val, (1 - (tanh_val)**2) * self.dual)

    # to handle -DV() directly
    def __neg__(self):
        return -1 * self

    def __sub__(self, other):

        other = DualValue(other, 0) if not isinstance(other, DualValue) else other
        return DualValue(self.val - other.val, self.dual - other.dual)

    def __rsub__(self, other):
        return -1 * (self - other)

    def __pow__(self, other):
        # other for now, is a float or an int #TODO: support exponentiation of DV objects
        assert isinstance(other, (float, int)) 
        # logic: <u(x), u'(x)>^k = <u(x)^k, k(u(x))^(k-1) * u'(x)> 
        return DualValue((self.val)**other, other * (self.val)**(other-1) * self.dual)

    def __truediv__(self, other):

        # to handle cases like DV/k, k constant 

        other = DualValue(other, 0) if not isinstance(other, DualValue) else other
        return self * (other)**(-1)

    def __rtruediv__(self, other):
        return (self)**(-1) * other # due to arg flip 

    def exp(self):
        exp_val = math.exp(self.val)
        return DualValue(exp_val, exp_val * self.dual)

    def __repr__(self):
        return f"value: {self.val}, derivative: {self.dual}"

# # testing 

# x = DualValue(2, 1) # say this is x 
# y = DualValue(3, 0) # this is y 

# u = x+y + x*y
# print(f"u(x=2, y=3): {u.val}, u'(x=2, y=3): {u.dual}") # 

x = DualValue(4, 1)
print(2 / x)          # want <0.5, -0.125>
y = DualValue(1, 1)
print(y.exp())        # want <2.718282, 2.718282>
print(DualValue(2, 1))  # should print readably, not a memory address

    


    

