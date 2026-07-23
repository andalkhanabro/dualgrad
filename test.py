'''a test training loop written using micrograd for a basic MLP'''

# toy values; simulated data 
x = [[1,2.0, 2.2], 
     [0, 0.5, 1], 
     [1, 1, 1], 
     [1, 2, 4]]

gs = [1, 1, -1, -1]

epochs = 500 # how many iterations the NN should run for 

from nn import MLP 

model = MLP(3, [3, 3,1]) # i.e a NN with 3 layers, first layer with 3 neurons, a hidden layer with 3 neurons and 1 output layer
                        # with a single neuron 

for i in range(epochs):

    # model starts out with random weights 

    for p in model.parameters():
        p.grad = 0 # reset the gradients before computing the next 
                   # loss backward

    # compute the actual predictions 

    y_pred = []
    for row in x:
        y_pred.append(model(row))

    loss = sum([(yhat - y)**2 for yhat,y in zip(y_pred, gs)])

    loss.backward()
    # print(f"epoch: {i}, loss: {loss.data}")

    # gradient descent 

    lr = 0.01 # hyperparameter

    for p in model.parameters():
        p.data -= lr * p.grad

# final model predictions 
print("")
for r in x:
    print(model(r))

# last run predictions: 

# Value(data=0.9055016749633923) ~ 1 
# Value(data=0.8978661870901237) ~ 1
# Value(data=-0.8696075192081286) ~ -1 
# Value(data=-0.9269783806417855) ~ -1

'''minor testing for forward mode auto-diff'''

from f_engine import DualValue

x = DualValue(2, 1) 
y = DualValue(3, 0)

u = x+y + x*y
print(f"u(x=2, y=3): {u.val}, u'(x=2, y=3): {u.dual}") # 

a = DualValue(4, 1)
print(2 / a)          # want <0.5, -0.125>
b = DualValue(1, 1)
print(b.exp())        # want <2.718282, 2.718282>










