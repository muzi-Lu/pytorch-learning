import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

t_out = torch.mean(tensor*tensor)
var_out = torch.mean(variable*variable)
var_out_test = torch.mean(variable.data*variable.data)  # variable.data --> t.out

print(t_out)
print(var_out)
print(var_out_test)
print(variable.grad)

var_out.backward()
print(variable.grad)
'''
v_out = 1/4 *sum(var*var)
d(var_out)/d(var) = 1/4 * 2 *variable = vaiable / 2
'''

print(variable.data)  #  variable --> tensor
print(variable.data.numpy()) # variable --> tensor --> numpy