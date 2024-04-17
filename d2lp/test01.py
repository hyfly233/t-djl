import os
import pandas as pd
import torch

# 2.1.1
print("2.1.1 --------------------")
x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

X = x.reshape(3, 4)
print(X)

print(torch.zeros((2, 3, 4)))

print(torch.ones((2, 3, 4)))

print(torch.randn(3, 4))

print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# 2.1.2
print("2.1.2 --------------------")

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y

print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

print(X == Y)

print(X.sum())

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)

print(X[-1], X[1:3])

X[1, 2] = 9
print(X)

X[0:2, :] = 12
print(X)

before = id(Y)
Y = Y + X
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
print(id(X) == before)

A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

# 2.2.1
print("2.2.1 --------------------")

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

# 2.2.2
print("2.2.2 --------------------")

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 2.2.3
print("2.2.3 --------------------")

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y)

# 2.3.1
print("2.3.1 --------------------")

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x ** y

# 2.3.2
print("2.3.2 --------------------")

x = torch.arange(4.0)
print(x)
print(x[3])
print(len(x))
print(x.shape)

# 2.3.3
print("2.3.3 --------------------")

A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

# 2.3.4
print("2.3.4 --------------------")
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 2.3.5
print("2.3.5 --------------------")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A, A + B, A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)

# 2.3.6
print("2.3.6 --------------------")
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
print(A.shape, A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)

print(A.sum(axis=[0, 1]))

print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A / sum_A)
print(A.cumsum(axis=0))

# 2.3.7
print("2.3.7 --------------------")

y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x * y))

# 2.3.8
print("2.3.8 --------------------")
print(A.shape, x.shape, torch.mv(A, x))

# 2.3.9
print("2.3.9 --------------------")

