import numpy as np

# cars = np.array([1, 2, 3, 4, 5, 6, 7])
# print("cars: ", cars, "\ndimension: ", cars.ndim, "\nshapes:", cars.shape)
# print("data address:", cars.data, "\ndata:", cars.data[1])

cars = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
# print("cars: ", cars, "\ndimension: ", cars.ndim, "\nshapes:", cars.shape)
# # print("data address:", cars.data, "\ndata:", cars.data[1][2]) wrong
# print("data address:", cars.data, "\ndata:", cars.data[1, 2])
# print("data address:", cars.data, "\ndata:", cars.data[2, 1])

cars = np.array([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [10, 21, 32, 43],
        [54, 65, 76, 87],
        [98, 109, 1110, 1211]
    ]
])
# print("cars: ", cars, "\ndimension: ", cars.ndim, "\nshapes:", cars.shape)
# # print("data address:", cars.data, "\ndata:", cars.data[1][2]) wrong
# print("data address:", cars.data, "\ndata:", cars.data[1, 2, 3])

################################################
#                                              #
#                 合并数据                      #
#                                              #
################################################

test1 = np.array([5, 10, 15, 20])
test2 = np.array([2.2, 4.1, 6.3, 8.4])

test = np.concatenate([test1, test2])
print(test)

test1 = np.expand_dims(test1, 0)
# test1 = np.expand_dims(test1, 2) #这是对第三维度拓展了
test2 = test2[np.newaxis, :]
print("test1 after epand the dimension:", test1)
print("test1 after epand the dimension:", test2)

test_2d = np.concatenate([test1, test2])
print(test_2d)

test_2d = np.concatenate([test1, test2], axis=0)
print(test_2d)

test_2d = np.concatenate([test1, test2], axis=1)
print(test_2d)

# a_test = ([
#     [1, 2, 3],
#     [2, 3, 4],
#     [4, 5, 6]
# ])

a_test = ([
    [1, 2, 3],
    [2, 3, 4]
])

b_test = ([
    [7, 8],
    [9, 10]
])
print(np.concatenate([a_test, b_test], axis=1))
# print(np.concatenate([a_test, b_test], axis=0)) # wrong 轴对不上，想要对上维度要对

a_test = ([
    [1, 2, 3],
    [2, 3, 4]
])

b_test = ([
    [7, 8, 9],
    [9, 10, 11]
])

print(np.concatenate([a_test, b_test], axis=0))

# 二维数据上好用的还有np.vstack(), np.hstack()

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([
    [3, 4],
    [6, 7]
])

print('垂直合并: \n', np.vstack([a, b]))
print('垂直合并: \n', np.hstack([a, b]))

test = np.array([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [10, 21, 32, 43],
        [54, 65, 76, 87],
        [98, 109, 1110, 1211]
    ]
])

print("总共数据:", test.size)
print("1d data:", test[0].size)
print("1d data:", test[0, 1].size)

print("1d data:", test.shape[0])
print("1d data:", test.shape[1])

################################################
#                                              #
#                 选择数据                      #
#                                              #
################################################
a = np.array([1, 2, 3])
print("a[0]:", a[0])
print("a[1]:", a[1])

print("a[[0,1]]:", a[[0, 1]])
print("a[[2,2,1]]:", a[[2, 2, 1]])

b = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])





