import numpy as np

print(list(map(lambda x:x+3, [11, 12, 13, 14]))) # 学一下Map， 复习lambda表达式

a = np.array([11, 12, 13, 14])
print(a+3)
print(a-3)

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([
    [1, 2],
    [3, 4]
])

print(a.dot(b))
print(np.dot(a, b))

a = np.array([11, 12, 13, 14])
print(a.max().dtype)
print(np.min(a))

print(a.mean())
print(np.median(a))
print(np.std(a))

print(np.argmax(a))
print(np.argmin(a))

a = np.array([11.1, 12.2, 13.3, 14.4])
print(np.ceil(a))
print(np.floor(a))