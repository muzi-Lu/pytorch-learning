# Python3 data structrues

# List

# 方法	描述
# list.append(x)	把一个元素添加到列表的结尾，相当于 a[len(a):] = [x]。
# list.extend(L)	通过添加指定列表的所有元素来扩充列表，相当于 a[len(a):] = L。
# list.insert(i, x)	在指定位置插入一个元素。第一个参数是准备插入到其前面的那个元素的索引，例如 a.insert(0, x) 会插入到整个列表之前，而 a.insert(len(a), x) 相当于 a.append(x) 。
# list.remove(x)	删除列表中值为 x 的第一个元素。如果没有这样的元素，就会返回一个错误。
# list.pop([i])	从列表的指定位置移除元素，并将其返回。如果没有指定索引，a.pop()返回最后一个元素。元素随即从列表中被移除。（方法中 i 两边的方括号表示这个参数是可选的，而不是要求你输入一对方括号，你会经常在 Python 库参考手册中遇到这样的标记。）
# list.clear()	移除列表中的所有项，等于del a[:]。
# list.index(x)	返回列表中第一个值为 x 的元素的索引。如果没有匹配的元素就会返回一个错误。
# list.count(x)	返回 x 在列表中出现的次数。
# list.sort()	对列表中的元素进行排序。
# list.reverse()	倒排列表中的元素。
# list.copy()	返回列表的浅复制，等于a[:]。

a = [66.25, 333, 333, 1, 1234.5]
print(a.count(333), a.count(66.25), a.count('x'))
print(a)
a.insert(2, -1)
print(a)
a.append(333)
print(a)
print(a.index(333))
c = a.reverse()
print(c)

# Use a list as a stack
stack = [3, 4, 5]
stack.append(6)
stack.append(7)
print(stack)
c = stack.pop()
print(c)

# Use a list as a list

from collections import deque
queue = deque(['Eric', 'John', 'Michael'])
print(queue)
queue.append('Terry')
queue.append('Gramham')
print(queue)
queue.popleft()
queue.popleft()
print(queue)

# list comprehension

vec = [2, 4, 6]
vec_changed = [3*x for x in vec]
print(vec_changed)

# nested list comprehension
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

matrix_changed = [[row[i] for row in matrix] for i in range(4)]
print(matrix_changed)
print("-------------------------------------------------------")
transposed = []
for i in range(4):
    print([row[i] for row in matrix])
    transposed.append([row[i] for row in matrix])
    print(transposed)
print("-------------------------------------------------------")

# del
a = [-1, 1, 66.25, 333, 312, 1234.5]
print(a)
del(a[0])
print(a)
del(a[2:4])
print(a)

print("-------------------------------------------------------")

t = (12345, 54321, 'hello')
u = t ,(1, 2, 3, 4, 5)
print(u)

basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)
print('orange' in basket)
print('watermelon' in basket)

# Module
import sys
import Module.support as support
# from ... import
# from ... import *

print('命令行参数如下：')
for i in sys.argv:
    print(i)
print('\n\nPython 路径为：',sys.path, '\n')

print("-------------------------------------------------------")
support.print_func('world')

print("-------------------------------------------------------")
# dir func
print(dir(sys))
print(dir(support))
# print(dir(Day1))

# Standard Module
'''
do not work
'''
# print(sys.ps1)
# print(sys.ps2)