# Python3 Input and Output

s = 'Hello, World'
string = str(s)
repr_string = repr(s)
print(string)
print(repr_string)

x = 10 * 3.25
y = 200 * 200
s = 'x value is:'+ repr(x) +', y value is:' + repr(y)
print(s)

'''
Table of Squares and cubes
'''

for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))

a = '12'.zfill(5)
print(a)

b = '-0.04'.zfill(8)
print(b)

print('{} address: "www.runoob.com"'.format('菜鸟教程', 'www.runoob.com'))

print('{0} 和 {1}'.format('Google', 'Runoob'))
print('{1} 和 {0}'.format('Google', 'Runoob'))

print('{name}网址： {site}'.format(name='菜鸟教程', site='www.runoob.com'))

import math
print('常量PI的值近似为：{}'.format(math.pi))
print('常量PI的值近似为：{!r}'.format(math.pi))
print('常量PI的值近似为：{0:.3f}'.format(math.pi))



# keyboard

# str = input('please input content:')
print('what you input is:', str)

f = open("File/wt_practice.txt", "w")
f.write("Day3_2 and Day4 Practice.\n" + "I really like Python.\n")
f.close()

f = open("File/wt_practice.txt", "r")
str = f.read()
print(str)
f.close()

f = open("File/wt_practice.txt", "r")
str1 = f.readline()
str2 = f.readline()
print(str1)
print(str2)
f.close()

f = open("File/wt_practice.txt", "r")
str = f.readlines()
print(str)
f.close()

f = open("File/wt_practice.txt", "r")
for line in f:
    print(line, end='')
f.close()

'''
overwrite the above write
'''
f = open("File/wt_practice.txt", "w")
num = f.write("Python is a great language.\n Yes, it is.\n")
print(num)
f.close()
print(f.closed)

'''
Problem1: do not work
'''
# f = open("File/wt_practice1.txt", "w")
# value = ('www.runoob.com', 14)
# s = str(value)
# f.write(s)
# f.close()

'''
用with读完后他会关闭文件
'''
with open("File/wt_practice.txt", "r") as f:
    read_data = f.read()
print(f.closed)

# Pickle Module

import pickle
data1 = {'a':[1, 2.0, 3, 4+6j],
         'b':('string', u'Unicode string'),
         'c':None
         }

selfref_list = [1, 2, 3]
print(selfref_list)
selfref_list.append(selfref_list)
print(selfref_list) # 为什么输出是[1, 2, 3, [...]]

output = open('File/data.pkl', 'wb')
# Pickle dictionary using protocol 0.
pickle.dump(data1, output)

# Pickle the list using the highest protocol available.
pickle.dump(selfref_list, output, -1)
output.close()

import pprint, pickle

pkl_file = open('File/data.pkl', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

data2 = pickle.load(pkl_file)
pprint.pprint(data2)

pkl_file.close()
