# # implicit type conversion
num_int = 123
num_flo = 1.23

num_new = num_flo + num_int

print("datatype of num_int:", type(num_int))
print("datatype of num_flo:", type(num_flo))

print("Value of num_new:", num_new)
print("datatype of num_new:", type(num_new))

# # explicit type conversion

x = int(1)
y = int(2.8)
z = int("3")

print("datatype of num_int:", type(x))
print("datatype of num_flo:", type(y))
print("datatype of num_flo:", type(z))

x = float(1)
y = float(2.8)
z = float("3")

print("datatype of num_int:", type(x))
print("datatype of num_flo:", type(y))
print("datatype of num_flo:", type(z))

x = str(1)
y = str(2.8)
z = str("3")

print("datatype of num_int:", type(x))
print("datatype of num_flo:", type(y))
print("datatype of num_flo:", type(z))

# operator

# # arithmetic operator
a = 21
b = 10
c = 0

# # comparison operator

# # assignment operator

# # bitwise operator

# # Logical operator

# # member operator

# # identify operator

# # operator precedence


# Conditional statements

flag = False
name = 'luren'
if name == 'python':
    flag = True
    print('welcome')
else:
    print('wrong person')

num = 10
if num == 2:
    print('2 is yes')
elif num == 1:
    print('1 is yes')
elif num == 0:
    print('0 is no')
else:
    print('who are you')

num = 9
if num >= 0 and num <= 10:
    print('hello')

num = 10
if num < 0 or num > 10:
    print('hello')
else:
    print('bye')

'''
当if有多个条件时可使用括号来区分判断先后顺序，括号中的判断优先执行，此外and和or的优先级低于>, <等判断符号，也就是说>,<在没有括号的情况下要优先判断。
'''

# loop statement
# # while loop statement

a = 1
while a < 10:
    print(a)
    a += 2

# number = [12, 23, 53, 48, 79, 54]
# even = []
# odd = []
# while len(number) > 0:
#     number = number.pop()
#     if(number % 2 == 0):
#         even.append(number)
#     else:
#         odd.append(number)
'''
TypeError: object of type 'int' has no len()
'''

count = 0
while (count < 9):
    print('The count is:', count)
    count += 1
print('Finished')

# # # continue and break
i = 1
while i < 10:
    i += 1
    if i % 2 > 0:
        continue
    print(i)

i = 1
while 1:
    print(i)
    i += 1
    if i > 10:
        break

# # for loop statement
for letter in 'Python':
    print('当前字母：%s' % letter)

fruits = ['banana', 'apple', 'mango']
a = len(fruits)
print(a)
for fruit in fruits:
    print('当前水果：%s' % fruit)

for index in range(a):
    print('当前水果：%s' % fruits[index])


# loop nesting

# # pass
class MyEmptyClass:
    pass


# Fibonacci series

a, b = 0, 1
while b < 1000:
    print(b)
    a, b = b, a + b

# # list comprehension
names = ['Bob', 'Tom', 'alice', 'Jerry', 'Wendy', 'Smith']
new_names = [name.upper() for name in names if len(name) > 3]
print(new_names)

# # dict comprehension
listdemo = ['Google', 'Runoob', 'Taobao']
newdict = {key: len(key) for key in listdemo}
print(newdict)
print(type(newdict))

# # set comprehension
a = {x for x in '1234567890' if x not in '1369'}
print(type(a))

# # tuple comprehension
a = (x for x in range(1, 10, 2))
print(a)
print(tuple(a))

# # iterator
list = [1, 2, 3, 4]
it = iter(list)
print(it)
print(next(it))
print(next(it))
print("--------------------------------")
for x in it:
    print(x, end=" ")

import sys

# list = [1, 2, 3, 4]
# it = iter(list)
# while True:
#     try:
#         print(next(it))
#     except StopIteration:
#         sys.exit()


# # create an iterator
'''
Do not Work
Solution: sys.exit()从这里直接结束
'''
class MyNumbers:
    def __init__(self):
        self.a = 1

    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return x


myclass = MyNumbers()
myiter = iter(myclass)
print("--------------------------------")
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))

# # Define Function
def hello():
    print('Hello world!')

hello()

def area(width, height):
    return width * height

w = 4
h = 5
print("width = ", w, "height =", h, "area = ", area(w, h))


def printme(str):
    print(str)
    return

printme("Runoob")

# Parameter passing
# # mutable and immutable class
'''
immutable class: strings,tuples,numbers
mutable class: list, dict

Parameter passing of python functions:
Immutable types: C++ like value passing such as integers, strings and tuples
Mutable types: C++ like passing by reference
'''

def change(a):
    print(id(a))
    a = 10
    print(id(a))

a = 1
print(id(a))
change(a)
print(id(a))
change(a)
print(a)

# Parameter
# # required parameter
# # keyword parameter
# # default parameter
# # variable length parameter

def printme(str):
    print(str)
    return

# printme() # 无参数
printme('Runoob')


# # keyword parameter
def printme(str):
    print(str)
    return

printme(str = 'Runoob')

# # default parameter
def printinfo(name, age=35):
    print('名字：', name)
    print('年龄：', age)

printinfo(age=50, name='runoob')
printinfo(name='runoob')

# # variable length parameter

'''
加了星号 * 的参数会以元祖形式导入，存放所有未命名的变量参数
'''

def printinfo(arg, *vartuple):
    print("输出：")
    print(arg)
    print(vartuple)

printinfo( 70, 60, 50)

def printinfo(arg, *vartuple):
    print("输出：")
    print(arg)
    for var in vartuple:
        print(var)
    return

printinfo( 70, 60, 50)

'''
加了两个星号 ** 的参数会以字典形式导入
'''

def printinfo(arg, **vardict):
    print('输出：')
    print(arg)
    print(vardict)

printinfo(1, a=2, b=3)

list = [1,2,3,4,5]
a,b,*c =list
print('a=',a)
print('b=',b)
print('c=',c)

def function(*args,**kwargs):
    print(args)
    print(kwargs)
    print(args[0])
    print(kwargs['b'])
my_list = [1,2,3,'a','b']
my_dict = {'a':1,'b':0,'c':2}
function(*my_list,**my_dict)

'''
这里有个问题啊，就是我 * 和 ** 一起出现，那他的传参应该怎么进行呢？


'''