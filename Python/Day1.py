# Lesson 1 Print
# Print Hello World

print("Hello World")

# Lesson 2 Basic Program
# Encode - (UTF-8) --unicode string
# 当然也可以在源文件中使用Windows-1252
# -*- coding: cp-1252 -*-

# Python Keyword
import keyword
print(keyword.kwlist)

# Make a Comment

# four type of number
a = 1
b = True
c = 1.23
d = 1+ 2j

# String
# '' equal to ""
# ''' and """ is used by a string that have many rows
# \ is Escapes and r can let \ do not occur
# + is connection, * is repeat
# Python have two index method, one is starts with 0 from left to right and -1 from right to left
# String can not be changed
# String Variable [head subscipt: tail subscript: step size]

str = 'string'
sentence = "This is a sentence"
paragraph = """This is a paragraph
and it contain two rows"""


str = '123456789'
print(str)
print(str[0:-1])
print(str[0])
print(str[2:5])
print(str[2:])
print(str[1:5:2])
print(str * 2)
print(str + '0')

print("-------------------------------")

print('hello\nrunoob')
print(r'hello\nrunoob')

# input("\n\n press enter to esc")

import sys; x = 'runoob'; sys.stdout.write(x + '\n')

x = 'a'
y = 'b'
# 不换行输出
print(x, end=" ")
print(y, end="")


import sys
print("+++++++++++++++++++Python import mode++++++++++++++++++++")
print("command line parameters")
for i in sys.argv:
    print(i)
print('\n python path is', sys.path)

# Python3 basic data type
count = 100
miles = 1000.0
name = 'runoob'
print(count)
print(miles)
print(name)

# variables are assigned the same value
a = b = c = 1
# multiple objects specify multiple variables
a, b, c = 1, 2, 'rum'

# standard data type
# # Number
# # String
# # Bool
# # List
# # Tuple
# # Set
# # Dictionary

# # # Immutable data(3): Number(number), String(string), Tuple(tuple)
# # # Variable data(3): List(list), Dictionary(dict), Set(collection)

a, b, c, d = 20, 5.5, True, 4+3j
print(type(a), type(b), type(c), type(d))
print(isinstance(a, int))

# isinstance and type
# type()不会认为子类是父类类型
# isinstance()

print('''_____________________________________________''')

class A:
    pass

class B(A):
    pass

print(isinstance(A(), A))

print(type(A()) == A)

print(isinstance(B(), A))

print(type(B()) == A)

# bool is subclass of int

var1 = 1
var2 = 10
print(var1)
del var1, var2
# print(var1)
# print(var2)

# Bumerical operations
print(5 + 4)
print(2.5 - 1.2)
print(3 * 7)
print(2 / 4)
print(3 // 5)
print(17 % 3)
print(2 ** 4)

# String
str = "python"
print(str[0], str[5])
print(str[-1], str[-6])

print("-------------")
# Bool
a = True
b = False
c = 321
print(a and b)
print(a or b)
print(not a)

# # type conversion
print(int(a))
print(float(b))
# print(str(a)) do not work
# print(str(c))

'''
Note: In python, all non-zero numbers and non-empty strings, listss, tuples, and other data types are considered True,
and only 0, empty strings, empty lists, empty tuples,etc are considered False. Therefore, when we performing Boolean type
conversion , you need to pat attention to the authenticity of the data type.
'''

# List
list = ['abcd', 765, 2.23, 'run', '70.2']
tinylist = ['123', 2222]

print(list)
print(list[0])
print(list[1:3])
print(list[2:])
print(tinylist * 2)
print(list + tinylist)
# # List have many built-in methods, such as append(), pop(), which will be discussed later

# # reverseWords
input = "I really like Python"
print(input)
dealed_input = input.split(' ')
print(dealed_input)

reversedWord = dealed_input[-1::-1]
print(reversedWord)
output = ' '.join(reversedWord)
print(output)

# Tuple
list = ('abcd', 765, 2.23, 'run', '70.2')
tinylist = ('123', 2222)

print(list)
print(list[0])
print(list[1:3])
print(list[2:])
print(tinylist * 2)
print(list + tinylist)

# Set
sites = {'Google', 'TaoBao', 'Zhihu', 'Baidu', "Me"}
sites2 = {'Google', 'TaoBao', 'Zhihu', 'ByteDance'}

print(sites)
print(sites | sites2)
print(sites - sites2)
print(sites & sites2)
print(sites ^ sites2)

dict = {}
dict[1] = "really really"
dict[2] = 'so so'

tinydict = {'name':'winner', 'number':4}
print(dict[1])
print(tinydict)
print(tinydict.keys())
print(tinydict.values())

# bytes
x = bytes('hello world', encoding = "utf-8")
print(x)