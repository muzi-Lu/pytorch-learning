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
while(count < 9):
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