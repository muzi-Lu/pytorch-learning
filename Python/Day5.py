# while True:
#     try:
#         x = int(input("请输入一个数字："))
#         break
#     except ValueError:
#         print("您输入的不是一个数字，请再次尝试输入")

import sys

def runoob():
    pass

try:
    f = open('File/myfile.txt', 'r')
    s = f.readline()
    f.close()
    print(f.closed)
    i = int(s.strip())

except OSError as err:
    print("OS error : {0}".format(err))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected Error:", sys.exc_info()[0])
    raise

'''
try 语句按照如下方式工作：
    首先，执行try子句（在关键字try 和 关键字except 之间的语句）。
    如果没有异常发生， 忽略except子句，try子句执行后结束。
    如果在执行try子句的过程中发生了异常，那么try子句余下的部分都会被忽略。如果异常的类型和except之后的名称相符，那么执行except子句部分
    如果一个异常没有与之匹配的except，那么这个异常会传递给try
'''
try:
    pass
except(RuntimeError, TypeError, NameError):
    pass

'''
try/except...else
try/except语句中还有一个可选的else子句，如果使用这个子句，那么必须放在所有的except子句之后
'''

for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except IOError:
        print('can not open', arg)
    else:
        print(arg, 'has', len(f.readlines()), 'lines')
        f.close()

try:
    runoob()
except AssertionError as error:
    print(error)
else:
    try:
        with open('File/myfile.log') as file:
            read_data = file.read()
            file.close()
    except FileNotFoundError as fnf_error:
        print(fnf_error) # 这句话为什么不打印
finally:
    print("这句话，无论什么时候都会发生。")

x = 10
if x > 5:
    raise Exception('x 不能大于 5。x的值为：{}'.format(x))

class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

try:
    raise MyError(2 * 2)
except MyError as e:
    print("My Exception occurred, value:", e.value)