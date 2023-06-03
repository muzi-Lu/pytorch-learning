class MyClass:
    i = 12345

    def f(self):
        return 'hello world'


x = MyClass()
print("MyClass 类的属性i 为：", x.i)
print("MyClass 的f方法 为：", x.f())


class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart


x = Complex(3, 4.5)
print(x.r, x.i)


class Test:
    def prt(self):
        print(self)
        print(self.__class__)


t = Test()
t.prt()


class Test:
    def prt(runoob):
        print(runoob)
        print(runoob.__class__)


t = Test()
t.prt()


class people:
    name = ''
    age = 0
    __weight = 0

    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w

    def speak(self):
        print("%s 说： 我 %d 岁了。" % (self.name, self.age))


p = people('runoob', 10, 30)
p.speak()


class student(people):
    grade = ''

    def __init__(self, n, a, w, g):
        people.__init__(self, n, a, w)
        self.grade = g

    def speak(self):
        print("%s 说： 我 %d 岁了。我在 %d 年级" % (self.name, self.age, self.grade))


s = student('ken', 10, 60, 3)
s.speak()


class speaker():
    topic = ''
    name = ''

    def __init__(self, n, t):
        self.name = n
        self.topic = t

    def speak(self):
        print("我是 %s, 我是一个演说家， 我演讲的主题是 %s" %(self.name, self.topic))

class sample(speaker, student):
    a = ''
    def __init__(self, n, a, w, g, t):
        student.__init__(self, n, a, w, g)
        speaker.__init__(self, n, t)
