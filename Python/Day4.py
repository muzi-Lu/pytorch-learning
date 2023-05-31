import os, sys

# 1.os.access() 方法使用当前的uid/gid尝试访问路径。大部分操作使用有效的 uid/gid, 因此运行环境可以在 suid/sgid 环境尝试。

'''
path -- 要用来检测是否有访问权限的路径。

mode -- mode为F_OK，测试存在的路径，或者它可以是包含R_OK, W_OK和X_OK或者R_OK, W_OK和X_OK其中之一或者更多。

os.F_OK: 作为access()的mode参数，测试path是否存在。
os.R_OK: 包含在access()的mode参数中 ， 测试path是否可读。
os.W_OK 包含在access()的mode参数中 ， 测试path是否可写。
os.X_OK 包含在access()的mode参数中 ，测试path是否可执行。
'''

ret = os.access("File/wt_practice.txt", os.F_OK)
print("F_OK - 返回值 %s" % ret)

ret = os.access("File/wt_practice.txt", os.F_OK)
print("R_OK - 返回值 %s" % ret)

ret = os.access("File/wt_practice.txt", os.W_OK)
print("W_OK - 返回值 %s" % ret)

ret = os.access("File/wt_practice.txt", os.X_OK)
print("X_OK - 返回值 %s" % ret)

# 2.os.chdir() 方法用于改变当前工作目录到指定的路径。

retval = os.getcwd()
print("当前工作目录为 %s" % retval)

path = '/home/benben/code/pytorch-learning/MFStudy'

os.chdir(path)

retval = os.getcwd()

print("目录修改成功： %s" % retval)