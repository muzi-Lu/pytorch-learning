import os.path as osp
import os
import numpy as np

##### 经典工程语句测试 #####
# test = [n for n in range(100) if n % 2 == 0]
# print(test)

##### 测试语句 #####
seq_dir = '/media/benben/0ECABB60A248B50C/Ambiguous_ReLoc_Dataset/meeting_table/train/seq00'
filenames = [n for n in os.listdir(osp.join(seq_dir, 'rgb_matched')) if
                           n.find('frame') >= 0]

print(filenames)
filenames = np.sort(filenames)
print(filenames)