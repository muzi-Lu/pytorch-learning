import numpy as np

################################################
#                                              #
#                 改变数据形态                   #
#                                              #
################################################

a = np.array([1, 2, 3, 4, 5, 6])
a_2d = a[np.newaxis, :]
print(a.shape, a_2d.shape)

a = np.array([
    [1, 2],
    [3, 4]
])
a_3d = a[np.newaxis, :, :]
a_3d_1 = a[:, np.newaxis, :]
a_3d_2 = a[:, :, np.newaxis]
print(a.shape, a_3d.shape, a_3d_1.shape, a_3d_2.shape)

a_none = a[:, None, :]
a_expand = np.expand_dims(a, axis=1)
print(a_none.shape, a_expand.shape)

a_squeeze = np.squeeze(a_expand)
a_squeeze_axis = a_expand.squeeze(axis=1)
print(a_squeeze.shape)
print(a_squeeze_axis.shape)
print(a_expand.shape)

a = np.array([1, 2, 3, 4, 5, 6])
a1 = a.reshape([2, 3])
a2 = a.reshape([3, 1, 2])
a3 = a.reshape([3, 2, 1])
print(a1.shape)
print(a1)
print(a2.shape)
print(a2)
print(a3.shape)
print(a3)
print(a2[0], a2[1], a2[2])
print(a2[0, 0, 1])

a = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3])
aT1 = a.T
aT2 = np.transpose(a)
print(aT1)
print(aT2)

################################################
#                                              #
#                 合并数据形态                   #
#                                              #
################################################

featrue_a = np.array([1, 2, 3])
featrue_b = np.array([2, 3, 4])
c_stack = np.column_stack([featrue_a, featrue_b])
print(c_stack)

c_stack = np.row_stack([featrue_a, featrue_b])
print(c_stack)

##### column_stack and row_stack VS vstack and hstack #####

featrue_a = np.array([1, 2, 3])[:, None]
featrue_b = np.array([2, 3, 4])[:, None]
c_stack = np.hstack([featrue_a, featrue_b])
print(c_stack)

featrue_a = np.array([1, 2, 3])[None, :]
featrue_b = np.array([2, 3, 4])[None, :]
c_stack = np.vstack([featrue_a, featrue_b])
print(c_stack)

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([
    [5, 6],
    [7, 8]
])

print(np.concatenate([a, b], axis=0))
print(np.concatenate([a, b], axis=1))

a = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [2, 3, 4, 5],
    [2, 3, 4, 5]
])

print(np.vsplit(a, indices_or_sections=2))
print(np.vsplit(a, indices_or_sections=[2, 3]))

print("hspilt:", np.hsplit(a, indices_or_sections=2))
print(np.hsplit(a, indices_or_sections=[2, 3]))

print(np.split(a, indices_or_sections=2, axis=0))
print(np.split(a, indices_or_sections=[2, 3], axis=1))