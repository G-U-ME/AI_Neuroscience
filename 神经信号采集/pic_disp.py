from scipy.io import loadmat
import numpy as np

# 读取Person数据
OSPerson_dict = {}
rawTracePerson_dict = {}
for i in range(1, 5):
    OSPerson_dict[f'OSPerson{i}'] = loadmat(f'C:\\Users\\Zhen\\Desktop\\大学\\神经AI\\人Task2数据整理\\Person{i}\\Person{i}\\OSPerson{i}.mat')
    rawTracePerson_dict[f'rawTracePerson{i}'] = loadmat(f'C:\\Users\\Zhen\\Desktop\\大学\\神经AI\\人Task2数据整理\\Person{i}\\Person{i}\\rawTracePerson{i}.mat')

# 读取其他数据
ChanName = loadmat('C:\\Users\\Zhen\\Desktop\\大学\\神经AI\\人Task2数据整理\\ChanName.mat')
Pair54 = loadmat('C:\\Users\\Zhen\\Desktop\\大学\\神经AI\\人Task2数据整理\\Pair54.mat')
timeRawTrace = loadmat('C:\\Users\\Zhen\\Desktop\\大学\\神经AI\\人Task2数据整理\\timeRawTrace.mat')
OSPerson_data = {}

for i in range(1, 5):
    print(OSPerson_dict[f'OSPerson{i}']['OS'].shape)
    # 把Trial维度提到最前面，现顺序为Trialx时间x频率x配对
    OSPerson_dict[f'OSPerson{i}']['OS'] = np.transpose(OSPerson_dict[f'OSPerson{i}']['OS'][-21:-5, 48:, :, :], (2, 0, 1, 3))
    OSPerson_dict[f'OSPerson{i}']['Time'] =  np.transpose(OSPerson_dict[f'OSPerson{i}']['Time'][:, -21:-5], (1, 0))
    OSPerson_dict[f'OSPerson{i}']['fOS'] = OSPerson_dict[f'OSPerson{i}']['fOS'][48:]
    time_list = OSPerson_dict[f'OSPerson{i}']['Time'].ravel().tolist()
    fOS_list = OSPerson_dict[f'OSPerson{i}']['fOS'].ravel().tolist()
    track_list = OSPerson_dict[f'OSPerson{i}']['Track'].ravel().tolist()
    print(OSPerson_dict[f'OSPerson{i}']['OS'].shape)
    
    # 更新OSPerson_data字典
    OSPerson_data[f'OSPerson{i}'] = {'Time': time_list, 'fOS': fOS_list, 'Track': track_list}

train_data_dict = {}
for i in range(1,5):
    data = OSPerson_dict[f'OSPerson{i}']['OS']
    for j in range(len(OSPerson_data[f'OSPerson{i}']['Track'])):
        # 检查键是否已经存在于字典中
        if OSPerson_data[f'OSPerson{i}']['Track'][j] not in train_data_dict:
            # 如果不存在，创建一个新的列表
            train_data_dict[OSPerson_data[f'OSPerson{i}']['Track'][j]] = [data[j]]
        else:
            # 如果存在，向列表中添加新的三维数组
            train_data_dict[OSPerson_data[f'OSPerson{i}']['Track'][j]].append(data[j])
print(train_data_dict.keys())
print(len(train_data_dict[2]))
print(train_data_dict[2][0].shape)
from collections import OrderedDict
# 使用sorted函数对字典的键进行排序，并创建一个OrderedDict以保持这个顺序
sorted_train_data_dict = OrderedDict(sorted(train_data_dict.items()))
# 打印排序后的字典的键，以验证排序是否成功
print(sorted_train_data_dict.keys())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个新的三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义40种颜色
colors = plt.cm.jet(np.linspace(0, 1, 40))

# 遍历字典
# 遍历字典
for idx, (key, arrays) in enumerate(sorted_train_data_dict.items()):
    # 对于每个键，遍历其对应的三维数组列表
    for array in arrays:
        # 只提取数值大于0.9的点的坐标
        x, y, z = np.where(array > 0.9)
        # 根据灰度值调整透明度
        alpha_values = np.clip(array[x, y, z] / array.max(), 0.1, 1)
        # 使用点的坐标和调整后的透明度在三维图中绘制散点
        ax.scatter(x, y, z, color=colors[idx], alpha=alpha_values)

# 设置图形的标题和坐标轴标签
ax.set_title('3D Scatter Plot')
ax.set_xlabel('time')
ax.set_ylabel('frequency')
ax.set_zlabel('pair')

# 显示图形
plt.show()