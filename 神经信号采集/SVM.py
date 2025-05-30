from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 读取Person数据
OSPerson_dict = {}
rawTracePerson_dict = {}
for i in range(1, 5):
    OSPerson_dict[f'OSPerson{i}'] = loadmat(f'D:\ShanghaiTech\Grade 1\summer_term\神经信号\人Task2数据整理\人Task2数据整理\Person{i}\Person{i}\OSPerson{i}.mat')
    rawTracePerson_dict[f'rawTracePerson{i}'] = loadmat(f'D:\ShanghaiTech\Grade 1\summer_term\神经信号\人Task2数据整理\人Task2数据整理\Person{i}\Person{i}\\rawTracePerson{i}.mat')

# 读取其他数据
ChanName = loadmat('D:\ShanghaiTech\Grade 1\summer_term\神经信号\人Task2数据整理\人Task2数据整理\ChanName.mat')
Pair54 = loadmat('D:\ShanghaiTech\Grade 1\summer_term\神经信号\人Task2数据整理\人Task2数据整理\Pair54.mat')
timeRawTrace = loadmat('D:\ShanghaiTech\Grade 1\summer_term\神经信号\人Task2数据整理\人Task2数据整理\\timeRawTrace.mat')

# 数据预处理
data = OSPerson_dict['OSPerson1']['OS'][-21:-5, 28:, :, :]

# 首先使用transpose调整维度顺序
data_transposed_3 = np.transpose(data, (2, 0, 1, 3))  # 将第三维（大小为40的维度）移到最前面
# 然后将其余维度合并为一维
data_reshaped_all_3 = data_transposed_3.reshape(40, -1)  # 重塑数据为二维数组

# 创建标签数组
labels_all_3 = np.array([0 if x > 10 else 1 for x in OSPerson_dict['OSPerson1']['Track'][0]])

for i in range(2, 5):
    data = OSPerson_dict[f'OSPerson{i}']['OS'][-21:-5, 28:, :, :]

    # 首先使用transpose调整维度顺序
    data_transposed_3 = np.transpose(data, (2, 0, 1, 3))  # 将第三维（大小为40的维度）移到最前面
    # 然后将其余维度合并为一维
    data_reshaped = data_transposed_3.reshape(data_transposed_3.shape[0], -1)  # 重塑数据为二维数组
    data_reshaped_all_3 = np.concatenate((data_reshaped_all_3, data_reshaped), axis=0)

    labels = np.array([0 if x > 10 else 1 for x in OSPerson_dict[f'OSPerson{i}']['Track'][0]])
    labels_all_3 = np.concatenate((labels_all_3, labels), axis=0)


# 数据预处理：特征缩放
scaler = StandardScaler()
data_reshaped_all_scaled_3 = scaler.fit_transform(data_reshaped_all_3)


# 分割数据为训练集和测试集
# 假设data_reshaped_all_scaled_3和labels_all_3已经准备好
num_samples = data_reshaped_all_scaled_3.shape[0]  # 总样本数
group_size = 10  # 每组的大小

# 初始化最终的训练集和测试集
X_train_final, X_test_final, y_train_final, y_test_final = [], [], [], []

# 按组处理数据
for start_idx in range(0, num_samples, group_size):
    end_idx = start_idx + group_size
    X_group = data_reshaped_all_scaled_3[start_idx:end_idx]
    y_group = labels_all_3[start_idx:end_idx]
    
    # 计算训练集和测试集的分割点
    split_idx = int(group_size * 0.8)  # 80%作为训练集
    
    # 分割训练集和测试集
    X_train, X_test = X_group[:split_idx], X_group[split_idx:]
    y_train, y_test = y_group[:split_idx], y_group[split_idx:]
    
    # 合并到最终的训练集和测试集
    X_train_final.append(X_train)
    X_test_final.append(X_test)
    y_train_final.append(y_train)
    y_test_final.append(y_test)

# 将列表转换为numpy数组并合并
X_train_final = np.concatenate(X_train_final, axis=0)
X_test_final = np.concatenate(X_test_final, axis=0)
y_train_final = np.concatenate(y_train_final, axis=0)
y_test_final = np.concatenate(y_test_final, axis=0)

param_grid = {
    'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100],
    'degree': [2, 3, 4, 5]  # 对于多项式核，尝试不同的多项式度数
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_final, y_train_final)

# 打印最佳参数和最佳准确率
print(f"Best parameters: {grid_search.best_params_}")

# 使用最佳参数的模型进行训练
svm_model = grid_search.best_estimator_

# 预测测试集
y_pred = svm_model.predict(X_test_final)

# 评估模型
accuracy = accuracy_score(y_test_final, y_pred)
print(f"Model accuracy: {accuracy}\n")