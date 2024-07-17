from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 读取Person数据
OSPerson_dict = {}

rawTracePerson_dict = {}
for i in range(1, 5):
    OSPerson_dict[f'OSPerson{i}'] = loadmat(f'E:\大一暑学期\神经AI\\Person{i}\\OSPerson{i}.mat')
    rawTracePerson_dict[f'rawTracePerson{i}'] = loadmat(f'E:\大一暑学期\神经AI\\Person{i}\\rawTracePerson{i}.mat')

# 读取其他数据
ChanName = loadmat('E:\大一暑学期\神经AI\\ChanName.mat')
print(type(ChanName['ChanName']))
Pair54 = loadmat('E:\大一暑学期\神经AI\\Pair54.mat')
print(type(Pair54['Pair54']))
timeRawTrace = loadmat('E:\大一暑学期\神经AI\\timeRawTrace.mat')
print(type(timeRawTrace['timeRawTrace']))

ChanName = loadmat('E:\大一暑学期\神经AI\\ChanName.mat')
print(type(ChanName['ChanName']))
Pair54 = loadmat('E:\大一暑学期\神经AI\\Pair54.mat')
print(type(Pair54['Pair54']))
timeRawTrace = loadmat('E:\大一暑学期\神经AI\\timeRawTrace.mat')
print(type(timeRawTrace['timeRawTrace']))

Pair54 = loadmat('E:\大一暑学期\神经AI\\Pair54.mat')
print(type(Pair54['Pair54']))
timeRawTrace = loadmat('E:\大一暑学期\神经AI\\timeRawTrace.mat')
print(type(timeRawTrace['timeRawTrace']))

# 在循环外初始化一个空字典
OSPerson_data = {}

for i in range(1, 5):
    print(OSPerson_dict[f'OSPerson{i}']['OS'].shape)
    # 把Trial维度提到最前面，现顺序为Trialx时间x频率x配对
    OSPerson_dict[f'OSPerson{i}']['OS'] = np.transpose(OSPerson_dict[f'OSPerson{i}']['OS'][-21:-5, 30:40, :, :], (2, 3, 0, 1))#这里通道数要提前到第二位，方便后续
    OSPerson_dict[f'OSPerson{i}']['Time'] =  np.transpose(OSPerson_dict[f'OSPerson{i}']['Time'][:, -21:-5], (1, 0))
    OSPerson_dict[f'OSPerson{i}']['fOS'] = OSPerson_dict[f'OSPerson{i}']['fOS'][30:]
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

class MyCustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features #特征
        self.labels = labels #标签，这里可以视为图像编号，也就是track

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): #idx是index（索引）的缩写
        feature_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature_tensor, label_tensor
    
class MyNeuralNetwork(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes, dropout_rate):
        super(MyNeuralNetwork, self).__init__()
        # 类比图像处理，输入(batch_size, channels, height, width)四个参数
        # 这里batch_size是一次处理的数据量
        # channels取通道数
        # height和width分别视作频率和时间
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1) # 卷积层
        self.relu = nn.ReLU() # 激活层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 池化层
        self.dropout = nn.Dropout(dropout_rate) # 防止过拟合
        # 重新计算卷积和池化后的特征图尺寸
        self.flattened_size = 64* 8* 5
        self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, self.flattened_size)  # 展平特征图
        x = self.fc(x)
        return x
    
# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available. Running on CPU instead.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
from sklearn.model_selection import train_test_split

label_all = []
array_all = []
for key in sorted_train_data_dict.keys():
    if len(sorted_train_data_dict[key]) >= 2 and key <= 10:
        for j in range(len(sorted_train_data_dict[key])):
            label_all.append(key)
            array_all.append(sorted_train_data_dict[key][j])
print(label_all)

train_features, test_features, train_labels, test_labels = train_test_split(
    array_all, label_all, test_size = 0.2, random_state = 42, stratify = label_all
)

train_dataset_1 = MyCustomDataset(train_features, train_labels)

test_dataset_1 = MyCustomDataset(test_features, test_labels)
test_loader_1 = DataLoader(test_dataset_1, batch_size = 20, shuffle = False)

# 初始化网络
input_size_1 = 54  # 特征数量，这里应该是指通道数
num_classes_1 = 11  # 类别数量

# 定义参数网格
param_grid_1 = {
    'lr': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    'hidden_size': [128, 256, 512, 1024],
    'batch_size': [10, 20, 30, 40, 50],
    'num_epochs': [3, 5, 7, 9],  # 训练的轮数
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'weight_decay': [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
}

# 用于记录最佳验证损失
best_loss = float('inf')
best_params = {}

# 网格搜索
for lr in param_grid_1['lr']:
    for hidden_size in param_grid_1['hidden_size']:
        for batch_size in param_grid_1['batch_size']:
            for num_epochs in param_grid_1['num_epochs']:
                for dropout_rate in param_grid_1['dropout_rate']:
                    for weight_decay in param_grid_1['weight_decay']:
                        print(
                            f'Testing with lr = {lr},'
                            f'hidden_size = {hidden_size},'
                            f'batch_size = {batch_size},'
                            f'num_epochs = {num_epochs},'
                            f'dropout_rate = {dropout_rate},'
                            f'weight_decay = {weight_decay}'
                        )

                        # 初始化网络
                        model = MyNeuralNetwork(input_size_1, hidden_size, num_classes_1, dropout_rate).to(device)
                        
                        # 初始化损失函数和优化器
                        criterion = nn.CrossEntropyLoss()
                        # 重新初始化优化器
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                        
                        train_loader = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
                        
                        # 初始化学习率调度器
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
                        
                        # 训练网络
                        for epoch in range(num_epochs):
                            model.train()
                            epoch_loss = 0.0
                            for features, labels in train_loader:
                                features, labels = features.to(device), labels.to(device)
                                optimizer.zero_grad()
                                outputs = model(features)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()
                                epoch_loss += loss.item()
                            scheduler.step()  # 更新学习率
                            print(f"Epoch[{epoch+1}/{num_epochs}], Loss:{epoch_loss / len(train_loader):.4f}")

                            # 评估模型
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for features, labels in test_loader_1:
                                features, labels = features.to(device), labels.to(device)
                                outputs = model(features)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item()

                        # 记录最佳参数
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_params = {
                                'lr': lr,
                                'hidden_size': hidden_size,
                                'batch_size': batch_size,
                                'num_epochs': num_epochs,
                                'dropout_rate': dropout_rate,
                                'weight_decay': weight_decay
                            }
                            print(f"New best params: {best_params}")

print(f"Best loss: {best_loss} with params: {best_params}")


# 使用最佳参数重新训练模型
best_lr = best_params['lr']
best_hidden_size = best_params['hidden_size']
best_batch_size = best_params['batch_size']
best_num_epochs = best_params['num_epochs']
best_dropout_rate = best_params['dropout_rate']
best_weight_decay = best_params['weight_decay']

# 重新初始化网络、损失函数、优化器
model_1 = MyNeuralNetwork(input_size_1, best_hidden_size, num_classes_1, best_dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(), lr=best_lr, weight_decay=best_weight_decay)
train_loader = DataLoader(train_dataset_1, batch_size=best_batch_size, shuffle=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # 添加学习率调度器

# 训练网络
for epoch in range(best_num_epochs):
    model_1.train()
    epoch_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_1(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    print(f"Epoch[{epoch+1}/{best_num_epochs}], Loss:{epoch_loss / len(train_loader):.4f}")

# 评估模型
model_1.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in test_loader_1:
        features, labels = features.to(device), labels.to(device)
        outputs = model_1(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test data: {100 * correct / total}%')

# 将模型转移至CPU保存
model_1.to('cpu')
model_1_path = 'E:\大一暑学期\神经AI\model_1.pth'
torch.save(model_1, model_1_path)