import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers import Conv3D, Conv3DTranspose
from keras.layers import LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from keras.layers import Input
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
    OSPerson_dict[f'OSPerson{i}']['OS'] = np.transpose(OSPerson_dict[f'OSPerson{i}']['OS'][-21:-5, 30:, :, :], (2, 3, 1, 0))#这里通道数要提前到第二位，方便后续
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

label_all = []
array_all = []

for key in sorted_train_data_dict.keys():
    if len(sorted_train_data_dict[key]) >= 2 and key <= 10:
        for j in range(len(sorted_train_data_dict[key])):
            label_all.append(key-2)
            array_all.append(sorted_train_data_dict[key][j])
print(label_all)

data = array_all
labels = label_all

# 将列表转换为 numpy 数组
data = np.array(data)
labels = np.array(labels)

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 增加一个维度以匹配输入形状
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

def build_generator(latent_dim):
    model = Sequential()
    model.add(Input(shape=(latent_dim,)))  # 使用Input层定义输入形状
    model.add(Dense(128 * 27 * 11 * 8, activation="relu"))
    model.add(Reshape((27, 11, 8, 128)))
    model.add(Conv3DTranspose(64, kernel_size=3, strides=(2, 2, 2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(negative_slope=0.2))  # 使用negative_slope代替alpha
    model.add(Conv3DTranspose(1, kernel_size=3, strides=(1, 1, 1), padding="same", activation="sigmoid"))
    model.add(Reshape((54, 22, 16)))  # 确保这里的形状与之前层的输出匹配
    return model
def build_discriminator(img_shape):
    model = Sequential()
    model.add(Input(shape=img_shape))  # 使用Input层定义输入形状
    model.add(Conv3D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def compile_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

def train_gan(generator, discriminator, gan, X_train, epochs=1000, batch_size=32, latent_dim=100):
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        real_labels = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        
        # Optionally print the progress
        print(f"Epoch: {epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

def generate_new_data(generator, latent_dim=100, labels_range=(0, 8), samples_per_label=5):
    generated_images = []
    generated_labels = []
    for label in range(labels_range[0], labels_range[1] + 1):
        for _ in range(samples_per_label):
            noise = np.random.normal(0, 1, (1, latent_dim))
            generated_image = generator.predict(noise)
            generated_images.append(generated_image)
            generated_labels.append(label)
    generated_images = np.vstack(generated_images)  # 将生成的图像列表转换为numpy数组
    generated_labels = np.array(generated_labels)
    return generated_images, generated_labels

# Example usage
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator((54, 22, 16, 1))
gan = compile_gan(generator, discriminator)

# 使用修改后的函数生成数据
n_samples = 5  # 每个标签生成的样本数量
labels_range = (0, 8)  # 标签范围
generated_images, generated_labels = generate_new_data(generator, latent_dim, labels_range, n_samples)

# 将生成的标签和图像加入到对应的列表中
label_all.extend(generated_labels)
array_all.extend(generated_images)

# 打印更新后的列表长度，确认数据已被添加
print(f"Total labels in label_all: {len(label_all)}")
print(f"Total arrays in array_all: {len(array_all)}")

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
        self.flattened_size = 64* 11* 8
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

label_all = []
array_all = []
for key in sorted_train_data_dict.keys():
    if len(sorted_train_data_dict[key]) >= 2 and key <= 10:
        for j in range(len(sorted_train_data_dict[key])):
            label_all.append(key-2)
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
num_classes_1 = 9  # 类别数量

# 定义参数网格
param_grid_1 = {
    'lr': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1], #学习率
    'hidden_size': [32, 64, 128, 256, 512], #隐藏层大小
    'batch_size': [10, 20, 30, 40, 50], #批量大小
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