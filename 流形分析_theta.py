from scipy.io import loadmat

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

import numpy as np

import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 数据预处理
data_reshaped_all_2 = np.zeros((0, 7776))
labels_all_2 = np.array([])
for i in range(1, 5):
    data_2 = OSPerson_dict[f'OSPerson{i}']['OS'][-21:-5, 4:13, :, :]

    # 首先使用transpose调整维度顺序
    data_transposed_2 = np.transpose(data_2, (2, 0, 1, 3))  # 将第三维（大小为40的维度）移到最前面
    # 然后将其余维度合并为一维
    data_reshaped_2 = data_transposed_2.reshape(data_transposed_2.shape[0], -1)  # 重塑数据为二维数组
    print(data_reshaped_2.shape)
    # 16x9x54 = 7776
    data_reshaped_all_2 = np.concatenate((data_reshaped_all_2, data_reshaped_2), axis=0)
    print(data_reshaped_all_2.shape)

    param_grid_2 = {
    'n_neighbors': [10, 12, 15, 20, 22, 25], 
    'metric': ['euclidean', 'cosine']
}
best_params_2 = {}
best_score_2 = 0
for metric in param_grid_2['metric']:
    for n_neighbors in param_grid_2['n_neighbors']:
        print(f'test with n_neighbors: {n_neighbors}, metric: {metric}')
        # 初始化UMAP
        umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=3, metric=metric, random_state=None)
        # 拟合模型
        embedding = umap_model.fit_transform(data_reshaped_all_2)
        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(embedding)

        # 计算轮廓系数
        silhouette_avg = silhouette_score(embedding, labels)
        print("Silhouette Coefficient: ", silhouette_avg)

        # 计算戴维斯-布老德指数
        dbi = davies_bouldin_score(embedding, labels)
        print("Davies-Bouldin Index: ", dbi)

        # 计算Calinski-Harabasz指数
        ch = calinski_harabasz_score(embedding, labels)
        print("Calinski-Harabasz Index: ", ch)

        score = 1 / (1 + np.exp(-1 - silhouette_avg)) + 1 / (1 + np.exp(- dbi)) + 1 / (1 + np.exp(-ch))
        if score > best_score_2:
            best_score_2 = score
            best_params_2 = {'n_neighbors': n_neighbors, 'metric': metric}

print(f'best score: {best_score_2} with best params: {best_params_2}')
best_best_score_2 = 0
for i in range(10):     
    # 初始化UMAP
    umap_model = umap.UMAP(n_neighbors=best_params_2['n_neighbors'], n_components=3, metric=best_params_2['metric'], random_state=None)
    # 拟合模型
    embedding = umap_model.fit_transform(data_reshaped_all_2)

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(embedding)

    # 计算轮廓系数
    silhouette_avg = silhouette_score(embedding, labels)
    print("Silhouette Coefficient: ", silhouette_avg)

    # 计算戴维斯-布老德指数
    dbi = davies_bouldin_score(embedding, labels)
    print("Davies-Bouldin Index: ", dbi)

    # 计算Calinski-Harabasz指数
    ch = calinski_harabasz_score(embedding, labels)
    print("Calinski-Harabasz Index: ", ch)

    score = 1 / (1 + np.exp(-1 - silhouette_avg)) + 1 / (1 + np.exp(- dbi)) + 1 / (1 + np.exp(-ch))
    if score > best_best_score_2:
        best_best_score_2 = score
        print(best_best_score_2)
        best_embedding_2 = embedding
print(f'the best best score: {best_best_score_2}')

# 创建一个新的图形实例
fig = plt.figure()

# 添加一个三维坐标轴
ax = fig.add_subplot(111, projection='3d')

# 可视化结果
ax.scatter(best_embedding_2[:, 0], best_embedding_2[:, 1], best_embedding_2[:, 2], s=5)
ax.set_title('UMAP Projection of Flattened Data')
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')
ax.set_zlabel('UMAP Component 3')
plt.show()