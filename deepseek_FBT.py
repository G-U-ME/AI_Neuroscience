import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import cv2
from torchvision.utils import make_grid

# ========================
# 环境配置
# ========================
MAP_SIZE = 20  # 20x20 的地图
PREDATOR_COUNT = 5
MAX_FOOD = 50
PIXEL_TYPES = {
    0: 'environment',
    1: 'predator',
    2: 'food',
    3: 'agent'
}
REWARDS = {
    'predator': -10,
    'food': +5,
    'environment': -1
}

# ========================
# 环境实现
# ========================
class SurvivalGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, image_datasets: Dict[str, List[np.ndarray]]):
        super(SurvivalGameEnv, self).__init__()
        
        # 图片数据集
        self.image_datasets = image_datasets
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 动作空间: 0=Escape, 1=Eat, 2=Wander
        self.action_space = spaces.Discrete(3)
        
        # 观察空间: 4张100x100x3的图片 (前、左、右、后)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(4, 100, 100, 3), 
            dtype=np.uint8
        )
        
        # 初始化地图和实体
        self.reset()
    
    def reset(self):
        """重置环境"""
        # 创建空白地图 (20x20)
        self.map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        self.food_active = {}  # 记录食物激活状态
        
        # 放置捕食者
        self.predators = []
        for _ in range(PREDATOR_COUNT):
            pos = self._random_edge_position()
            self.map[pos] = 1
            self.predators.append(pos)
        
        # 放置食物 (集群分布)
        self.foods = []
        food_clusters = 3  # 3个食物集群
        for _ in range(food_clusters):
            cluster_center = self._random_position()
            for _ in range(MAX_FOOD // food_clusters):
                offset = np.random.randint(-2, 3, size=2)
                pos = (cluster_center[0] + offset[0], cluster_center[1] + offset[1])
                if self._is_valid_position(pos) and self.map[pos] == 0:
                    self.map[pos] = 2
                    self.foods.append(pos)
                    self.food_active[pos] = False  # 初始未激活
        
        # 放置智能体
        self.agent_pos = self._random_position()
        while self.map[self.agent_pos] != 0:  # 确保初始位置为空
            self.agent_pos = self._random_position()
        self.map[self.agent_pos] = 3
        
        # 记录上一位置
        self.prev_agent_pos = self.agent_pos
        
        # 游戏状态
        self.steps = 0
        self.done = False
        
        return self._get_observation()
    
    def step(self, action: int):
        """执行动作"""
        # 获取当前观察
        observation = self._get_observation()
        
        # 根据动作计算移动
        move_dist = self._calculate_movement(action, observation)
        new_pos = self._move_agent(move_dist)
        
        # 处理实体交互
        reward = 0
        entity_type = self.map[new_pos]
        
        if entity_type == 1:  # 捕食者
            reward = REWARDS['predator']
            self.done = True
        elif entity_type == 2:  # 食物
            # 检查食物是否已激活（被访问过）
            if self.food_active.get(new_pos, False):
                reward = REWARDS['food']
                self.foods.remove(new_pos)
                self.map[new_pos] = 0  # 食物消失
                del self.food_active[new_pos]
            else:
                # 第一次访问食物，激活它
                self.food_active[new_pos] = True
                reward = 0  # 第一次访问不给奖励
        else:  # 环境
            reward = REWARDS['environment']
        
        # 处理食物激活状态（当智能体离开食物位置时）
        if self.prev_agent_pos in self.foods and self.agent_pos != self.prev_agent_pos:
            if self.food_active.get(self.prev_agent_pos, False):
                self.map[self.prev_agent_pos] = 0  # 食物消失
                if self.prev_agent_pos in self.foods:
                    self.foods.remove(self.prev_agent_pos)
                if self.prev_agent_pos in self.food_active:
                    del self.food_active[self.prev_agent_pos]
        
        # 更新智能体位置
        self.map[self.agent_pos] = 0
        self.prev_agent_pos = self.agent_pos
        self.agent_pos = new_pos
        self.map[self.agent_pos] = 3
        
        # 移动捕食者
        self._move_predators()
        
        # 检查游戏结束条件
        self.steps += 1
        if self.steps >= 1000:  # 最大步数
            self.done = True
        
        return observation, reward, self.done, {}
    
    def render(self, mode='human'):
        """可视化地图 (简化版)"""
        if mode == 'human':
            plt.imshow(self.map, cmap='viridis')
            plt.title(f'Step: {self.steps}')
            plt.show()
    
    def _get_observation(self) -> np.ndarray:
        """获取四方向观察图片"""
        directions = [
            (0, 1),   # 前方
            (-1, 0),  # 左方
            (1, 0),   # 右方
            (0, -1)   # 后方
        ]
        
        observations = []
        for dx, dy in directions:
            # 计算目标位置
            target_pos = (
                (self.agent_pos[0] + dx) % MAP_SIZE,
                (self.agent_pos[1] + dy) % MAP_SIZE
            )
            
            # 获取实体类型
            entity_type = PIXEL_TYPES[self.map[target_pos]]
            
            # 随机选择对应类型的图片
            img_array = np.random.choice(self.image_datasets[entity_type])
            observations.append(img_array)
        
        return np.array(observations, dtype=np.uint8)
    
    def _calculate_movement(self, action: int, observation: np.ndarray) -> Tuple[float, float]:
        """根据动作计算移动分布"""
        # 动作解释
        move_vectors = []
        
        for i in range(4):  # 四个方向
            if action == 0:  # Escape: 远离感知到的像素
                # 根据方向生成远离向量
                if i == 0: move_vectors.append((0, -1))   # 远离前方 -> 向后
                elif i == 1: move_vectors.append((1, 0))   # 远离左方 -> 向右
                elif i == 2: move_vectors.append((-1, 0))  # 远离右方 -> 向左
                elif i == 3: move_vectors.append((0, 1))   # 远离后方 -> 向前
            elif action == 1:  # Eat: 靠近感知到的像素
                # 根据方向生成靠近向量
                if i == 0: move_vectors.append((0, 1))    # 靠近前方 -> 向前
                elif i == 1: move_vectors.append((-1, 0))  # 靠近左方 -> 向左
                elif i == 2: move_vectors.append((1, 0))   # 靠近右方 -> 向右
                elif i == 3: move_vectors.append((0, -1))  # 靠近后方 -> 向后
            else:  # Wander: 随机移动
                move_vectors.append((
                    np.random.choice([-1, 0, 1]), 
                    np.random.choice([-1, 0, 1])
                )
        
        # 合并所有方向向量
        total_dx, total_dy = 0, 0
        for dx, dy in move_vectors:
            total_dx += dx
            total_dy += dy
        
        # 归一化
        norm = np.sqrt(total_dx**2 + total_dy**2) + 1e-8
        return total_dx/norm, total_dy/norm
    
    def _move_agent(self, move_dist: Tuple[float, float]) -> Tuple[int, int]:
        """移动智能体"""
        dx, dy = move_dist
        new_x = max(0, min(MAP_SIZE-1, int(self.agent_pos[0] + dx)))
        new_y = max(0, min(MAP_SIZE-1, int(self.agent_pos[1] + dy)))
        return (new_x, new_y)
    
    def _move_predators(self):
        """随机移动所有捕食者"""
        new_predators = []
        for pos in self.predators:
            # 随机移动
            dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
            new_pos = (pos[0] + dx, pos[1] + dy)
            
            # 检查捕食者是否移出地图
            if not self._is_valid_position(new_pos):
                # 在边缘生成新捕食者
                new_pos = self._random_edge_position()
                # 确保新位置有效
                while self.map[new_pos] != 0:
                    new_pos = self._random_edge_position()
                self.map[pos] = 0
                self.map[new_pos] = 1
                new_predators.append(new_pos)
                continue
            
            # 检查新位置是否有效
            if self.map[new_pos] == 0:  # 仅当位置为空时移动
                self.map[pos] = 0
                self.map[new_pos] = 1
                new_predators.append(new_pos)
            elif new_pos == self.agent_pos:  # 捕获智能体
                self.done = True
                new_predators.append(pos)  # 保持原位置
            else:
                new_predators.append(pos)  # 无法移动，保持原位置
        
        self.predators = new_predators
    
    def _random_position(self) -> Tuple[int, int]:
        """生成随机位置"""
        return (np.random.randint(0, MAP_SIZE), np.random.randint(0, MAP_SIZE))
    
    def _random_edge_position(self) -> Tuple[int, int]:
        """生成地图边缘位置"""
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            return (0, np.random.randint(0, MAP_SIZE))
        elif edge == 'bottom':
            return (MAP_SIZE-1, np.random.randint(0, MAP_SIZE))
        elif edge == 'left':
            return (np.random.randint(0, MAP_SIZE), 0)
        else:  # right
            return (np.random.randint(0, MAP_SIZE), MAP_SIZE-1)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效"""
        return 0 <= pos[0] < MAP_SIZE and 0 <= pos[1] < MAP_SIZE

# ========================
# 模型架构
# ========================
class PerceptionModule(nn.Module):
    """感知模块 (基于ConvNeXt)"""
    def __init__(self):
        super(PerceptionModule, self).__init__()
        self.convnext = models.convnext_tiny(pretrained=True)
        self.convnext.classifier = nn.Identity()  # 移除分类层
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: (batch_size, 4, 3, 100, 100)
        batch_size = x.size(0)
        x = x.view(-1, 3, 100, 100)  # 合并方向维度
        features = self.convnext(x)
        features = features.view(batch_size, 4, -1)  # 恢复方向维度
        return features

class DecisionModule(nn.Module):
    """决策模块 (多层感知机)"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256]):
        super(DecisionModule, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 3))  # 输出3个动作
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: (batch_size, 4, feature_dim)
        x = x.view(x.size(0), -1)  # 展平所有方向特征
        return self.net(x)

class AgentModel(nn.Module):
    """完整智能体模型"""
    def __init__(self):
        super(AgentModel, self).__init__()
        self.perception = PerceptionModule()
        self.decision = DecisionModule(input_dim=4*768)  # ConvNeXt Tiny输出768维特征
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.perception(x)
        return self.decision(features)

# ========================
# 自编码器 (用于Truth方法)
# ========================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器 (与感知模块相同)
        self.encoder = models.convnext_tiny(pretrained=True)
        self.encoder.classifier = nn.Identity()
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 768, 3, 3)  # 调整形状以匹配解码器输入
        x = self.decoder(x)
        return x

# ========================
# 训练方法
# ========================
class CustomCNN(nn.Module):
    """自定义CNN特征提取器"""
    def __init__(self, perception_module):
        super(CustomCNN, self).__init__()
        self.perception = perception_module
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 输入: (batch, 4, 100, 100, 3)
        observations = observations.permute(0, 1, 4, 2, 3)  # 转换为 (batch, 4, 3, 100, 100)
        features = self.perception(observations)
        return features.view(observations.size(0), -1)  # 展平

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, perception_module=None, **kwargs):
        self.perception_module = perception_module
        super(CustomPolicy, self).__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        # 覆盖此方法以使用我们的决策模块
        self.mlp_extractor = DecisionModule(input_dim=4*768)

def fitness_training(env: gym.Env, total_timesteps: int = 10000):
    """Fitness方法训练 (端到端RL)"""
    perception = PerceptionModule()
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(perception_module=perception),
        net_arch=[]  # 使用自定义决策模块
    )
    
    model = PPO(
        CustomPolicy,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=total_timesteps)
    return model

def truth_training(env: gym.Env, ae_path: str, total_timesteps: int = 10000):
    """Truth方法训练 (固定感知模块)"""
    # 加载预训练自编码器
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(ae_path))
    perception = autoencoder.encoder
    
    # 冻结感知模块参数
    for param in perception.parameters():
        param.requires_grad = False
    
    # 创建模型
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(perception_module=perception),
        net_arch=[]  # 使用自定义决策模块
    )
    
    model = PPO(
        CustomPolicy,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=total_timesteps)
    return model

def train_autoencoder(dataset: List[np.ndarray], epochs: int = 10, batch_size: int = 32):
    """训练自编码器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 简化数据集处理
    data = torch.stack([transform(img) for img in dataset[:1000]]).to(device)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    print("Training Autoencoder...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, inputs in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 10 == 9:
                print(f'Epoch {epoch+1}, Batch {i+1}: Loss {running_loss/10:.4f}')
                running_loss = 0.0
    
    print("Autoencoder training complete.")
    return model

# ========================
# 可视化工具
# ========================
class GradCAM:
    """Grad-CAM可视化类"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        # 前向传播
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[:, class_idx] = 1
        output.backward(gradient=one_hot)
        
        # 计算权重
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 加权激活图
        cam = torch.sum(self.activations * pooled_gradients, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        # 上采样到输入大小
        cam = torch.nn.functional.interpolate(cam, size=(100, 100), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

def visualize_gradcam(model, image, layer_name='convnext.blocks[3].layers[0].block[0]'):
    """使用Grad-CAM可视化模型关注区域"""
    # 获取目标层
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        print(f"Layer {layer_name} not found!")
        return None
    
    # 创建Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # 预处理图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    # 获取CAM
    cam = gradcam(input_tensor)
    
    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # 原始图像
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # CAM叠加
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    ax[1].imshow(superimposed_img)
    ax[1].set_title('Grad-CAM')
    ax[1].axis('off')
    
    plt.tight_layout()
    return fig

# ========================
# 评估函数
# ========================
def evaluate_survival(model, env: gym.Env, n_episodes: int = 10) -> float:
    """评估生存时间"""
    survival_times = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            steps += 1
        
        survival_times.append(steps)
    
    return np.mean(survival_times)

def visualize_perception(model, image_datasets):
    """可视化感知特征"""
    # 随机选择样本图像
    sample_images = {
        'predator': np.random.choice(image_datasets['predator']),
        'food': np.random.choice(image_datasets['food']),
        'environment': np.random.choice(image_datasets['environment'])
    }
    
    # 创建可视化
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    for i, (label, img) in enumerate(sample_images.items()):
        # Grad-CAM可视化
        cam_fig = visualize_gradcam(model.perception, img)
        if cam_fig:
            # 从图中提取图像
            cam_fig.canvas.draw()
            cam_img = np.frombuffer(cam_fig.canvas.tostring_rgb(), dtype=np.uint8)
            cam_img = cam_img.reshape(cam_fig.canvas.get_width_height()[::-1] + (3,))
            
            # 原始图像
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Original: {label}')
            axes[i, 0].axis('off')
            
            # Grad-CAM结果
            axes[i, 1].imshow(cam_img)
            axes[i, 1].set_title(f'Grad-CAM: {label}')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Perception Visualization with Grad-CAM', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

# ========================
# 主工作流程
# ========================
def main():
    # 模拟图片数据集 (实际应加载真实图片)
    image_datasets = {
        'predator': [np.random.rand(100, 100, 3) * 255 for _ in range(100)],
        'food': [np.random.rand(100, 100, 3) * 255 for _ in range(100)],
        'environment': [np.random.rand(100, 100, 3) * 255 for _ in range(100)],
        'agent': [np.random.rand(100, 100, 3) * 255 for _ in range(100)]
    }
    
    # 创建环境
    env = SurvivalGameEnv(image_datasets)
    
    # 训练自编码器 (Truth方法需要)
    print("\nPretraining Autoencoder...")
    # 合并所有图片用于自编码器训练
    all_images = []
    for images in image_datasets.values():
        all_images.extend(images)
    autoencoder = train_autoencoder(all_images, epochs=5)
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    
    # 训练Fitness模型
    print("\nTraining Fitness model...")
    fitness_model = fitness_training(env, total_timesteps=5000)
    
    # 训练Truth模型
    print("\nTraining Truth model...")
    truth_model = truth_training(env, ae_path="autoencoder.pth", total_timesteps=5000)
    
    # 评估模型
    print("\nEvaluating models...")
    fitness_score = evaluate_survival(fitness_model, env)
    truth_score = evaluate_survival(truth_model, env)
    
    print(f"\nResults:")
    print(f"Fitness Model Survival Time: {fitness_score:.1f} steps")
    print(f"Truth Model Survival Time: {truth_score:.1f} steps")
    
    # 可视化
    print("\nVisualizing Fitness Model Perception...")
    visualize_perception(fitness_model.policy, image_datasets)
    
    print("\nVisualizing Truth Model Perception...")
    visualize_perception(truth_model.policy, image_datasets)

if __name__ == "__main__":
    main()