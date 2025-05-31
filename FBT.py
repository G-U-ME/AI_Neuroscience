import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym # Changed from import gym
from gymnasium import spaces # Changed from from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BaseFeaturesExtractor
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Any
import cv2
import random
from collections import defaultdict

# ========================
# 环境配置
# ========================
MAP_SIZE = 20
PREDATOR_COUNT = 5
MAX_FOOD = 50
CLUSTER_RADIUS = 3
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
NUM_VIEWS = 4 # Front, Left, Right, Back

# ========================
# 环境实现
# ========================
class SurvivalGameEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, image_datasets: Dict[str, List[np.ndarray]]):
        super(SurvivalGameEnv, self).__init__()

        self.image_datasets = image_datasets
        # This transform is for external use if needed, model handles its own
        self.vis_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.action_space = spaces.Discrete(3)  # Corresponds to probabilities for Escape, Eat, Wander
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(NUM_VIEWS, 100, 100, 3), # (4 views, H, W, C)
            dtype=np.uint8
        )
        self.current_map_image = None # For rendering

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed) # Important for gymnasium compatibility
        self.map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        self.predators = []

        for _ in range(PREDATOR_COUNT):
            pos = self._random_edge_position()
            while self.map[pos] != 0: # Ensure it's empty
                 pos = self._random_edge_position()
            self.map[pos] = 1 # Predator type
            self.predators.append({
                'pos': pos,
                'img_idx': np.random.randint(len(self.image_datasets['predator']))
            })

        self.foods = {}
        food_clusters = 3
        if MAX_FOOD > 0 and food_clusters > 0 :
            cluster_centers = [self._random_position() for _ in range(food_clusters)]
            for center in cluster_centers:
                for _ in range(MAX_FOOD // food_clusters):
                    angle = random.uniform(0, 2 * np.pi)
                    distance = random.uniform(0, CLUSTER_RADIUS)
                    dx = int(distance * np.cos(angle))
                    dy = int(distance * np.sin(angle))
                    pos = (np.clip(center[0] + dx, 0, MAP_SIZE-1), np.clip(center[1] + dy, 0, MAP_SIZE-1))

                    if self._is_valid_position(pos) and self.map[pos] == 0: # Empty
                        self.map[pos] = 2 # Food type
                        self.foods[pos] = {
                            'img_idx': np.random.randint(len(self.image_datasets['food'])), # Fixed Chinese comma
                            'consumed': False
                        }
                        if len(self.foods) >= MAX_FOOD: break
                if len(self.foods) >= MAX_FOOD: break

        self.agent_pos = self._random_position()
        while self.map[self.agent_pos] != 0: # Ensure agent starts on empty
            self.agent_pos = self._random_position()
        # Agent type not explicitly on map, agent_pos tracks it. Observation shows entities around.

        self.prev_agent_pos = self.agent_pos
        self.steps = 0
        self.terminated = False
        self.truncated = False

        return self._get_observation(), {}

    def step(self, action_probs: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Calculate movement probabilities based on action_probs from agent
        move_probs_dist = self._calculate_movement_distribution(action_probs)

        # 2. Sample a discrete move direction (0:Up, 1:Down, 2:Left, 3:Right for example)
        # Let's define directions: 0:Front(Up), 1:Left, 2:Right, 3:Back(Down) relative to agent's map orientation
        # Agent's front is (0,1) in map coord. So Up on map.
        # Map directions: 0: (0,1) Up/Front, 1: (-1,0) Left, 2: (1,0) Right, 3: (0,-1) Down/Back
        chosen_direction_idx = np.random.choice(NUM_VIEWS, p=move_probs_dist)

        # 3. Move agent
        self.prev_agent_pos = self.agent_pos
        self.agent_pos = self._move_in_direction(self.agent_pos, chosen_direction_idx)

        # 4. Handle interactions and rewards
        reward = 0 # Initialize reward for the step
        collided_entity_type_on_map = self.map[self.agent_pos] # Entity type at the agent's new position

        if collided_entity_type_on_map == 1: # Agent landed on a Predator
            reward += REWARDS['predator']
            self.terminated = True
        elif collided_entity_type_on_map == 2: # Agent landed on Food
            if self.agent_pos in self.foods: # Check if this food is tracked and valid
                reward += REWARDS['food']
                # Food is consumed: remove from tracking and change map state *immediately*
                del self.foods[self.agent_pos]
                self.map[self.agent_pos] = 0 # Pixel becomes environment *immediately*
                                            # Note: collided_entity_type_on_map is now stale if we were to re-read from self.map[self.agent_pos] here
            else:
                # Map showed food (2), but it's not in self.foods (e.g., error or already processed by other means).
                # Treat as landing on an environment cell.
                reward += REWARDS['environment']
        elif collided_entity_type_on_map == 0: # Agent landed on an Environment pixel
            reward += REWARDS['environment']
        else:
            # This case should ideally not be reached if map only contains 0, 1, 2.
            # If agent has its own map type (e.g., 3) and can land on itself (not typical), or other unhandled types.
            print(f"Warning: Agent at {self.agent_pos} landed on unexpected map type: {collided_entity_type_on_map}")
            reward += REWARDS['environment'] # Default to environment penalty

        # 5. Move predators
        self._move_predators()

        # 6. Check if predator moved onto agent (this can override previous reward)
        for pred in self.predators:
            if pred['pos'] == self.agent_pos:
                reward = REWARDS['predator'] # Override reward if caught after predator move
                self.terminated = True
                break

        self.steps += 1
        if self.steps >= 1000: # Max steps
            self.truncated = True

        return self._get_observation(), reward, self.terminated, self.truncated, {}

    #... (rest of the step function remains the same)...
    def _get_direction_vectors(self) -> List[Tuple[int, int]]:
        # Agent's front is always "map North" (0,1)
        # Observations: Front, Left, Right, Back
        # Map Coords: (row, col) or (y,x) - let's use (y,x) for consistency with plotting
        # Front: (y-1, x) (assuming map y increases downwards) -> let's use (row_change, col_change)
        # Standard image coordinates: (0,0) is top-left. y increases down, x increases right.
        # Directions: Front (Up on map: (-1,0)), Left ((0,-1)), Right ((0,1)), Back (Down on map: (1,0))
        return [(-1, 0), (0, -1), (0, 1), (1, 0)]


    def _get_observation(self) -> np.ndarray:
        observations = []
        direction_vectors = self._get_direction_vectors() # Front, Left, Right, Back

        for dr, dc in direction_vectors:
            target_pos = (
                (self.agent_pos[0] + dr + MAP_SIZE) % MAP_SIZE, # Ensure positive for modulo
                (self.agent_pos[1] + dc + MAP_SIZE) % MAP_SIZE
            )
            entity_on_map = self.map[target_pos]
            
            img_array = None
            if entity_on_map == 1: # Predator
                # Find which predator is at target_pos to get its specific image
                found_pred_img = False
                for p in self.predators:
                    if p['pos'] == target_pos:
                        img_array = self.image_datasets['predator'][p['img_idx']]
                        found_pred_img = True
                        break
                if not found_pred_img: # Should not happen if map is consistent
                    img_array = np.random.choice(self.image_datasets['predator'])

            elif entity_on_map == 2: # Food
                if target_pos in self.foods:
                     img_array = self.image_datasets['food'][self.foods[target_pos]['img_idx']]
                else: # Should not happen
                    img_array = np.random.choice(self.image_datasets['food'])
            else: # Environment
                img_array = np.random.choice(self.image_datasets['environment'])
            
            observations.append(img_array)
        return np.array(observations, dtype=np.uint8)


    def _calculate_movement_distribution(self, action_probs: np.ndarray) -> np.ndarray:
        # action_probs: [prob_escape, prob_eat, prob_wander]
        # output: [prob_move_front, prob_move_left, prob_move_right, prob_move_back]
        
        # Get surrounding entities for decision making
        # Directions: 0:Front, 1:Left, 2:Right, 3:Back
        surrounding_entities = [] # List of (map_entity_type, original_direction_idx)
        direction_vectors = self._get_direction_vectors()

        for i, (dr, dc) in enumerate(direction_vectors):
            pos = (
                (self.agent_pos[0] + dr + MAP_SIZE) % MAP_SIZE,
                (self.agent_pos[1] + dc + MAP_SIZE) % MAP_SIZE
            )
            surrounding_entities.append(self.map[pos])

        move_dist = np.zeros(NUM_VIEWS)

        # Escape: move 1 pixel AGAINST perceived pixel (predator)
        # If predator in front, escape contributes to moving back.
        # If predator left, escape contributes to moving right.
        # Opposite directions: Front (0) <-> Back (3), Left (1) <-> Right (2)
        opposite_dir_map = {0: 3, 1: 2, 2: 1, 3: 0}
        
        predator_sensed_escape = False
        for i in range(NUM_VIEWS):
            if surrounding_entities[i] == 1: # Predator in direction i
                move_dist[opposite_dir_map[i]] += action_probs[0] # Prob of escape
                predator_sensed_escape = True
        if not predator_sensed_escape and action_probs[0] > 0: # Escape wander if no predator
             move_dist += (action_probs[0] / NUM_VIEWS)


        # Eat: move 1 pixel TOWARDS perceived pixel (food)
        food_sensed_eat = False
        for i in range(NUM_VIEWS):
            if surrounding_entities[i] == 2: # Food in direction i
                move_dist[i] += action_probs[1] # Prob of eat
                food_sensed_eat = True
        if not food_sensed_eat and action_probs[1] > 0: # Eat wander if no food
            move_dist += (action_probs[1] / NUM_VIEWS)

        # Wander: move 1 pixel randomly (front, left, right, back will have same probability)
        # Probability of this move (each direction) = a quarter of the probability of Wander
        move_dist += (action_probs[2] / NUM_VIEWS)

        # Normalize
        if np.sum(move_dist) > 0:
            move_dist /= np.sum(move_dist)
        else: # If all probs are zero (e.g. from bad network output), move uniformly random
            move_dist = np.ones(NUM_VIEWS) / NUM_VIEWS
        return move_dist

    def _move_in_direction(self, current_pos: Tuple[int,int], direction_idx: int) -> Tuple[int,int]:
        dr, dc = self._get_direction_vectors()[direction_idx]
        new_r = (current_pos[0] + dr + MAP_SIZE) % MAP_SIZE
        new_c = (current_pos[1] + dc + MAP_SIZE) % MAP_SIZE
        return new_r, new_c

    def _move_predators(self):
        new_predator_list = []
        # Predator moves: 4 directions (Up, Down, Left, Right on map)
        pred_move_vectors = [(-1,0), (1,0), (0,-1), (0,1)]

        for pred_idx, pred_data in enumerate(self.predators):
            current_pos = pred_data['pos']
            self.map[current_pos] = 0 # Temporarily remove from map for its own move calculation

            # Try to move
            moved = False
            random.shuffle(pred_move_vectors) # Try directions in random order
            for dr, dc in pred_move_vectors:
                next_pos = (
                    (current_pos[0] + dr + MAP_SIZE) % MAP_SIZE,
                    (current_pos[1] + dc + MAP_SIZE) % MAP_SIZE
                )
                # Check if predator moves out of map
                is_out_of_bounds = not (0 <= current_pos[0] + dr < MAP_SIZE and 0 <= current_pos[1] + dc < MAP_SIZE)

                if is_out_of_bounds:
                    new_edge_pos = self._random_edge_position()
                    while self.map[new_edge_pos] != 0: # Find empty edge spot
                        new_edge_pos = self._random_edge_position()
                    
                    pred_data['pos'] = new_edge_pos
                    pred_data['img_idx'] = np.random.randint(len(self.image_datasets['predator']))
                    moved = True
                    break 
                
                # Can't overlap food (2) or other predators (1). Can overlap environment (0) or agent.
                if self.map[next_pos] == 0 or self.map[next_pos] == 3: # Empty or Agent
                    pred_data['pos'] = next_pos
                    moved = True
                    break
            
            if not moved: # Couldn't find a valid move, stays in place
                pass # Position remains pred_data['pos']

            new_predator_list.append(pred_data)
            self.map[pred_data['pos']] = 1 # Place predator back on map at new/old position
        
        self.predators = new_predator_list


    def _random_position(self) -> Tuple[int, int]:
        return (np.random.randint(0, MAP_SIZE), np.random.randint(0, MAP_SIZE))

    def _random_edge_position(self) -> Tuple[int, int]:
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top': return (0, np.random.randint(0, MAP_SIZE))
        if edge == 'bottom': return (MAP_SIZE - 1, np.random.randint(0, MAP_SIZE))
        if edge == 'left': return (np.random.randint(0, MAP_SIZE), 0)
        return (np.random.randint(0, MAP_SIZE), MAP_SIZE - 1) # right

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < MAP_SIZE and 0 <= pos[1] < MAP_SIZE

    def render(self):
        if self.current_map_image is None:
            fig, ax = plt.subplots()
            self.current_map_image = ax.imshow(self.map, cmap='viridis', vmin=0, vmax=len(PIXEL_TYPES)-1)
            plt.ion() # Interactive mode
            plt.show()
        else:
            # Create a display map with agent explicitly marked if not implicitly part of map values
            display_map = self.map.copy()
            # if PIXEL_TYPES[3] == 'agent', agent is already a type.
            # If agent is just a position, mark it:
            # display_map[self.agent_pos] = 3 # Mark agent if needed

            self.current_map_image.set_data(display_map)
            plt.title(f"Step: {self.steps}, Agent@ {self.agent_pos}")
            plt.gcf().canvas.draw_idle()
            plt.pause(0.1)

    def close(self):
        if self.current_map_image is not None:
            plt.ioff()
            plt.close()
            self.current_map_image = None

# ========================
# Model Architecture (Wrappers for Perception)
# ========================
class PerceptionModule(nn.Module):
    def __init__(self, pretrained_convnext=True):
        super(PerceptionModule, self).__init__()
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_convnext else None)
        self.convnext.classifier = nn.Identity() # Remove original classifier
        
        self.do_normalize = pretrained_convnext
        if self.do_normalize:
            self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.gradients = None # For GradCAM
        self.activations = None # For GradCAM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, N_VIEWS, C, H, W), scaled [0,1]
        
        # Hooks for visualization (GradCAM needs these on a specific layer, see ModelVisualizer)
        # For general input/output grads if needed:
        # if x.requires_grad:
        #     x.register_hook(lambda grad: setattr(self, 'input_gradients', grad))
        
        batch_size, num_views, C, H, W = x.size()
        x_input_to_convnext = x.view(batch_size * num_views, C, H, W)

        if self.do_normalize:
            x_input_to_convnext = self.normalize_transform(x_input_to_convnext)
        
        # Register hook for GradCAM on the output of the target layer if not done externally
        # features = self.convnext(x_input_to_convnext)
        # For GradCAM, typically want features from a specific conv layer, not the final output of convnext here.
        # The ModelVisualizer will handle hooking the specific internal layer.
        
        raw_features = self.convnext(x_input_to_convnext) # Output: (B*N_VIEWS, feature_dim e.g. 768)
        return raw_features.view(batch_size, num_views, -1) # (B, N_VIEWS, feature_dim)

class PretrainedEncoderWrapper(nn.Module):
    def __init__(self, encoder_model: nn.Module):
        super().__init__()
        self.encoder = encoder_model # This is a ConvNeXt model (autoencoder.encoder)
        # AE encoder was trained on [0,1] images, so no ImageNet normalization here.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, num_views, C, H, W), already scaled to [0,1]
        batch_size, num_views, C, H, W = x.size()
        x_reshaped = x.view(batch_size * num_views, C, H, W)
        
        features = self.encoder(x_reshaped) # encoder is ConvNeXt, outputs (B*N_VIEWS, 768)
        return features.view(batch_size, num_views, -1)

# Decision module as defined in doc (used by ModelVisualizer, SB3 PPO has its own MLP head)
class StandaloneDecisionModule(nn.Module):
    def __init__(self, input_dim: int = NUM_VIEWS * 768, hidden_dims: List[int] = [512, 256]):
        super(StandaloneDecisionModule, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 3)) # Output 3 action probabilities
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, N_VIEWS * feature_dim)
        return self.net(x)


# ========================
# 自编码器
# ========================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT) # Or None for from scratch
        self.encoder.classifier = nn.Identity() # Output of encoder is (B, 768)

        # Decoder: (B, 768, 1, 1) -> (B, 3, 100, 100)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=5, stride=1, padding=0), # (B, 512, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=5, padding=0), # (B, 256, 25, 25)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (B, 128, 50, 50)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 100, 100)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),   # (B, 3, 100, 100)
            nn.Sigmoid() # Output images in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is (B, 3, H, W)
        encoded_flat = self.encoder(x) # (B, 768)
        encoded_reshaped = encoded_flat.view(-1, 768, 1, 1) # Reshape for ConvTranspose
        decoded = self.decoder(encoded_reshaped)
        return decoded

# ========================
# 特征提取器 for Stable Baselines3
# ========================
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, perception_module: nn.Module):
        # features_dim is N_VIEWS * perception_module_output_dim_per_view
        # Assuming perception_module outputs 768 features per view
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=NUM_VIEWS * 768)
        self.perception = perception_module # This is PerceptionModule or PretrainedEncoderWrapper

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input: (batch, NUM_VIEWS, H, W, C) uint8
        # Permute to (batch, NUM_VIEWS, C, H, W) and scale to [0,1]
        observations_processed = observations.permute(0, 1, 4, 2, 3).float() / 255.0
        
        # self.perception module (PerceptionModule or PretrainedEncoderWrapper)
        # expects (B, N_VIEWS, C, H, W) scaled [0,1]
        # and returns (B, N_VIEWS, feature_dim_per_view)
        features_per_view = self.perception(observations_processed)
        
        # Flatten features from all views: (B, N_VIEWS * feature_dim_per_view)
        return features_per_view.reshape(observations.size(0), -1)

# ========================
# 训练方法
# ========================
def fitness_training(env: gym.Env, total_timesteps: int = 10000):
    # PerceptionModule handles its own normalization if pretrained
    perception_fitness = PerceptionModule(pretrained_convnext=True) 
    
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(perception_module=perception_fitness),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])], # MLP layers for PPO policy and value
    )
    
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device="auto")
    model.learn(total_timesteps=total_timesteps)
    return model

def truth_training(env: gym.Env, ae_path: str, total_timesteps: int = 10000):
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(ae_path))
    ae_encoder = autoencoder.encoder
    
    for param in ae_encoder.parameters(): # Freeze encoder parameters
        param.requires_grad = False
    
    # Wrap the frozen AE encoder to match the perception module interface
    perception_truth = PretrainedEncoderWrapper(ae_encoder)
    
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(perception_module=perception_truth),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )
    
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device="auto")
    model.learn(total_timesteps=total_timesteps)
    return model

def train_autoencoder(image_datasets: Dict[str, List[np.ndarray]], epochs: int = 10, batch_size: int = 32, ae_save_path="autoencoder.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Combine all images for training
    all_images_np = [img for cat_imgs in image_datasets.values() for img in cat_imgs]
    # Convert to tensors and scale to [0,1]
    # Taking only a subset for faster example training
    dataset_tensors = torch.stack([transforms.ToTensor()(Image.fromarray(img.astype(np.uint8))) for img in all_images_np[:1000]]).to(device) 

    dataloader = torch.utils.data.DataLoader(dataset_tensors, batch_size=batch_size, shuffle=True)
    
    print(f"Training Autoencoder with {len(dataset_tensors)} images on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs in dataloader: # inputs are (B, C, H, W)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}: Loss {epoch_loss:.4f}')
    
    torch.save(model.state_dict(), ae_save_path)
    print(f"Autoencoder trained and saved to {ae_save_path}")
    return model

# ========================
# 生存评估函数
# ========================
def evaluate_survival(model: PPO, env: gym.Env, n_episodes: int = 10) -> float:
    total_steps = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_steps = 0
        while not (terminated or truncated):
            action_sb3, _ = model.predict(obs, deterministic=True) # SB3 PPO predict gives action_index
            # The environment step() expects action probabilities.
            # For evaluation, we need to convert PPO's action index to the action_probs format
            # Or, modify PPO to output probs, or adapt env.step()
            # Simplest for eval: assume PPO's action is the one with prob 1.0
            action_probs_for_env = np.zeros(3)
            action_probs_for_env[action_sb3] = 1.0 

            obs, _, terminated, truncated, _ = env.step(action_probs_for_env)
            episode_steps += 1
        total_steps += episode_steps
    return total_steps / n_episodes

# ========================
# 可视化评估工具
# ========================
class ModelVisualizer:
    def __init__(self, perception_module: nn.Module, decision_module: nn.Module):
        self.perception = perception_module
        self.decision = decision_module # PPO's policy_net (mlp_extractor.policy_net)
        
        # --- Hooks and stored data management ---
        # For GradCAM specific target layer
        self.target_layer_hook_activations: Optional[torch.Tensor] = None
        self.target_layer_hook_gradients: Optional[torch.Tensor] = None
        
        # For VBP/GuidedBP (and general hook management)
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.module_to_forward_output: Dict[nn.Module, torch.Tensor] = {}

    def cleanup_hooks(self):
        """Removes all registered hooks and clears stored hook-related data."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        
        self.target_layer_hook_activations = None
        self.target_layer_hook_gradients = None
        self.module_to_forward_output = {}

    # --- Hook callback functions ---
    def _store_forward_output_hook(self, module: nn.Module, input_val: Any, output_val: torch.Tensor):
        """Stores the output of a module during forward pass (for VBP)."""
        self.module_to_forward_output[module] = output_val.detach()

    def _gelu_hook_function_vbp(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
        """Approximated Guided Backpropagation hook for GELU."""
        if module not in self.module_to_forward_output:
            # This can happen if forward pass didn't go through this specific module
            # or if hooks were not registered correctly for the forward pass.
            return None # Or grad_input if we want to allow passthrough

        corresponding_forward_output = self.module_to_forward_output[module]
        # Guided BP logic: only pass gradient if grad_output is positive AND forward output was positive
        # grad_output[0] is the gradient w.r.t. the module's output.
        # torch.clamp(grad_output[0], min=0.0) ensures only positive gradients from above are considered.
        # (corresponding_forward_output > 0).float() ensures neuron was active.
        guided_grad = torch.clamp(grad_output[0], min=0.0) * (corresponding_forward_output > 0).float()
        return (guided_grad,)

    def _relu_hook_function_vbp(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
        """Guided Backpropagation hook for ReLU."""
        if module not in self.module_to_forward_output:
            return None
            
        corresponding_forward_output = self.module_to_forward_output[module]
        guided_grad = torch.clamp(grad_output[0], min=0.0) * (corresponding_forward_output > 0).float()
        return (guided_grad,)

    def _gradcam_activation_hook(self, module: nn.Module, input_val: Any, output_val: torch.Tensor):
        """Stores activations for GradCAM."""
        self.target_layer_hook_activations = output_val.detach()

    def _gradcam_gradient_hook(self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]):
        """Stores gradients for GradCAM."""
        self.target_layer_hook_gradients = grad_output[0].detach()

    # --- Main Visualization Methods ---
    def _register_gradcam_hooks(self):
        """Registers hooks for GradCAM on the target ConvNeXt layer."""
        self.cleanup_hooks() # Clear any previous hooks

        convnext_model = None
        if hasattr(self.perception, 'convnext'): # For PerceptionModule
            convnext_model = self.perception.convnext
        elif hasattr(self.perception, 'encoder'): # For PretrainedEncoderWrapper
            convnext_model = self.perception.encoder
        else:
            raise TypeError("Perception module is of an unknown type for GradCAM hook registration.")

        try:
            # Target the output of the last stage in ConvNeXt features
            target_layer = convnext_model.features[-1] 
            handle_fwd = target_layer.register_forward_hook(self._gradcam_activation_hook)
            handle_bwd = target_layer.register_full_backward_hook(self._gradcam_gradient_hook)
            self._hook_handles.extend([handle_fwd, handle_bwd])
        except Exception as e:
            print(f"Error registering GradCAM hooks: {e}. GradCAM might not work correctly.")
            self.cleanup_hooks() # Ensure partial hooks are removed
    
    def activation_maximization(self, action_idx: int, lr: float = 0.1, steps: int = 200, num_views_for_am = NUM_VIEWS) -> np.ndarray:
        self.perception.eval()
        self.decision.eval()
        print(f"Starting AM for action {action_idx}...")

        # Optimizes a single (1,3,H,W) image, assuming it's one of the N_VIEWS inputs
        # The perception module will process it as (1, 1, C, H, W) effectively
        # Then its features are replicated for the decision module.
        optimized_image_tensor = torch.rand(1, 3, 100, 100, requires_grad=True, device=next(self.perception.parameters()).device)
        optimizer = optim.Adam([optimized_image_tensor], lr=lr, weight_decay=1e-4)

        for i in range(steps):
            optimizer.zero_grad()
            
            # Clamp and ensure image is in [0,1] range for perception module
            current_image_0_1 = torch.clamp(optimized_image_tensor, 0.0, 1.0)
            
            # Perception module expects (B, N_VIEWS, C, H, W)
            # We form an input where one view is the optimized image, others could be neutral (e.g., gray)
            # For simplicity in AM: assume the optimized image is so dominant it works if it's just one view.
            # The self.perception here is the SB3 model's feature extractor's perception part.
            # It expects (B, N_VIEWS, C, H, W) format.
            # So, let's treat the optimized image as if it's all N_VIEWS for AM purposes.
            multi_view_input = current_image_0_1.repeat(1, num_views_for_am, 1, 1, 1).squeeze(0) # (N_VIEWS, C, H, W)
            multi_view_input = multi_view_input.unsqueeze(0) # (1, N_VIEWS, C, H, W)


            # Get features from perception ( (1, N_VIEWS, feat_dim) )
            features_per_view = self.perception(multi_view_input)
            # Flatten for decision module ( (1, N_VIEWS * feat_dim) )
            flat_features = features_per_view.view(1, -1)
            
            action_distribution = self.decision(flat_features) # decision is PPO's policy_net
            
            loss = -action_distribution[0, action_idx] # Maximize prob of this action
            
            # Add some regularization to the image (e.g., total variation)
            loss += 0.0001 * torch.sum(torch.abs(current_image_0_1[:, :, :, :-1] - current_image_0_1[:, :, :, 1:])) + \
                    0.0001 * torch.sum(torch.abs(current_image_0_1[:, :, :-1, :] - current_image_0_1[:, :, 1:, :]))

            loss.backward()
            optimizer.step()
            if i % (steps // 10) == 0:
                 print(f"AM step {i}, loss {loss.item()}")

        final_image_0_1 = torch.clamp(optimized_image_tensor.detach(), 0.0, 1.0)
        generated_np = final_image_0_1.squeeze().permute(1, 2, 0).cpu().numpy()
        return (generated_np * 255).astype(np.uint8)

    def grad_cam(self, obs_tensor_0_1: torch.Tensor, action_idx: int, target_view_idx: int = 0) -> Optional[np.ndarray]:
        # obs_tensor_0_1 is (1, N_VIEWS, C, H, W), scaled [0,1]
        # target_view_idx specifies which of the N_VIEWS to generate GradCAM for.
        self.perception.eval()
        self.decision.eval()
        self._register_hooks_for_target_layer() # Ensure hooks are on the correct layer
        
        if not self._hook_handles: # Check if hooks were successfully registered
            return None
        
        
        obs_tensor_0_1.requires_grad_(True)
        
        # Forward pass
        # self.perception is the perception module from the SB3 agent (e.g., PerceptionModule or PretrainedEncoderWrapper)
        features_per_view = self.perception(obs_tensor_0_1) # (1, N_VIEWS, feat_dim)
        flat_features = features_per_view.view(1, -1)       # (1, N_VIEWS * feat_dim)
        
        # self.decision is the PPO's policy_net
        action_distribution = self.decision(flat_features) # (1, num_actions)
        
        # Backward pass for the target action
        self.perception.zero_grad() # Zero grads for perception module's ConvNeXt
        if self.decision.parameters(): # Also zero grads for decision MLP if it has params
            self.decision.zero_grad()

        score = action_distribution[0, action_idx]
        score.backward(retain_graph=True)

        if self.target_layer_hook_activations is None or self.target_layer_hook_gradients is None:
            print("GradCAM: Activations or gradients not captured. Hooks might not be set correctly.")
            return None

        # Activations/Gradients are from the ConvNeXt internal layer, shape (N_VIEWS_eff, C_feat, H_feat, W_feat)
        # N_VIEWS_eff is batch_size * num_views from the perception module's internal reshaping. Here batch_size=1.
        activations_all_views = self.target_layer_hook_activations # (NUM_VIEWS, C_feat, H_feat, W_feat)
        gradients_all_views = self.target_layer_hook_gradients     # (NUM_VIEWS, C_feat, H_feat, W_feat)

        # Select the specific view
        activations_target_view = activations_all_views[target_view_idx] # (C_feat, H_feat, W_feat)
        gradients_target_view = gradients_all_views[target_view_idx]   # (C_feat, H_feat, W_feat)
        
        # Compute weights (alpha_k)
        pooled_gradients = torch.mean(gradients_target_view, dim=[1, 2]) # (C_feat)
        
        # Weight activations
        for i in range(activations_target_view.shape[0]): # Loop over channels
            activations_target_view[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations_target_view, dim=0).cpu().numpy() # (H_feat, W_feat)
        heatmap = np.maximum(heatmap, 0) # ReLU
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap) # Normalize
        
        # Resize to original image size
        original_h, original_w = obs_tensor_0_1.shape[-2:]
        heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
        return heatmap_resized
    
    def visual_back_prop(self, obs_tensor_0_1: torch.Tensor, target_view_idx: int = 0) -> Optional[np.ndarray]:
        """
        Generates a VisualBackProp (Guided Backpropagation style) saliency map.
        Shows general input patterns contributing to the perception module's features for a view.
        """
        self.perception.eval()
        self.cleanup_hooks() # Clears all hooks and stored data (module_to_forward_output too)

        convnext_model = None
        if hasattr(self.perception, 'convnext'):
            convnext_model = self.perception.convnext
        elif hasattr(self.perception, 'encoder'):
            convnext_model = self.perception.encoder
        else:
            print("VBP: Perception module type not recognized.")
            return None

        # Register VBP hooks on all GELU/ReLU layers in the ConvNeXt model
        for module_name, module in convnext_model.named_modules():
            if isinstance(module, nn.GELU):
                self._hook_handles.append(module.register_forward_hook(self._store_forward_output_hook))
                self._hook_handles.append(module.register_full_backward_hook(self._gelu_hook_function_vbp))
            elif isinstance(module, nn.ReLU): # Fallback for any ReLUs
                self._hook_handles.append(module.register_forward_hook(self._store_forward_output_hook))
                self._hook_handles.append(module.register_full_backward_hook(self._relu_hook_function_vbp))
        
        if not self._hook_handles:
            print("VBP: No suitable activation layers (GELU/ReLU) found to hook in ConvNeXt.")
            return None # cleanup_hooks already called, so state is clean.

        # Prepare input image: clone, detach, and set requires_grad
        input_img_for_vbp = obs_tensor_0_1.clone().detach().requires_grad_(True)
        
        # --- Forward pass ---
        # This single forward pass will:
        # 1. Populate self.module_to_forward_output via the forward hooks.
        # 2. Give us features_per_view to backpropagate from.
        features_per_view = self.perception(input_img_for_vbp) # Output: (B, N_VIEWS, feature_dim)
        
        # --- Backward pass ---
        self.perception.zero_grad() # Zero gradients for the perception model parameters
        if input_img_for_vbp.grad is not None:
            input_img_for_vbp.grad.data.zero_()

        # Target for backpropagation: Sum of features for the target_view_idx.
        # This gives a "general pattern" for that view's features.
        target_features_for_bp = features_per_view[0, target_view_idx, :] # Shape: (feature_dim)
        
        # Gradient for backward pass (sum of features -> gradient of 1 for each feature)
        grad_outputs_for_bp = torch.ones_like(target_features_for_bp)
        
        target_features_for_bp.backward(gradient=grad_outputs_for_bp, retain_graph=False)

        # --- Retrieve and process gradient on the input image ---
        saliency_grad = input_img_for_vbp.grad
        if saliency_grad is None:
            print("VBP: Gradient for input image was not computed.")
            # Hooks will be cleaned up by the next call to a viz method or external `visualizer.cleanup_hooks()`
            return None

        # Take absolute value of gradients for the target view
        saliency_abs = saliency_grad.data.abs() # Shape: (1, N_VIEWS, C, H, W)
        saliency_target_view_abs = saliency_abs[0, target_view_idx, :, :, :] # Shape: (C, H, W)
        
        saliency_target_view_np = saliency_target_view_abs.cpu().numpy()
        
        # Normalize: max across channels, then scale to [0, 255]
        saliency_map = np.max(saliency_target_view_np, axis=0) # Shape: (H, W)
        if np.max(saliency_map) > 0:
            saliency_map /= np.max(saliency_map)
        saliency_map_uint8 = (saliency_map * 255).astype(np.uint8)
        
        # Hooks will be cleaned up by the next call to a viz method or an explicit call to self.cleanup_hooks().
        return saliency_map_uint8
    def cleanup_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


# ========================
# 可视化评估函数
# ========================
def visualize_truth_evaluation(model: PPO, env: gym.Env, n_samples: int = 3, run_name="default"):
    print(f"\nPerforming Truth visualization evaluation for {run_name}...")
    
    # Extract perception and decision modules from the PPO model
    # The feature extractor itself contains the perception_module
    feature_extractor = model.policy.features_extractor
    if not hasattr(feature_extractor, 'perception'):
        print("Error: PPO model's feature_extractor does not have a 'perception' attribute.")
        return
    
    perception_module_from_agent = feature_extractor.perception
    # The decision module is the policy network part of the MLP extractor
    decision_module_from_agent = model.policy.mlp_extractor.policy_net

    visualizer = ModelVisualizer(perception_module_from_agent, decision_module_from_agent)
    
    action_names = ['Escape', 'Eat', 'Wander']

    # 1. Activation Maximization
    print("Generating Activation Maximization visualizations...")
    for action_idx, action_name in enumerate(action_names):
        am_image_np = visualizer.activation_maximization(action_idx)
        plt.figure(figsize=(5, 5))
        plt.imshow(am_image_np)
        plt.title(f'AM ({run_name}): {action_name}')
        plt.axis('off')
        plt.savefig(f'am_{run_name}_{action_name.lower()}.png')
        plt.close()

    # 2. Saliency Maps (GradCAM)
    print("Generating Saliency Maps (GradCAM)...")
    obs_np, _ = env.reset() # obs_np is (N_VIEWS, H, W, C) uint8
    for i in range(n_samples):
        # Convert current observation to tensor, scale to [0,1]
        # (N_VIEWS, H, W, C) -> (1, N_VIEWS, C, H, W)
        obs_tensor_0_1 = torch.tensor(obs_np, dtype=torch.float32, device=model.device).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        
        # Get action from model (PPO model.predict expects the numpy obs)
        action_idx_sb3, _ = model.predict(obs_np, deterministic=True)
        action_name = action_names[action_idx_sb3]

        # Generate GradCAM for the first view (e.g., "front" view)
        target_view_for_gradcam = 0 # 0: Front, 1: Left, etc.
        grad_cam_map = visualizer.grad_cam(obs_tensor_0_1, action_idx_sb3, target_view_idx=target_view_for_gradcam)
        
        input_image_to_show_np = obs_np[target_view_for_gradcam] # (H, W, C) uint8

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image_to_show_np)
        plt.title(f'Input View ({run_name}, Sample {i+1})\nAction: {action_name}')
        plt.axis('off')
        
        if grad_cam_map is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(input_image_to_show_np, alpha=0.7)
            plt.imshow(grad_cam_map, cmap='jet', alpha=0.3)
            plt.title('GradCAM')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'saliency_{run_name}_sample_{i+1}.png')
        plt.close()
        
        action_probs_for_env = np.zeros(3)
        action_probs_for_env[action_idx_sb3] = 1.0
        obs_np, _, terminated, truncated, _ = env.step(action_probs_for_env)
        if terminated or truncated:
            obs_np, _ = env.reset()
            if i + 1 >= n_samples: break # Avoid reset if last sample

    # 3. VisualBackProp
    print("Generating VisualBackProp visualizations...")
    # Get a fresh observation if needed, or reuse
    # obs_np, _ = env.reset() # current obs_np is from end of GradCAM loop
    
    for i in range(n_samples): # Use n_samples or a different number for VBP
        # Ensure obs_np is current for this iteration
        if i > 0 or not ('obs_np' in locals() and obs_np is not None): # if not first iter or obs_np is not set
             action_probs_dummy = np.array([0.0, 0.0, 1.0]) # e.g., Wander
             obs_np, _, terminated, truncated, _ = env.step(action_probs_dummy)
             if terminated or truncated:
                 obs_np, _ = env.reset()
                 if i + 1 >= n_samples: break # Avoid issues if last sample leads to reset

        obs_tensor_vbp_0_1 = torch.tensor(obs_np, dtype=torch.float32, device=model.device).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        
        target_view_for_vbp = 0 # e.g., Front view
        vbp_map = visualizer.visual_back_prop(obs_tensor_vbp_0_1, target_view_idx=target_view_for_vbp)
        
        input_image_np_vbp = obs_np[target_view_for_vbp] # (H, W, C) uint8 for plotting

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image_np_vbp)
        plt.title(f'Input View ({run_name}, VBP Sample {i+1})')
        plt.axis('off')

        if vbp_map is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(vbp_map, cmap='gray') # VBP typically shown in grayscale
            plt.title('VisualBackProp')
            plt.axis('off')
        else:
            plt.subplot(1,2,2)
            plt.text(0.5, 0.5, "VBP Failed", ha='center', va='center')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'vbp_{run_name}_sample_{i+1}.png')
        # plt.show() # Optional: show plot interactively
        plt.close()
        
        if i + 1 >= n_samples and (terminated or truncated): # Check if loop should break after reset
            break

    visualizer.cleanup_hooks() # Important to remove hooks after use
    print(f"Truth visualization for {run_name} complete.")


# ========================
# 主工作流程
# ========================
def main():
    # Create mock image datasets
    image_datasets_mock = {
        'predator': [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(20)], # Smaller for faster testing
        'food': [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(20)],
        'environment': [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(20)],
    }
    
    env = SurvivalGameEnv(image_datasets_mock)
    
    # Pretrain Autoencoder
    print("Pretraining Autoencoder...")
    autoencoder_model = train_autoencoder(image_datasets_mock, epochs=2, ae_save_path="autoencoder.pth") # Short epochs for test
    
    # Training parameters
    TRAIN_TIMESTEPS = 2000 # Very short for testing, increase for real training (e.g., 50000+)

    # Train Fitness model
    print("\nTraining Fitness model...")
    fitness_model = fitness_training(env, total_timesteps=TRAIN_TIMESTEPS)
    fitness_model.save("ppo_fitness_model")
    
    # Train Truth model
    print("\nTraining Truth model...")
    truth_model = truth_training(env, ae_path="autoencoder.pth", total_timesteps=TRAIN_TIMESTEPS)
    truth_model.save("ppo_truth_model")
    
    # Load models if needed (example)
    # fitness_model = PPO.load("ppo_fitness_model", env=env)
    # truth_model = PPO.load("ppo_truth_model", env=env)

    # Evaluate models
    print("\nEvaluating models for survival...")
    fitness_score = evaluate_survival(fitness_model, env, n_episodes=5)
    truth_score = evaluate_survival(truth_model, env, n_episodes=5)
    
    print(f"\nResults:")
    print(f"Fitness Model Average Survival: {fitness_score:.1f} steps")
    print(f"Truth Model Average Survival: {truth_score:.1f} steps")

    # Truth visualization evaluation
    visualize_truth_evaluation(fitness_model, env, run_name="FitnessModel")
    visualize_truth_evaluation(truth_model, env, run_name="TruthModel")

    env.close()

if __name__ == "__main__":
    main()