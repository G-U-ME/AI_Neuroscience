#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fitness_bit_truth.py

完整实现“Fitness Bit Truth”项目：
- 环境：20×40 像素地图，三类像素（捕食者、食物、环境），可视化支持实时和回放
- 模型1：端到端 CNN+MLP，通过 PPO 训练，CNN+MLP 参数均可更新
- 模型2：Autoencoder 预训练 CNN 固定，仅训练 MLP 部分，用 PPO 优化
- 使用 stable-baselines3 实现 PPO
"""

import os
import random
import argparse
import numpy as np
from PIL import Image
import gym
from gym import spaces
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ====================
# 1. 环境定义
# ====================
class FitnessBitEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data_dir, map_size=(20,40), max_food=50, max_pred=20):
        super().__init__()
        self.map_h, self.map_w = map_size
        self.data_dir = data_dir
        self.max_food = max_food
        self.max_pred = max_pred
        self._load_images()
        self.action_space = spaces.Discrete(3)
        C,H,W=3,32,32
        self.observation_space=spaces.Box(0,255,shape=(4,C,H,W),dtype=np.uint8)
        self.reset()
    def _load_images(self):
        self.imgs={}
        for cls in ['predator','food','env']:
            pth=os.path.join(self.data_dir,cls)
            self.imgs[cls]=[]
            for fn in os.listdir(pth):
                if fn.lower().endswith(('.png','.jpg','.jpeg')):
                    img=Image.open(os.path.join(pth,fn)).convert('RGB').resize((32,32))
                    self.imgs[cls].append(img)
            assert self.imgs[cls], f"{cls} no images"
    def reset(self):
        self.grid=np.zeros((self.map_h,self.map_w),dtype=np.int8)
        for _ in range(self.max_food):
            x,y=random.randrange(self.map_h),random.randrange(self.map_w)
            self.grid[x,y]=1
        for _ in range(self.max_pred):
            while True:
                x,y=random.randrange(self.map_h),random.randrange(self.map_w)
                if self.grid[x,y]==0:
                    self.grid[x,y]=2;break
        while True:
            ax,ay=random.randrange(self.map_h),random.randrange(self.map_w)
            if self.grid[ax,ay]==0:
                self.agent_pos=[ax,ay];break
        self.steps=0
        return self._get_obs()
    def _get_obs(self):
        obs=[]
        dirs=[(-1,0),(1,0),(0,-1),(0,1)]
        for dx,dy in dirs:
            x,y=self.agent_pos[0]+dx,self.agent_pos[1]+dy
            cls='env'
            if 0<=x<self.map_h and 0<=y<self.map_w:
                cls={0:'env',1:'food',2:'predator'}[self.grid[x,y]]
            obs.append(np.array(random.choice(self.imgs[cls])))
        stack=np.stack(obs,axis=0)
        return stack.transpose(0,3,1,2)
    def step(self,action):
        self.steps+=1
        ax,ay=self.agent_pos
        if action==0:
            coords=np.argwhere(self.grid==2)
            if coords.size:
                d=coords-np.array(self.agent_pos)
                idx=np.argmin(np.linalg.norm(d,axis=1))
                vx,vy=-np.sign(d[idx])
            else: vx,vy=random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        elif action==1:
            coords=np.argwhere(self.grid==1)
            if coords.size:
                d=coords-np.array(self.agent_pos)
                idx=np.argmin(np.linalg.norm(d,axis=1))
                vx,vy=np.sign(d[idx])
            else: vx,vy=random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        else: vx,vy=random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        nx,ny=ax+vx,ay+vy
        if not(0<=nx<self.map_h and 0<=ny<self.map_w):nx,ny=ax,ay
        reward=0.0;cls=self.grid[nx,ny]
        if cls==1:reward+=1.0;self.grid[nx,ny]=0
        elif cls==2:reward-=1.0
        else:reward-=0.01
        self.agent_pos=[nx,ny]
        done=(reward<-0.5) or (self.steps>=500)
        return self._get_obs(),reward,done,{}
    def render(self,mode='human'):
        canvas=Image.new('RGB',(self.map_w*32,self.map_h*32))
        for i in range(self.map_h):
            for j in range(self.map_w):
                cls={0:'env',1:'food',2:'predator'}[self.grid[i,j]]
                canvas.paste(random.choice(self.imgs[cls]),(j*32,i*32))
        ax,ay=self.agent_pos
        overlay=Image.new('RGBA',(32,32),(255,0,0,128))
        canvas.paste(overlay,(ay*32,ax*32),overlay)
        plt.imshow(canvas);plt.axis('off');plt.pause(0.001)

# ====================
# 2. Autoencoder 预训练 CNN
# ====================
class SimpleAE(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,16,4,2,1),nn.ReLU(),
            nn.Conv2d(16,32,4,2,1),nn.ReLU(),
            nn.Flatten(),nn.Linear(32*8*8,latent_dim)
        )
        self.decoder=nn.Sequential(
            nn.Linear(latent_dim,32*8*8),nn.ReLU(),
            nn.Unflatten(1,(32,8,8)),
            nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(16,3,4,2,1),nn.Sigmoid()
        )
    def forward(self,x):return self.decoder(self.encoder(x))

def train_autoencoder(data_dir,epochs=10,batch_size=128,latent_dim=128,device='cpu'):
    ds=datasets.ImageFolder(
        root=os.path.join(data_dir,'env'),
        transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
    )
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True)
    ae=SimpleAE(latent_dim).to(device)
    opt=torch.optim.Adam(ae.parameters(),lr=1e-3)
    for ep in range(epochs):
        tot,cnt=0,0
        for x,_ in loader:
            x=x.to(device);r=ae(x);loss=F.mse_loss(r,x)
            opt.zero_grad();loss.backward();opt.step()
            tot+=loss.item();cnt+=1
        print(f"AE {ep+1}/{epochs}, loss={tot/cnt:.4f}")
    return ae.encoder

# ====================
# 3. 特征提取器
# ====================
class CnnFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,obs_space,features_dim=256,pretrained_encoder=None):
        super().__init__(obs_space,features_dim)
        in_ch=obs_space.shape[0]*3
        layers=[nn.Conv2d(in_ch,32,3,1,1),nn.ReLU(),
                nn.Conv2d(32,64,3,1,1),nn.ReLU(), nn.Flatten(),
                nn.Linear(64*32*32,features_dim),nn.ReLU()]
        if pretrained_encoder:
            layers[0]=pretrained_encoder
        self.cnn=nn.Sequential(*layers)
    def forward(self,obs):
        B=obs.shape[0]
        x=obs.view(B,-1,32,32)
        return self.cnn(x)

# ====================
# 4. 主流程
# ====================
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',required=True)
    p.add_argument('--train_ae',action='store_true')
    p.add_argument('--model',choices=['end2end','frozen_cnn'],required=True)
    p.add_argument('--timesteps',type=float,default=1e5)
    p.add_argument('--visual',action='store_true')
    args=p.parse_args()

    env=FitnessBitEnv(args.data_dir)
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec=DummyVecEnv([lambda:env])

    enc=None
    if args.model=='frozen_cnn':
        assert args.train_ae
        dev='cuda' if torch.cuda.is_available() else 'cpu'
        enc=train_autoencoder(args.data_dir,device=dev)
        for p in enc.parameters():p.requires_grad=False

    policy_kws=dict(
        features_extractor_class=CnnFeatureExtractor,
        features_extractor_kwargs=dict(pretrained_encoder=enc,features_dim=256)
    )
    model=PPO('CnnPolicy',vec,policy_kwargs=policy_kws,verbose=1)
    model.learn(total_timesteps=int(args.timesteps))
    model.save(f"ppo_fbt_{args.model}")

    obs=env.reset()
    for _ in range(200):
        act,_=model.predict(obs,deterministic=True)
        obs,_,done,_=env.step(act)
        if args.visual:env.render()
        if done:obs=env.reset()

if __name__=='__main__':main()
