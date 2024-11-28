import tkinter as tk
from tkinter import messagebox
import random
import threading
import time

class GameObject:
    def __init__(self, x, y, health, damage, speed=0):
        self.x = x
        self.y = y
        self.health = health
        self.damage = damage
        self.speed = speed
        self.alive = True

class Plant(GameObject):
    def __init__(self, x, y, health, damage, cost, plant_type):
        super().__init__(x, y, health, damage)
        self.cost = cost
        self.type = plant_type
        self.cooldown = False
        
class Peashooter(Plant):
    def __init__(self, x, y):
        super().__init__(x, y, health=100, damage=20, cost=100, plant_type="豌豆射手")
        self.symbol = "🌱"
        
class IceShooter(Plant):
    def __init__(self, x, y):
        super().__init__(x, y, health=100, damage=15, cost=175, plant_type="冰豌豆")
        self.symbol = "❄️"
        # 减速效果
        self.slow_factor = 0.5
        
class Sunflower(Plant):
    def __init__(self, x, y):
        super().__init__(x, y, health=80, damage=0, cost=50, plant_type="向日葵")
        self.symbol = "🌻"
        self.sun_production = 25
        
class BombPlant(Plant):
    def __init__(self, x, y):
        super().__init__(x, y, health=150, damage=100, cost=150, plant_type="爆炸草")
        self.symbol = "💥"
        self.explosion_range = 2

class HealPlant(Plant):
    def __init__(self, x, y):
        super().__init__(x, y, health=100, damage=0, cost=200, plant_type="治疗花")
        self.symbol = "💚"
        self.heal_amount = 30
        self.heal_range = 1

class Zombie(GameObject):
    def __init__(self, x, y, health=100, damage=10, speed=1):
        super().__init__(x, y, health, damage, speed)
        self.symbol = "🧟"
        self.slowed = False

class Game:
    def __init__(self, master):
        self.master = master
        self.master.title("创新植物大战僵尸")
        
        # 游戏参数
        self.rows = 5
        self.cols = 9
        self.cell_size = 60
        self.sun_energy = 100
        self.game_over = False
        
        # 游戏元素
        self.plants = []
        self.zombies = []
        self.buttons = []
        self.selected_plant = None
        
        # 创建游戏界面
        self.create_gui()
        
        # 启动游戏循环
        self.start_game_loop()
        
    def create_gui(self):
        # 创建顶部信息栏
        info_frame = tk.Frame(self.master)
        info_frame.pack()
        
        self.sun_label = tk.Label(info_frame, text=f"阳光: {self.sun_energy}")
        self.sun_label.pack(side=tk.LEFT, padx=10)
        
        # 创建植物选择按钮
        plants_frame = tk.Frame(self.master)
        plants_frame.pack()
        
        plant_types = [
            ("豌豆射手 (100)", Peashooter),
            ("冰豌豆 (175)", IceShooter),
            ("向日葵 (50)", Sunflower),
            ("爆炸草 (150)", BombPlant),
            ("治疗花 (200)", HealPlant)
        ]
        
        for name, plant_class in plant_types:
            btn = tk.Button(plants_frame, text=name,
                          command=lambda p=plant_class: self.select_plant(p))
            btn.pack(side=tk.LEFT, padx=5)
        
        # 创建游戏板
        self.game_frame = tk.Frame(self.master)
        self.game_frame.pack()
        
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                btn = tk.Button(self.game_frame, width=4, height=2,
                              command=lambda r=i, c=j: self.place_plant(r, c))
                btn.grid(row=i, column=j)
                row.append(btn)
            self.buttons.append(row)
            
    def select_plant(self, plant_class):
        self.selected_plant = plant_class
        
    def place_plant(self, row, col):
        if not self.selected_plant:
            return
            
        # 检查是否已有植物
        for plant in self.plants:
            if plant.x == col and plant.y == row:
                return
                
        # 检查阳光是否足够
        temp_plant = self.selected_plant(0, 0)
        if self.sun_energy >= temp_plant.cost:
            plant = self.selected_plant(col, row)
            self.plants.append(plant)
            self.sun_energy -= plant.cost
            self.update_gui()
            
    def update_gui(self):
        # 更新阳光数量
        self.sun_label.config(text=f"阳光: {self.sun_energy}")
        
        # 清空所有格子
        for row in self.buttons:
            for button in row:
                button.config(text="")
                
        # 显示植物
        for plant in self.plants:
            if plant.alive:
                self.buttons[plant.y][plant.x].config(text=plant.symbol)
                
        # 显示僵尸
        for zombie in self.zombies:
            if zombie.alive and 0 <= zombie.x < self.cols:
                self.buttons[zombie.y][zombie.x].config(text=zombie.symbol)
                
    def spawn_zombie(self):
        if not self.game_over:
            row = random.randint(0, self.rows-1)
            zombie = Zombie(self.cols-1, row)
            self.zombies.append(zombie)
            self.update_gui()
            
    def start_game_loop(self):
        def game_loop():
            while not self.game_over:
                # 生产阳光
                for plant in self.plants:
                    if isinstance(plant, Sunflower) and plant.alive:
                        self.sun_energy += plant.sun_production
                
                # 植物攻击
                for plant in self.plants:
                    if not plant.alive:
                        continue
                        
                    if isinstance(plant, (Peashooter, IceShooter)):
                        # 直线攻击
                        for zombie in self.zombies:
                            if zombie.alive and zombie.y == plant.y and zombie.x > plant.x:
                                zombie.health -= plant.damage
                                if isinstance(plant, IceShooter):
                                    zombie.slowed = True
                                break
                                
                    elif isinstance(plant, BombPlant):
                        # 范围攻击
                        for zombie in self.zombies:
                            if zombie.alive and abs(zombie.y - plant.y) <= plant.explosion_range and \
                               abs(zombie.x - plant.x) <= plant.explosion_range:
                                zombie.health -= plant.damage
                                
                    elif isinstance(plant, HealPlant):
                        # 治疗周围植物
                        for other_plant in self.plants:
                            if other_plant.alive and other_plant != plant and \
                               abs(other_plant.y - plant.y) <= plant.heal_range and \
                               abs(other_plant.x - plant.x) <= plant.heal_range:
                                other_plant.health = min(100, other_plant.health + plant.heal_amount)
                
                # 僵尸移动和攻击
                for zombie in self.zombies:
                    if not zombie.alive:
                        continue
                        
                    # 检查前方是否有植物
                    plant_in_front = None
                    for plant in self.plants:
                        if plant.alive and plant.y == zombie.y and plant.x == zombie.x - 1:
                            plant_in_front = plant
                            break
                            
                    if plant_in_front:
                        plant_in_front.health -= zombie.damage
                        if plant_in_front.health <= 0:
                            plant_in_front.alive = False
                    else:
                        # 移动
                        speed = zombie.speed * (0.5 if zombie.slowed else 1)
                        zombie.x -= speed
                        
                    # 检查是否到达最左边
                    if zombie.x < 0:
                        self.game_over = True
                        messagebox.showinfo("游戏结束", "僵尸进入了你的家！")
                        break
                
                # 移除死亡的物体
                self.plants = [p for p in self.plants if p.alive]
                self.zombies = [z for z in self.zombies if z.alive and z.health > 0]
                
                # 更新界面
                self.update_gui()
                time.sleep(0.5)
                
        # 在新线程中运行游戏循环
        game_thread = threading.Thread(target=game_loop)
        game_thread.daemon = True
        game_thread.start()
        
        # 定期生成僵尸
        def spawn_zombies():
            if not self.game_over:
                self.spawn_zombie()
                self.master.after(5000, spawn_zombies)
                
        self.master.after(5000, spawn_zombies)

if __name__ == "__main__":
    root = tk.Tk()
    game = Game(root)
    root.mainloop()
