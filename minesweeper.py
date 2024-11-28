import tkinter as tk
from tkinter import messagebox
import random
import time
import threading

class Minesweeper:
    def __init__(self, master):
        self.master = master
        self.master.title("植物大战僵尸扫雷")
        self.rows = 9
        self.cols = 9
        self.mines = 10
        self.buttons = []
        self.mines_positions = []
        self.game_over = False
        self.flags = set()
        self.revealed = set()
        self.peashooters = set()  # 存储豌豆射手位置
        
        # 创建游戏界面
        self.create_board()
        self.place_mines()
        
        # 添加重新开始按钮
        self.restart_button = tk.Button(master, text="重新开始", command=self.restart_game)
        self.restart_button.grid(row=self.rows, column=0, columnspan=self.cols)
        
    def create_board(self):
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                button = tk.Button(self.master, width=2, height=1, bg='gray75')
                button.bind('<Button-1>', lambda e, row=i, col=j: self.click(row, col))
                button.bind('<Button-3>', lambda e, row=i, col=j: self.flag(row, col))
                button.grid(row=i, column=j)
                row.append(button)
            self.buttons.append(row)
    
    def shoot_pea(self, row, col):
        if not self.game_over and (row, col) in self.peashooters:
            # 向右发射豌豆
            def animate_pea():
                current_col = col + 1
                while current_col < self.cols and not self.game_over:
                    if (row, current_col) in self.mines_positions and (row, current_col) not in self.revealed:
                        self.reveal(row, current_col)  # 击中僵尸
                        break
                    current_col += 1
                    time.sleep(0.1)  # 动画延迟
            
            # 在新线程中运行动画
            threading.Thread(target=animate_pea).start()
    
    def place_mines(self):
        positions = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        self.mines_positions = random.sample(positions, self.mines)
    
    def count_adjacent_mines(self, row, col):
        count = 0
        for i in range(max(0, row-1), min(self.rows, row+2)):
            for j in range(max(0, col-1), min(self.cols, col+2)):
                if (i, j) in self.mines_positions:
                    count += 1
        return count
    
    def reveal(self, row, col):
        if (row, col) in self.revealed or (row, col) in self.flags:
            return
            
        self.revealed.add((row, col))
        if (row, col) in self.mines_positions:
            self.game_over = True
            self.show_all_mines()
            messagebox.showinfo("游戏结束", "僵尸吃掉了你的脑子！")
            return
            
        adjacent = self.count_adjacent_mines(row, col)
        self.buttons[row][col].config(relief="sunken", text=str(adjacent) if adjacent else "", bg='white')
        
        if adjacent == 0:
            for i in range(max(0, row-1), min(self.rows, row+2)):
                for j in range(max(0, col-1), min(self.cols, col+2)):
                    if (i, j) != (row, col):
                        self.reveal(i, j)
        
        if len(self.revealed) + len(self.mines_positions) == self.rows * self.cols:
            messagebox.showinfo("恭喜", "你成功击退了所有僵尸！")
            self.game_over = True
    
    def show_all_mines(self):
        for row, col in self.mines_positions:
            self.buttons[row][col].config(text="🧟", bg="red")
    
    def click(self, row, col):
        if self.game_over:
            return
            
        if (row, col) in self.peashooters:
            self.shoot_pea(row, col)
        elif (row, col) not in self.flags:
            self.reveal(row, col)
    
    def flag(self, row, col):
        if self.game_over:
            return
            
        if (row, col) in self.flags:
            self.flags.remove((row, col))
            self.peashooters.remove((row, col))
            self.buttons[row][col].config(text="")
        elif (row, col) not in self.revealed:
            self.flags.add((row, col))
            self.peashooters.add((row, col))
            self.buttons[row][col].config(text="🌱")
    
    def restart_game(self):
        self.game_over = False
        self.flags.clear()
        self.revealed.clear()
        self.mines_positions.clear()
        self.peashooters.clear()
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.buttons[i][j].config(text="", relief="raised", bg="gray75")
        
        self.place_mines()

if __name__ == "__main__":
    root = tk.Tk()
    game = Minesweeper(root)
    root.mainloop()
