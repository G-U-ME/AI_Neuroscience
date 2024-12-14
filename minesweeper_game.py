import tkinter as tk
from tkinter import messagebox
import random

class Minesweeper:
    def __init__(self, master):
        self.master = master
        self.master.title("扫雷游戏")
        self.rows = 9
        self.cols = 9
        self.mines = 10
        self.buttons = []
        self.mines_positions = []
        self.game_over = False
        
        # 创建游戏网格
        self.create_board()
        # 放置地雷
        self.place_mines()
        
    def create_board(self):
        # 创建按钮网格
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                button = tk.Button(self.master, width=2, height=1,
                                 command=lambda x=i, y=j: self.click(x, y))
                button.bind('<Button-3>', lambda event, x=i, y=j: self.right_click(event, x, y))
                button.grid(row=i, column=j)
                row.append(button)
            self.buttons.append(row)
    
    def place_mines(self):
        # 随机放置地雷
        positions = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        self.mines_positions = random.sample(positions, self.mines)
    
    def count_adjacent_mines(self, row, col):
        # 计算周围地雷数量
        count = 0
        for i in range(max(0, row-1), min(self.rows, row+2)):
            for j in range(max(0, col-1), min(self.cols, col+2)):
                if (i, j) in self.mines_positions:
                    count += 1
        return count
    
    def click(self, row, col):
        if self.game_over:
            return
            
        button = self.buttons[row][col]
        
        # 如果点击到地雷
        if (row, col) in self.mines_positions:
            button.config(text="💣", bg="red")
            self.game_over = True
            self.show_all_mines()
            messagebox.showinfo("游戏结束", "很遗憾，你踩到地雷了！")
            return
            
        # 显示周围地雷数量
        mines = self.count_adjacent_mines(row, col)
        button.config(state="disabled", relief="sunken")
        
        if mines == 0:
            # 如果周围没有地雷，递归显示周围的格子
            button.config(text="")
            self.reveal_empty_cells(row, col)
        else:
            button.config(text=str(mines))
            
        # 检查是否胜利
        self.check_win()
    
    def right_click(self, event, row, col):
        if self.game_over:
            return
            
        button = self.buttons[row][col]
        if button["state"] != "disabled":
            if button["text"] == "🚩":
                button.config(text="")
            else:
                button.config(text="🚩")
    
    def reveal_empty_cells(self, row, col):
        # 递归显示空白格子
        for i in range(max(0, row-1), min(self.rows, row+2)):
            for j in range(max(0, col-1), min(self.cols, col+2)):
                button = self.buttons[i][j]
                if button["state"] != "disabled":
                    mines = self.count_adjacent_mines(i, j)
                    button.config(state="disabled", relief="sunken")
                    if mines == 0:
                        button.config(text="")
                        self.reveal_empty_cells(i, j)
                    else:
                        button.config(text=str(mines))
    
    def show_all_mines(self):
        # 游戏结束时显示所有地雷
        for row, col in self.mines_positions:
            self.buttons[row][col].config(text="💣", bg="red")
    
    def check_win(self):
        # 检查是否胜利
        unopened = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.buttons[i][j]["state"] != "disabled":
                    unopened += 1
        
        if unopened == self.mines:
            self.game_over = True
            messagebox.showinfo("恭喜", "你赢了！")

def main():
    root = tk.Tk()
    game = Minesweeper(root)
    root.mainloop()

if __name__ == "__main__":
    main()
