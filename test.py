import tkinter as tk
import random
from tkinter import messagebox

class Minesweeper:
    def __init__(self, rows, cols, mines):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.revealed = [[False for _ in range(cols)] for _ in range(rows)]
        mine_positions = random.sample(range(rows * cols), mines)
        for pos in mine_positions:
            row = pos // cols
            col = pos % cols
            self.board[row][col] = -1
        for row in range(rows):
            for col in range(cols):
                if self.board[row][col] == -1:
                    continue
                for r in range(max(0, row - 1), min(rows, row + 2)):
                    for c in range(max(0, col - 1), min(cols, col + 2)):
                        if self.board[r][c] == -1:
                            self.board[row][col] += 1

    def reveal(self, row, col):
        if not self.is_valid(row, col):
            return
        if self.revealed[row][col]:
            return
        self.revealed[row][col] = True
        if self.board[row][col] == -1:
            return -1  # 地雷
        elif self.board[row][col] > 0:
            return self.board[row][col]
        else:
            for r in range(max(0, row - 1), min(self.rows, row + 2)):
                for c in range(max(0, col - 1), min(self.cols, col + 2)):
                    self.reveal(r, c)
            return 0

    def is_valid(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

class MinesweeperGUI:
    def __init__(self, root, rows, cols, mines):
        self.root = root
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.board = Minesweeper(rows, cols, mines)
        self.buttons = [[None for _ in range(cols)] for _ in range(rows)]
        frame = tk.Frame(root)
        frame.pack()
        for row in range(rows):
            for col in range(cols):
                button = tk.Button(frame, text="*", width=2, height=1,
                                   command=lambda r=row, c=col: self.on_click(r, c))
                button.grid(row=row, column=col)
                self.buttons[row][col] = button

    def on_click(self, row, col):
        if self.board.revealed[row][col]:
            return
        result = self.board.reveal(row, col)
        if result == -1:
            # 地雷
            self.buttons[row][col].config(text="*", bg="red")
            # 显示所有地雷
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.board.board[r][c] == -1:
                        self.buttons[r][c].config(text="*", bg="red")
            messagebox.showinfo("游戏结束", "你踩到地雷了！")
        else:
            # 更新当前格子
            if result > 0:
                self.buttons[row][col].config(text=str(result), bg="white")
            else:
                self.buttons[row][col].config(text="", bg="white")
                # 递归翻开周围格子
                for r in range(max(0, row - 1), min(self.rows, row + 2)):
                    for c in range(max(0, col - 1), min(self.cols, col + 2)):
                        if not self.board.revealed[r][c]:
                            res = self.board.reveal(r, c)
                            if res == 0:
                                self.buttons[r][c].config(text="", bg="white")
                            elif res > 0:
                                self.buttons[r][c].config(text=str(res), bg="white")
            # 更新所有已翻开的格子
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.board.revealed[r][c]:
                        if self.board.board[r][c] == -1:
                            self.buttons[r][c].config(text="*", bg="red")
                        elif self.board.board[r][c] > 0:
                            self.buttons[r][c].config(text=str(self.board.board[r][c]), bg="white")
                        else:
                            self.buttons[r][c].config(text="", bg="white")
            self.root.update()
        if self.check_win():
            messagebox.showinfo("游戏结束", "你赢了！")

    def check_win(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board.board[row][col] != -1 and not self.board.revealed[row][col]:
                    return False
        return True

if __name__ == "__main__":
    root = tk.Tk()
    root.title("扫雷游戏")
    game = MinesweeperGUI(root, 10, 10, 10)
    root.mainloop()