import tkinter as tk
from tkinter import messagebox
import random

class Minesweeper:
    def __init__(self, master):
        self.master = master
        self.master.title("Minesweeper")
        self.rows = 10
        self.cols = 10
        self.mines = 10
        self.buttons = []
        self.mines_positions = []
        self.game_over = False
        self.flags = set()
        self.revealed = set()
        
        # Create buttons
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                button = tk.Button(master, width=2, height=1)
                button.grid(row=i, column=j)
                button.bind('<Button-1>', lambda e, row=i, col=j: self.click(row, col))
                button.bind('<Button-3>', lambda e, row=i, col=j: self.flag(row, col))
                row.append(button)
            self.buttons.append(row)
        
        self.place_mines()

    def place_mines(self):
        positions = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        self.mines_positions = random.sample(positions, self.mines)

    def get_adjacent_mines(self, row, col):
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
        button = self.buttons[row][col]
        
        if (row, col) in self.mines_positions:
            button.configure(text="💣", bg='red')
            self.game_over = True
            self.show_all_mines()
            messagebox.showinfo("Game Over", "You hit a mine!")
            return
        
        mines = self.get_adjacent_mines(row, col)
        if mines > 0:
            button.configure(text=str(mines), relief=tk.SUNKEN)
        else:
            button.configure(text="", relief=tk.SUNKEN)
            # Reveal adjacent cells
            for i in range(max(0, row-1), min(self.rows, row+2)):
                for j in range(max(0, col-1), min(self.cols, col+2)):
                    if (i, j) != (row, col):
                        self.reveal(i, j)
        
        if len(self.revealed) == self.rows * self.cols - len(self.mines_positions):
            messagebox.showinfo("Congratulations!", "You won!")
            self.game_over = True

    def show_all_mines(self):
        for row, col in self.mines_positions:
            self.buttons[row][col].configure(text="💣", bg='red')

    def click(self, row, col):
        if not self.game_over:
            self.reveal(row, col)

    def flag(self, row, col):
        if not self.game_over and (row, col) not in self.revealed:
            if (row, col) in self.flags:
                self.flags.remove((row, col))
                self.buttons[row][col].configure(text="")
            else:
                self.flags.add((row, col))
                self.buttons[row][col].configure(text="🚩")

if __name__ == "__main__":
    root = tk.Tk()
    game = Minesweeper(root)
    root.mainloop()