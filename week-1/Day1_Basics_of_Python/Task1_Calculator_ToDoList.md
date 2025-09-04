from tkinter import *

def add_task():
    task = entry.get()
    if task != "":
        listbox.insert(END, task)
        entry.delete(0, END)

def complete_task():
    selected = listbox.curselection()
    if selected:
        listbox.delete(selected)

def delete_task():
    listbox.delete(0, END)

root = Tk()
root.title("To-Do List")
root.geometry("300x400")

Label(root, text="My To-Do List", font=("Arial", 18)).pack(pady=10)

entry = Entry(root, font=("Arial", 14))
entry.pack(pady=10)

add_btn = Button(root, text="Add Task", font=("Arial", 12), command=add_task)
add_btn.pack(pady=5)

complete_btn = Button(root, text="Complete Task", font=("Arial", 12), command=complete_task)
complete_btn.pack(pady=5)

delete_btn = Button(root, text="Delete All Tasks", font=("Arial", 12), command=delete_task)
delete_btn.pack(pady=5)

listbox = Listbox(root, font=("Arial", 14), width=30, height=10)
listbox.pack(pady=10)

root.mainloop()
