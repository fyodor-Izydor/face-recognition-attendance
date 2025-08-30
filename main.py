from tkinter import *
import os
from tkinter import ttk
from PIL import Image, ImageTk

def FaceRegist():
    os.system("python realtime_face_regist.py")

def Attendance():
    os.system("python realtime_face_recognition.py")

def Help():
    os.system("start \"chrome\" \"main.html\"")

def resize(event):
    # Mengatur ulang posisi tombol saat jendela diubah ukurannya
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=1)
    window.update_idletasks()

window = Tk()
window.title("Attendance")
window.configure(bg="white")
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)
window.bind("<Configure>", resize)

# Create Style
style = ttk.Style()
style.configure('TButton', background='#8B5FBF', foreground='#8B5FBF', font=("Arial", 12, "bold"), relief="raised")

# Path gambar lokal
window.iconbitmap('assets\img\Gunadarma.ico')
image_path = "assets\img\Gunadarma.png"

# Load gambar dari path lokal
image = Image.open(image_path)
image = image.resize((150, 150))  # Menyesuaikan ukuran gambar sesuai kebutuhan
photo = ImageTk.PhotoImage(image)

# Judul dengan gambar
label_title = Label(window, text="Registration and Attendance System", bg="white", fg="#8B5FBF", font=("Arial", 18, "bold"))
label_title.grid(row=0, column=0, columnspan=2, pady=(20, 10))

label_image = Label(window, image=photo, padx=10, pady=10, bg="white")
label_image.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Button Face Registration
button_d1 = ttk.Button(window, text="Face Registration", command=FaceRegist, style='TButton')
button_d1.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

# Button Attendance
button_d2 = ttk.Button(window, text="Attendance", command=Attendance, style='TButton')
button_d2.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="nsew")

# Button Instruction
button_help = ttk.Button(window, text="Help", command=Help, style='TButton')
button_help.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")

# Button Exit
button_quit = ttk.Button(window, text="Exit", command=window.quit, style='TButton')
button_quit.grid(row=4, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")

window.mainloop()
