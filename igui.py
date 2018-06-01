#all the libraries needed to run the code
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
   import tkinter as tk

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import xlrd
import numpy as np

from tkinter import *
from tkinter import messagebox
from math import *

#from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

def write_slogan():
    print("the patient is normal")


    
def hello():
   q =' sunny  '
   messagebox.showinfo("RESULTS", q )
   msg = tk.Label(root, compound =tk.LEFT, padx = 10, text="sunny")
   msg.config(bg='lightblue', font=('times', 20, 'italic'))
   msg.place(x = 550,y =100 ) 
       
def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate   

               
def PVC():
    img = PhotoImage(file
    ="PVC.gif")
    canvas.create_image(5,-5, anchor=NW, image=img)
    root.mainloop()
    
def NORMAL():
   img3 = PhotoImage(file="normal.gif")
   canvas.create_image(1,-5, anchor=NW, image=img3)
   canvas.pack(expand = YES, fill = BOTH)
   root.mainloop()
   
def FUSION():
   img4 = PhotoImage(file="fusion.gif")
   canvas.create_image(30,-5, anchor=NW, image=img4)
   root.mainloop()

root = tk.Tk()
frame = tk.Frame(root)  
frame.pack()


root.wm_title("ECG SIGNAL CLASSIFIER")

canvas_width = 800
canvas_height =480



canvas = Canvas(root, width=canvas_width, height=canvas_height,bg = 'lightblue')
canvas.pack(expand = YES, fill = BOTH)
        

img2 = PhotoImage(file="viit.gif")
canvas.create_image(50,280, anchor=NW, image=img2)

img6 = PhotoImage(file="ecg.gif")
canvas.create_image(30,10, anchor=NW, image=img6)

def motion(event):
        print("Mouse position: (%s %s)" % (event.x, event.y))
        return 
 
''' 

msg = tk.Message(root,justify =tk.LEFT, text = explanation)
msg.config(bg='lightblue', font=('times', 16, 'italic'))
msg.place(x = 240,y = 550)
msg.bind('<Motion>',motion)
'''
#tk.Label(root,justify=LEFT,compound = LEFT, text=txt, fg = "darkblue", bg = "lightblue", font = "Verdana 80 italic").place(x=200,y=150)
        
def donothing():
   x = 0
 
#root = Tk()
''' 
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New", command=PVC)
filemenu.add_command(label="Open", command=donothing)
filemenu.add_command(label="Save", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=_quit)
menubar.add_cascade(label="File", menu=filemenu)
 
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index", command=donothing)
helpmenu.add_command(label="About...", command=donothing)
menubar.add_cascade(label="Help", menu=helpmenu)
 
root.config(menu=menubar)
               
'''                  
explanation = """Project Name:
 Multi-Class ECG signal Clasification
 shraddha Kapse 411028
 Sunny C.J Francis 413017
 Sanket Naik 412018"""
 
msg = tk.Label(root, 
              compound =tk.LEFT,
              padx = 10, 
              text=explanation)
msg.config(bg='lightblue', font=('times', 12, 'italic'))
msg.place(x = 200,y = 300) 



msg2 = tk.Label(root, 
              compound =tk.LEFT,
              padx = 10, 
              text='Acurracy:')
msg2.config(bg='lightblue', font=('times', 20, 'italic'))
msg2.place(x = 550,y = 20) 

msg3 = tk.Label(root, 
              compound =tk.LEFT,
              padx = 10, 
              text='Confusion Matrix:')
msg3.config(bg='lightblue', font=('times', 12, 'italic'))
msg3.place(x = 550,y = 80)            


'''
logo = tk.PhotoImage(file="viit.gif")
w1 = tk.Label(root, image=logo).pack(side="left")
'''
'''
def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)
canvas.mpl_connect('key_press_event', on_key_event)
'''    
  
       
                           
 #various buttons                          
button = tk.Button(master=root, text='CLEAR',fg="red", command=root.quit)
button.place(x = 320,y = 400) 

button2 = tk.Button(master=root, padx = 10, text='RESULT', command=hello)
button2.place(x = 600,y = 300) 

button3 = tk.Button(master=root, padx = 10, text='PVC', command=PVC)
button3.place(x = 480,y = 350)

button4 = tk.Button(master=root, padx = 10, text='NORMAL', command=NORMAL)
button4.place(x = 560,y = 350)

button5 = tk.Button(master=root, padx = 10, text='FUSION', command=FUSION)
button5.place(x = 660,y = 350)

mainloop()

