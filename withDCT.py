import xlrd
import xlwt
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import math as cos
#from scipy.fftpack import dct


file_location = "C:\Users\DELL\Desktop\ecg\normal.xls"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
#book = xlwt.Workbook(encoding="utf-8")
#sheet1 = book.add_sheet("Sheet 1")
print first_sheet
'''

for j in range (0,200):
    x=[first_sheet.cell_value(j,i) for i in range (180)]
    a=dct(x)
    print a
Xstk=stack()
    m=Xstk.push(x)
    print m

#a=(:,"1")
#print x
file_location = "/home/pi/Desktop/ECG/fusion.xls"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for j in range (200):
    y= [first_sheet.cell_value(j,i) for i in range (180)]
    #print y
  #b=dct(y)
  #print b
for j in range (200):
    x=[first_sheet.cell_value(j,i) for i in range (180)]
    y= [first_sheet.cell_value(j,i) for i in range (180)]
    S=np.vstack((x,y))
    print S                
                

for j in range (0,200):
  b=(:,"0")
  print b
  

for j in range (0,30):
  b=dct(a)
  print b
  '''




