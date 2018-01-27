import xlrd
import xlwt
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import math as cos
from scipy.fftpack import dct 


file_location = "I:\project\raspberry_project\ecg\normal.xls"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
#book = xlwt.Workbook(encoding="utf-8")
#sheet1 = book.add_sheet("Sheet 1")
for j in range (0,200):
    list =[first_sheet.cell_value(j,i) for i in range (180)]
    #a=dct(x)
    print list
for i in range(200):
    first_sheet.write(i,179,'1')
    
#list.append('1');

  
#a=(:,"1")
#print x
file_location = "I:\project\raspberry_project\ecg\trial.xls"#sunny
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
                   
'''
for j in range (0,200):
  b=(:,"0")
  print b
  

for j in range (0,30):
  b=dct(a)
  print b
  '''




