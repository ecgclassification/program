import xlrd
import xlwt
import numpy as np
import matplotlib.pyplot as plt
import math as cos
from scipy.fftpack import dct

file_location = "/home/pi/Desktop/ECG/normal.xls"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for j in range (0,200):
  x= [first_sheet.cell_value(j,i) for i in range (180)]
  a=dct(x)
  print a



