import xlrd
#import xlwt
import numpy as np
import matplotlib.pyplot as plt
import math as cos
from scipy.fftpack import dct

# normal
a = np.zeros((200,180))
x1 = np.empty((200,183))
file_location = r'feature1.xls'
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for j in range (0,200):
  x= [first_sheet.cell_value(j,i) for i in range (180)]
  a[j,:]=dct(x)
  x= np.append(a[j,:],[1,0,0])
  x1= np.append(x1,[x],axis=0)
  n=np.max(x1)
b =x1.shape
print n
print x1
