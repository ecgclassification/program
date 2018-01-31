'''
DCT and append 1,0,0
'''
import xlrd
#import xlwt
import numpy as np
import matplotlib.pyplot as plt
import math as cos
from scipy.fftpack import dct

# normal
a = np.zeros((200,180))
x1 = np.empty((200,183))
file_location = r'normal.xls'
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for j in range (0,200):
  x= [first_sheet.cell_value(j,i) for i in range (180)]
  a[j,:]=dct(x)
  x= np.append(a[j,:],[1,0,0])
  x1= np.append(x1,[x],axis=0)
b =x1.shape
print b
print"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5"
print "fusion"

#fusion
b = np.zeros((200,180))
y1 = np.empty((200,183))
file_location = r'I:\Git Hub\ECG clssi\ECG code\fusion.xls'
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for k in range (0,200):
  y= [first_sheet.cell_value(k,l) for l in range (180)]
  b[k,:]=dct(y)
  y= np.append(b[k,:],[0,1,0])
  y1= np.append(y1,[y],axis=0)
  #print y1
y2=y1.shape
print y2
print"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5"
print "PVC"

#PVC
c = np.zeros((200,180))
z1 = np.empty((200,183))
file_location = r'I:\Git Hub\ECG clssi\ECG code\PVC.xls'
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for m in range (0,200):
  z= [first_sheet.cell_value(m,n) for n in range (180)]
  c[m,:]=dct(z)
  z= np.append(c[m,:],[0,0,1])
  z1= np.append(z1,[z],axis=0)
z2=z1.shape
print z2
#np.savetxt("newPVC.csv",z1,delimiter=",")

total = np.vstack((x1[200:400,:],y1[200:400,:],z1[200:400,:]))
#t=np.append((x,y),axis =1)
#np.savetxt("feature.csv",total,delimiter=",")
#print total
#t=total.shape
#print t
np.random.shuffle(total)
#print total1
np.savetxt("feature1.csv",total,delimiter=",")

