'''
DCT and append 1,0,0
norma the data
and save the data in "dataset.csv" file
'''
import xlrd
import numpy as np
import csv
from sklearn.preprocessing import normalize
import math as cos
from scipy.fftpack import dct

# normal
a = np.zeros((400,180))
x1 = np.empty((400,183))
file_location = r'normal.xls' # importing normal XL sheet
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for j in range (0,400):
  x= [first_sheet.cell_value(j,i) for i in range (180)]
  a[j,:]=dct(x)                 # DCT opratin
  x= np.append(a[j,:],[0])
  x1= np.append(x1,[x],axis=0)
b =x1.shape
print "normal"
print b
print"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5"
print "fusion"

#fusion
b = np.zeros((400,180))
y1 = np.empty((400,183))
file_location = r'fusion.xls' # importing fusion XL sheet
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for k in range (0,400):
  y= [first_sheet.cell_value(k,l) for l in range (180)]
  b[k,:]=dct(y)
  y= np.append(b[k,:],[1])
  y1= np.append(y1,[y],axis=0)
  #print y1
y2=y1.shape
print y2
print"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5"
print "PVC"

#PVC
c = np.zeros((400,180))
z1 = np.empty((400,183))
file_location = r'PVC.xls'  # importing PVC XL sheet
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for m in range (0,400):
  z= [first_sheet.cell_value(m,n) for n in range (180)]
  c[m,:]=dct(z)
  z= np.append(c[m,:],[2])
  z1= np.append(z1,[z],axis=0)
z2=z1.shape
print z2

total = np.vstack((x1[400:800,:],y1[400:800,:],z1[400:800,:]))

np.random.shuffle(total)        #shuffle all the rows
#print total1
np.savetxt("DCT_data.csv",total,delimiter=",")  #creat CSV file of DCT values


with open('DCT_data.csv') as csvfile:             # open the csv file
    readCSV = csv.reader(csvfile, delimiter=',')# define readCSV as variable
    x=list(readCSV)
    x = normalize(x,axis=0,norm='max')
    x_normed =(x-x.min(0))/x.ptp(0)
print "    "    
print "normalize data"
print x_normed
np.savetxt("dataset.csv",x_normed,delimiter=",") #creat csv file of norm data

    


