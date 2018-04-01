'''
NOTE :- the csv file should be in same folder wher the code is save

'''
import csv
import numpy as np
from sklearn.preprocessing import normalize

with open('feature1.csv') as csvfile:             # open the csv file
    readCSV = csv.reader(csvfile, delimiter=',')# define readCSV as variable
    x=list(readCSV)
    x = normalize(x,axis=0,norm='max')
    x_normed =(x-x.min(0))/x.ptp(0)
    print x_normed
    
    #Sprint x
    #for row in readCSV:
       # print(row)
result= np.array(x).astype("float")
#print (result)
       # print(row[0])                          # print all rows
       # print(row[0],row[1],row[2],)           # print specific cell
   
np.savetxt("dataset.csv",x_normed,delimiter=",")



