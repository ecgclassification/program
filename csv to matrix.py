'''
NOTE :- the csv file should be in same folder wher the code is save

'''
import csv
import numpy 
with open('combine.csv') as csvfile:             # open the csv file
    readCSV = csv.reader(csvfile, delimiter=',')# define readCSV as variable
    x=list(readCSV)
    #for row in readCSV:
       # print(row)
result= numpy.array(x).astype("float")
print (result)
       # print(row[0])                          # print all rows
       # print(row[0],row[1],row[2],)           # print specific cell    


