import csv
import numpy
import xlwt
import xlrd

'''
the below code is use to concatnate diffrant xl sheet
'''
wkbk = xlwt.Workbook()
outsheet = wkbk.add_sheet('Sheet1')
# add the address of XL sheet
xlsfiles = [r'I:\project\raspberry_project\database\200_180 data\fusion.xlsx', r'I:\project\raspberry_project\database\200_180 data\normal.xlsx']
outrow_idx = 0
for f in xlsfiles:
    insheet = xlrd.open_workbook(f).sheets()[0]
    for row_idx in xrange(insheet.nrows):
        for col_idx in xrange(insheet.ncols):
            outsheet.write(outrow_idx, col_idx, 
                           insheet.cell_value(row_idx, col_idx))
        outrow_idx += 1# space between 2rows
wkbk.save(r'I:\project\raspberry_project\ecg\combine.csv')
print wkbk

'''
NOTE :- the csv file should be in same folder wher the code is save
'''
with open('combine.csv') as csvfile:             # open the csv file
    readCSV = csv.reader(csvfile, delimiter=',') # define readCSV as variable
    x=list(readCSV)
   # for row in readCSV:
    #    print(row)
result= numpy.array(x)#.astype("float")
print (result)
       # print(row[0])                          # print all rows
       # print(row[0],row[1],row[2],)           # print specific cell    

