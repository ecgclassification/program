'''
the below code is use to concatnate diffrant xl sheet
'''

import xlwt
import xlrd

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
wkbk.save(r'I:\project\raspberry_project\database\200_180 data\combine.csv')
print wkbk
