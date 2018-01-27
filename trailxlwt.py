import xlwt



list1=[2.34,4.346,4.234,2.34,4.346,4.234,2.34,4.346,4.234,2.34,4.346,4.234,2.34,4.346,4.234,2.34,4.346,4.234]

book = xlwt.Workbook(encoding="utf-8")

sheet1 = book.add_sheet("Sheet 1")

for i in range (0,5):     
 for j in range (0,18):
  for q in list1:
    q = q+1
    sheet1.write(i, j, q))
  




'''
i=4

for n in list1:
    i = i+1
    sheet1.write(i, 0, n)
'''


book.save("trial.xls")
