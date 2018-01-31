import xlrd
import numpy as np
import matplotlib.pyplot as plt
import math

file_location = "I:\project\raspberry_project\ecg\new.xls"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)

y = [first_sheet.cell_value(0,i) for i in range(180)]
plt.plot(y)
plt.show()

file_location ="I:\project\raspberry_project\ecg\new.xls"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)

z = [first_sheet.cell_value(0,i) for i in range(180)]

plt.plot(z)

plt.show()
