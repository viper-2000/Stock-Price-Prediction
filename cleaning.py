import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

link = "C:\\Users\\karth\\Desktop\\GRE\\NCSU\\SEMESTER1\\ECE592_Data_Science_Dror_Baron\\Project\\"

abbrev = {}
abbrev["GM"] = "General Motors"
abbrev["TSLA"] = "Tesla"
abbrev["F"] = "Ford"
abbrev["EC"] = "Ecopetrol"
abbrev["MT"] = "ArcelorMittal"
abbrev["HMC"] = "Honda Motor Company"
abbrev["GE"] = "General Electrical"
abbrev["SAP"] = "SAP SE"
abbrev["TM"] = "Toyota Motor Corp"

companies = ["GM","F","TSLA"]

company_supplier_list = {}
company_supplier_list["GM"] = ["GM","HMC","MT","EC","TSLA","GE"]
company_supplier_list["TSLA"] = ["TSLA","HMC","MT","GM","EC","GE"]
company_supplier_list["F"] = ["F","TM","HMC","MT","GM","SAP"]

companies_data = {}

for c in companies:
    company_data = pd.read_csv(link+c+"\\"+c+".csv")
    companies_data[c] = []
    for x in company_supplier_list[c]:
        supplier_data = pd.read_csv(link+c+"\\"+x+".csv")
        companies_data[c].append(supplier_data)

dates = []

for c in companies_data.keys():
    for x in range(len(companies_data[c])):
        pd.DataFrame.dropna(companies_data[c][x],axis=0)

for c in companies_data.keys():
    company_data = companies_data[c]
    d="Date"
    for x in range(len(company_data)):
        print(f"{company_supplier_list[c][x]}")
        print(f"Oldest Date: {company_data[x][d][0]} Latest Date: {company_data[x][d][len(company_data[x][d])-1]}")
        dates.append(company_data[x][d][0])

#print(f"{max(dates)}")

d="Date"
for c in companies_data.keys():
    company_data = companies_data[c]
    for x in range(len(company_data)):
        for i in range(len(companies_data[c][x]["Date"])):
            if companies_data[c][x]["Date"][i] == max(dates):
                companies_data[c][x] = companies_data[c][x][i:]



for c in companies_data.keys():
    for x in range(len(companies_data[c])):
        companies_data[c][x].to_csv(link+c+"\\"+company_supplier_list[c][x]+"_clean.csv")

for c in companies_data.keys():
    for x in range(len(companies_data[c])):
        companies_data[c][x] = pd.read_csv(link+"\\"+c+"\\"+company_supplier_list[c][x]+"_clean.csv")

for c in companies_data.keys():
    for x in range(len(companies_data[c])):
        print(len(companies_data[c][x]))

for c in companies_data.keys():
    for x in range(len(companies_data[c])):
        print(f"{company_supplier_list[c][x]}\nOldest Date: {companies_data[c][x][d][0]} Latest Date: {companies_data[c][x][d][len(companies_data[c][x][d])-1]}")
