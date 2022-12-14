import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

suffix = "_clean.csv"

for c in companies:
    companies_data[c] = []
    for x in company_supplier_list[c]:
        companies_data[c].append(pd.read_csv(link+c+"\\"+x+suffix))

for c in companies_data.keys():
    for x in range(len(companies_data[c])):
        company = pd.DataFrame(companies_data[c][x])
        company["c/o"] = company["Close"]/company["Open"]
        company["c/o"] = np.where(company["c/o"]>=1,1,0)
        companies_data[c][x] = company

for c in companies_data.keys():
    for x in range(len(companies_data[c])):
        company = pd.DataFrame(companies_data[c][x])
        company.to_csv(link+c+"\\"+company_supplier_list[c][x]+"_clean.csv")

for c in companies_data.keys():
    for x in range(len(companies_data[c])):
        companies_data[c][x] = pd.read_csv(link+c+"\\"+company_supplier_list[c][x]+"_clean.csv")

