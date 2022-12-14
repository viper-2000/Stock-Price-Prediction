import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn import svm
from termcolor import colored

np.set_printoptions(suppress=True)

# FILE STRUCTURE
# link = \path_to_folder\
# in \path_to_folder\
# \path_to_folder\
# |
# |->Company
#    |
#    |->company + suppler as STC.csv STC_clean.csv



pd.options.mode.chained_assignment = None  # default='warn'

link = "C:\\Users\\karth\\Desktop\\GRE\\NCSU\\SEMESTER1\\ECE592_Data_Science_Dror_Baron\\Project\\"
suffix = "clean.csv"
format = '%Y-%m-%d'

days = 100

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
abbrev["JNJ"] = "Johnson and Johnson"
abbrev["LLY"] = "Eli Lilly"
abbrev["PFE"] = "Pfizer Inc"
abbrev["AAPL"] = "Apple"
abbrev["MSFT"] = "Microsoft"
abbrev["QCOM"] = "Qualcomm"
abbrev["ACN"] = "Accenture"
abbrev["CVX"] = "Chevron Corp"
abbrev["DOW"] = "Dow Inc"
abbrev["TMO"] = "Thermo Fisher Scientific Inc"
abbrev["XOM"] = "Exxon Mobil Corporation"
abbrev["DHR"] = "Danaher Corporation"
abbrev["ITW"] = "Illinois Tool Works Inc"
abbrev["WLK"] = "Westlake Corporation"
abbrev["J"] = "Jacobs Solutions Inc"
abbrev["SONY"] = "Sony"
abbrev["PKX"] = "Posco Holdings inc"
abbrev["TSM"] = "TSMC"
abbrev["LPL"] = "LG Display Corp"
abbrev["INTC"] = "Intel Corp"
abbrev["BRKA"] = "Berkshire Hathaway Inc"
abbrev["DELL"] = "Dell"
abbrev["ASML"] = "ASML Holdings"
abbrev["ASX"] = "Ase Technology"
abbrev["CSCO"] = "Cisco"
abbrev["DIS"] = "Walt Disney"


industries = ["Tech","Auto","Hlth"]

companies = {}
companies["Tech"] = ["AAPL","QCOM","MSFT"]
companies["Auto"] = ["GM","F","TSLA"]
companies["Hlth"] = ["JNJ","LLY","PFE"]

company_supplier_list = {}

company_supplier_list["Tech"] = {}
company_supplier_list["Tech"]["AAPL"] = ["AAPL","SONY","PKX","INTC","MMM","QCOM"]
company_supplier_list["Tech"]["QCOM"] = ["QCOM","INTC","ASML","CSCO","MMM","IBM"]
company_supplier_list["Tech"]["MSFT"] = ["MSFT","CSCO","INTC","T","NVDA","LPL"]

company_supplier_list["Auto"] = {}
company_supplier_list["Auto"]["GM"] = ["GM","HMC","MT","EC","TSLA","GE"]
company_supplier_list["Auto"]["TSLA"] = ["TSLA","HMC","MT","GM","EC","GE"]
company_supplier_list["Auto"]["F"] = ["F","TM","HMC","MT","GM","SAP"]

company_supplier_list["Hlth"] = {}
company_supplier_list["Hlth"]["JNJ"] = ["JNJ","XOM","CVX","ACN","DOW","TMO"]
company_supplier_list["Hlth"]["LLY"] = ["LLY","DOW","TMO","DHR","ITW","WLK"]
company_supplier_list["Hlth"]["PFE"] = ["PFE","DOW","TMO","ITW","WLK","J"]

companies_data = {}

columns = ["Open","Close","c/o"]

def readData():
    global companies_data
    print("Read Data")
    for i in range(len(industries)):
        Industry = industries[i]
        companies_data[Industry] = {}
        for c in range(len(companies[Industry])):
            Company = companies[Industry][c]
            companies_data[Industry][Company] = []
            supplier_list = company_supplier_list[Industry][Company]
            print(f"{Company}:")
            for s in range(len(supplier_list)):
                company_name = supplier_list[s]
                company = pd.read_csv(f"{link}{Company}\\{company_name}_{suffix}")
                #company = company.tail(days)
                companies_data[Industry][Company].append(company)
                print(f"{company_name}",end=" | ")
            print("\n")
    return companies_data

def eachFrame(write_to):
    global companies_data
    print("Each Frame")
    for i in range(len(industries)):
        Industry = industries[i]
        for c in range(len(companies[Industry])):
            Company = companies[Industry][c]
            supplier_list = company_supplier_list[Industry][Company]
            print(f"{Company}:")
            for s in range(len(supplier_list)):
                company_name = supplier_list[s]
                print(company_name,end=" | ")
                company = pd.DataFrame(companies_data[Industry][Company][s])
            print("\n")

def DataSplit():
    global companies_data,columns
    print("DataSplit")
    consolidated_data = {}
    company = ""

    for i in range(len(industries)):
        Industry = industries[i]
        consolidated_data[Industry] = {}
        for c in range(len(companies[Industry])):
            Company = companies[Industry][c]
            supplier_list = company_supplier_list[Industry][Company]
            consolidated_data[Industry][Company] = pd.DataFrame()
            for s in range(len(supplier_list)):
                supplier = supplier_list[s]
                company = companies_data[Industry][Company][s]
                for col in columns:
                    consolidated_data[Industry][Company][f"{supplier}_{col}"] = companies_data[Industry][Company][s][col]

    for i in range(len(industries)):
        Industry = industries[i]
        for c in range(len(companies[Industry])):
            Company = companies[Industry][c]
            print(Company,":")
            #print(consolidated_data[Industry][Company].columns)
            #df = df[df['EPS'].notna()]
            for col in consolidated_data[Industry][Company].columns:
                consolidated_data[Industry][Company] = consolidated_data[Industry][Company][consolidated_data[Industry][Company][col].notna()]


    train_data, test_data = {},{}
    for i in range(len(industries)):
        Industry = industries[i]
        train_data[Industry] = {}
        test_data[Industry] = {}
        for c in range(len(companies[Industry])):
            Company = companies[Industry][c]
            train_data[Industry][Company],test_data[Industry][Company] = train_test_split(consolidated_data[Industry][Company],test_size=0.2)
    
    print("Train/Test Data Split Successfully")
    
    for i in range(len(industries)):
        Industry = industries[i]
        for c in range(len(companies[Industry])):
            Company = companies[Industry][c]

            #LASSO(train_data[Industry][Company],test_data[Industry][Company],Industry,Company)

            #DTREE(train_data[Industry][Company],test_data[Industry][Company],Industry,Company)
    return train_data,test_data,consolidated_data

def LASSO(train,test,Industry,Company):
    global columns
    print(f"\nLASSO\nIndustry: {Industry} | Company: {Company}")
    rinse(train,Industry,Company)
    rinse(test,Industry,Company)

    X = pd.DataFrame()
    Train = pd.DataFrame()

    inputs = columns
    outputs = ["c/o"]

    print("#########################")

    for supplier in company_supplier_list[Industry][Company][1:]:
        for i in inputs:
            Train[f"{supplier}_{i}"] = train[f"{supplier}_{i}"]
            X[f"{supplier}_{i}"] = np.asarray(test[f"{supplier}_{i}"])
    
    scaler = StandardScaler()
    Train[Train.columns] = scaler.fit_transform(Train[Train.columns])
    

    lso = Lasso(random_state=0)
    lso.fit(Train,train[f"{Company}_c/o"])

    y = np.asarray(test[f"{Company}_c/o"])

    prediction = lso.predict(X)

    prediction = np.where(prediction>=1,1,0)

    error = np.zeros((len(y),1))

    for outcome in range(len(y)):
        if y[outcome]!=prediction[outcome]:
            error[outcome][0] = 1
    
    percent = accuracy_score(y,prediction)

    print(colored(f"Accuracy of LASSO for {Company} : {(percent*100):0.2f}%","green"))
    print("#########################\n")
    
def DTREE(train,test,Industry,Company):
    global columns
    print(f"\nDecision Tree Regression\nIndustry: {Industry} | Company: {abbrev[Company]}")

    X = pd.DataFrame()
    Train = pd.DataFrame()

    inputs = columns
    ouputs = ["c/o"]

    print("#########################")
    for supplier in company_supplier_list[Industry][Company][1:]:
        for i in inputs:
            Train[f"{supplier}_{i}"] = train[f"{supplier}_{i}"]
            X[f"{supplier}_{i}"] = np.asarray(test[f"{supplier}_{i}"])

    scaler = StandardScaler()
    Train[Train.columns] = scaler.fit_transform(Train[Train.columns])

    dt = DecisionTreeRegressor(random_state=0)
    dt.fit(Train,train[f"{Company}_c/o"])

    y = np.asarray(test[f"{Company}_c/o"])

    prediction = dt.predict(X)

    error=np.zeros((len(y),1))

    for outcome in range(len(y)):
        if y[outcome]!=prediction[outcome]:
            error[outcome][0] = 1
    
    percent = accuracy_score(y,prediction)

    print(colored(f"Accuracy of DT for {Company} : {(percent*100):0.2f}%","green"))
    print("#########################\n")

def knnmodel(train,test,Industry,Company):
    global columns
    featureList = []
    print(f"\nKNN Classification\nIndustry: {Industry} | Company: {abbrev[Company]}")
    print("#########################")
    for c in range(1,len(company_supplier_list[Industry][Company])):
        XC = company_supplier_list[Industry][Company][c]
        for f in columns:
            featureList.append(f"{XC}_{f}")
    
    x_train = train[featureList] 
    y_train = train[f'{Company}_c/o'] 
    x_test = test[featureList] 
    y_test = test[f'{Company}_c/o']

    accuracies = []

    for K in range(20):
        K = K+1
        model = KNeighborsClassifier(n_neighbors = K)
        model.fit(x_train, y_train)  #fit the model
        pred=model.predict(x_test) #make prediction on test set
        accuracy = accuracy_score(y_test, pred)
        accuracies.append(100*accuracy)
    
    print(colored(f"Accuracy of KNN for {Company}: {max(accuracies):0.2f}% at K = {accuracies.index(max(accuracies))}","green"))
    print("#########################\n")

def somethingANN(train,test,Industry,Company,XX):
    global columns
    attr = XX
    featureList = []
    features = columns
    inputs = pd.DataFrame()
    names = company_supplier_list[Industry][Company]
    for i in range(1,len(names)):
        company = names[i]
        for f in features:
            inputs[company+"_"+f] = attr[company+"_"+f]
    print(f"\nANN\nIndustry: {Industry} | Company: {abbrev[Company]}")
    print("#########################")
    for c in range(1,len(company_supplier_list[Industry][Company])):
        XC = company_supplier_list[Industry][Company][c]
        for f in columns:
            featureList.append(f"{XC}_{f}")

    Xtest = test[featureList]
    Ytest = test[f"{Company}_c/o"]
    Xtrain = train[featureList]
    Ytrain = train[f"{Company}_c/o"]

    model = Sequential()
    model.add(Dense(units=6, input_dim = Xtrain.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(Xtrain,Ytrain, batch_size = 15, epochs = 100, verbose=0)

    Predictions = model.predict(Xtest)
    #inputs_tf_test = xsc.inverse_transform(Xtest)

    Predictors = list(inputs.columns)
    TestingData = pd.DataFrame(data = Xtest, columns = Predictors)
    TestingData[f'{Company}_c/o'] = Ytest

    pco = "Predicted_c/o"
    TestingData[pco] = Predictions
    print(Predictions.shape, len(Ytest))
    Predictions = np.where(Predictions>0.5,1,0)
    print(type(Ytest))

    correct = 0
    i=0
    for Y in Ytest:
        if Y == Predictions[i][0]:
            correct += 1
        i+=1
    print(colored(f"Accuracy of ANN for {Company} : {(100*(correct/len(TestingData[pco]))):0.2f}%","green"))
    print("#########################\n")

def svmclass(train,test,Industry,Company):
    global columns
    featureList = []
    print(f"\nSVM\nIndustry: {Industry} | Company: {abbrev[Company]}")
    print("#########################")
    for c in range(1,len(company_supplier_list[Industry][Company])):
        XC = company_supplier_list[Industry][Company][c]
        for f in columns:
            featureList.append(f"{XC}_{f}")

    x_train = train[featureList] 
    y_train = train[f'{Company}_c/o'] 
    x_test = test[featureList] 
    y_test = test[f'{Company}_c/o']

    clf = svm.SVC(kernel='linear') # Linear Kernel

    clf.fit(x_train, y_train)

    #Predict the response for test dataset
    pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    
    print(colored(f"Accuracy of {Company}: {(accuracy*100):0.2f}%","green"))
    print("#########################\n")

def Shrinkage(Industry,Company):
    global columns
    features = columns
    names = company_supplier_list[Industry][Company]
    attr = pd.DataFrame()
    Data = companies_data[Industry][Company]
    for i in range(len(Data)):
        company  = Data[i]
        for f in features:
            attr[names[i]+"_"+f] = company[f]
    print(f"\Shrinkage Classification\nIndustry: {Industry} | Company: {abbrev[Company]}")
    print("#########################")
    
    inputs = pd.DataFrame()
    for i in range(1,len(names)):
        company = names[i]
        for f in features:
            inputs[company+"_"+f] = attr[company+"_"+f]

    xsc = StandardScaler()
    in_sc = xsc.fit(inputs)
    inputs_tf = xsc.transform(inputs)
    outputs = attr[f'{Company}_c/o']
    outputs = np.asarray(outputs).reshape(-1,1)
    inputs_tf_train, inputs_tf_test, outputs_tf_train, outputs_tf_test = train_test_split(inputs_tf, outputs, test_size=0.2, random_state=0)

    lasso = Lasso(alpha=0.03)
    lasso.fit(inputs_tf_train, outputs_tf_train)           
    Prediction = lasso.predict(inputs_tf_test)
    Prediction = Prediction.reshape(-1,1)

    inputs_tf_test = xsc.inverse_transform(inputs_tf_test)

    Predictors = list(inputs.columns)
    TestingData = pd.DataFrame(data = inputs_tf_test, columns = Predictors)
    pco = "Predicted_c/o"
    TestingData[f'{Company}_c/o'] = outputs_tf_test
    TestingData[pco] = Prediction

    for i in range(len(TestingData[pco])):
        if TestingData[pco][i] >=  0.5:
            TestingData[pco][i] = 1
        else:
            TestingData[pco][i] = 0
    correct = 0
    for i in range(len(TestingData[pco])):
        if TestingData[pco][i] == TestingData[f'{Company}_c/o'][i]:
            correct += 1
    print(f"Accuracy of SVM on {Company} : {(100*correct/len(TestingData[pco])):0.2f}%")
    print("#########################\n")
    


def rinse(data, Industry="DefaultIndustry", Company="DefaultCompany"):
    inputs = columns
    interest = []
    
    if (type(data)==pd.DataFrame):
        alert = False
        for C in data.columns:
            for i in inputs:
                if i in C:
                    interest.append(C)


        for feature in interest:
            for element in data[feature]:
                if(math.isnan(element)):
                    print(f"{element} {feature}")
                    alert=True
                    break
        if alert:
            print(f"DataFrame {Industry}_{Company} Not Clean")
        else:
            print(f"DataFrame {Industry}_{Company} Clean")


    else:
        alert = False
        for I in industries:
            for C in companies[I]:
                company = data[I][C]
                interest = []
                for C in company.columns:
                    for i in inputs:
                        if i in C:
                            interest.append(C)
                for feature in interest:
                    for element in company[feature]:
                        if(math.isnan(element)):
                            print(f"{element} {feature}")
                            alert = True
                            break
        if alert:
            print(f"Data dictionary is not clean")
        else:
            print(f"Data dictionary is clean")


def machineLearning(train,test):
    global total
    for I in industries:
        for C in companies[I]:
            rinse(train[I][C],I,C)
            rinse(test[I][C],I,C)
            LASSO(train[I][C],test[I][C],I,C)
            DTREE(train[I][C],test[I][C],I,C)
            knnmodel(train[I][C],test[I][C],I,C)
            somethingANN(train[I][C],test[I][C],I,C,total[I][C])
            svmclass(train[I][C],test[I][C],I,C)
            #Shrinkage(I,C)



companies_data = readData()

print("Loaded")

train,test,total =DataSplit()

#eachFrame(True)

rinse(train)
rinse(test)

def Test():
    I = "Hlth"
    C = "JNJ"
    LASSO(train[I][C],test[I][C],I,C)
    DTREE(train[I][C],test[I][C],I,C)
    knnmodel(train[I][C],test[I][C],I,C)
    somethingANN(train[I][C],test[I][C],I,C,total[I][C])
    svmclass(train[I][C],test[I][C],I,C)

machineLearning(train,test)
