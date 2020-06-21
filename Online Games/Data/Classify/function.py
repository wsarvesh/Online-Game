from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

clfa = LogisticRegression(random_state = 42)
clfb = SVC(random_state = 912, kernel = 'rbf')
clfc = xgb.XGBClassifier(seed = 2)
clfd = DecisionTreeClassifier()
clff = RandomForestClassifier()

clf = LogisticRegression(random_state = 42)

data = pd.DataFrame()
xtrain = pd.DataFrame()
ytrain = pd.DataFrame()
xtest = pd.DataFrame()
ytest = pd.DataFrame()

scaler = MinMaxScaler()

t = 0

accu = 0
predval = []
feat = []
title = []
str_dict = {}
int_feat = []
str_feat = []
filename = ""

def tc():
    global xtrain,ytrain,clf
    clf.fit(xtrain,ytrain)

def pl():
    global xtest,ytest,clf
#     print(xtest,ytest)
    yp = clf.predict(xtest)
    return sum(ytest == yp) / float(len(yp))

def pred(y):
    global clf
    print(type(y))
    yp = clf.predict(y)
    print(yp)
    global preval
    preval = yp
    prelbl.config(text="PREDICTION IS: "+str(preval[0]))

def tp():
    global clf
    print(clf)
    tc()
    acc = pl()
    global accu
    accu = acc

def tp2(data):
    global clf
    pred(data)

def browse():
    global filename
    filename = filedialog.askopenfilename()
    fileloc.config(text=filename)


import tkinter
tk=tkinter.Tk()
tk.title("CLASSIFICATION")
tk.geometry('900x500+80+80')
tk.configure(background='black')
from tkinter import *
from tkinter import filedialog

var = StringVar()


csvip=Label(tk,text="ENTER CSV PATH",bg='black',fg='lawn green')
csvip.place(x=50,y=50)

csvloc=StringVar()
featlist = StringVar()
predlist = StringVar()
target = StringVar()

et = 0

def load():
    global filename
    csvf = filename
    try:
        global data
        data = pd.read_csv(csvf)
        global title
        title = data.columns.values
        print(data)
        ld()
    except:
        err.config(text="INCORRECT CSV FILE")


def ld():
    global t
    if t == 0:
        colm.config(text=" ; ".join(title))
        err.config(text="")
        acclbl.config(text="")
        acclbl2.config(text="")
        predf.config(text="")
        prelbl.config(text="")
        featip=Label(tk,text="ENTER FEATURE LIST",bg='black',fg='lawn green')
        featip.place(x=25,y=125)
        textbox2=Entry(tk,textvariable=featlist,width=75)
        textbox2.place(x=150,y=125)
        tarip=Label(tk,text="ENTER TARGET COLUMN",bg='black',fg='lawn green')
        tarip.place(x=575,y=125)
        textbox3=Entry(tk,textvariable=target)
        textbox3.place(x=725,y=125)
        SubmitButton2=Button(tk,text="ENTER",fg="Red")
        SubmitButton2.place(x=425,y=180)
        SubmitButton2.config(command=enter)
        global var
        var.set("Select a model for prediction")
        ch = ['Logistic Regression','Support Vector Machine','XGBoost','Decision Tree','Random Forest']
        dd = OptionMenu(tk,var,*ch)
        dd.place(x = 170,y = 180)
        t = 1
    elif t == 1:
        featlist.set("")
        target.set("")
        t = 0
        enter()
        ld()




def solve():
    err2.config(text="")
    pr = predlist.get()
    if pr=="":
        err3.config("FILL TEXT FIELD")
    else:
        pred = pr.split(",")
        predi = [float(i) if i.replace('.','').isdigit() else i for i in pred]
        predict = {}
        for i in range(len(predi)):
            if feat[i] in  int_feat:
                predict[feat[i]] = predi[i]
            else:
                predict[feat[i]] =str_dict[feat[i]][predi[i]]
        pre = pd.DataFrame(predict,index = [0])
        global scaler
        pre[int_feat] = scaler.transform(pre[int_feat])
        tp2(pre)

def accuracy():
    global xtrain,xtext,ytrain,ytest
#     print(xtrain,type(xtrain))
    tp()
    global accu
    accu1 = "{:.5f}".format(accu)
    acclbl.config(text="ACCURACY = \n\n"+str(accu))
    accu2 = "{:.2f}".format(accu*100)
    acclbl2.config(text="ACCURACY % = \n\n"+str(accu2)+"%")

def enter():
    global t
    global AccuracyBtn,PredictLb,predip,textbox4,SolveBtn
    if t == 1:
        print("damn")
        acclbl.config(text="")
        acclbl2.config(text="")
        err2.config(text="")
        global data
        try:
            global str_feat,str_dict,int_feat
            ft = featlist.get()
            tar = target.get()
            v = var.get()
#             print(ft,tar,v)
            if ft=="" or tar=="" or v=="Select a model for prediction":
                err2.config(text="ENTER ALL VALUES")
            else:
                global feat
                feat = ft.split(',')
                x = data[feat]
#                 print(x)
                y = data[tar]
                for c in feat:
                    if type(x[c][0]) == str:
                        str_feat.append(c)
                        x1 = list(set(x[c]))
                        di = {x1[i]:i for i in range(len(x1))}
                        str_dict[c] = di
                    else:
                        int_feat.append(c)
                for c in str_feat:
                    x[c] = x[c].map(str_dict[c])
#                 print(x)
                global xtrain,xtest,ytrain,ytest
                xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
                global scaler
                xtrain[int_feat] = scaler.fit_transform(xtrain[int_feat])
                xtest[int_feat] = scaler.transform(xtest[int_feat])
                global clf
                if v == "Logistic Regression":
                    clf = clfa
                if v == "Support Vector Machine":
                    clf = clfb
                if v == "XGBoost":
                    clf = clfc
                if v == "Decision Tree":
                    clf = clfd
                if v == "Random Forest":
                    clf = clff
                AccuracyBtn=Button(tk,text="ACCURACY",fg="Red")
                AccuracyBtn.place(x=100,y=250)
                AccuracyBtn.config(command=accuracy)
                PredictLb=Label(tk,text="PREDICTION",bg='black',fg="lawn green",font=14)
                PredictLb.place(x=300,y=250)
                predf.config(text=" ; ".join(feat))
                predip.config(text="ENTER UNKNOWN FEATURE LIST")
                textbox4=Entry(tk,textvariable=predlist,width=50)
                textbox4.place(x=550,y=300)
                SolveBtn=Button(tk,text="PREDICT",fg="Red")
                SolveBtn.place(x=300,y=325)
                SolveBtn.config(command=solve)
        except:
            err2.config(text="INCORRECT VALUES")
    elif t == 0:
#         global AccuracyBtn,PredictLb,predip,textbox4,SolveBtn
#         print("Ghusa")
        AccuracyBtn.destroy()
        PredictLb.destroy()
        predip.destroy()
        predlist.set("")
        SolveBtn.destroy()
        textbox4.destroy()
        predip=Label(tk,text="",bg='black',fg='lawn green')
        predip.place(x=300,y=300)
        int_feat = []



# textbox=Entry(tk,textvariable=csvloc,width=100)
# textbox.place(x=150,y=50)

BrowseButton=Button(tk,text="BROWSE",fg="Red")
BrowseButton.place(x=150,y=50)
BrowseButton.config(command=browse)

fileloc=Label(tk,text="",bg='black',fg='lawn green')
fileloc.place(x=250,y=50)

SubmitButton=Button(tk,text="LOAD",fg="Red")
SubmitButton.place(x=800,y=50)
SubmitButton.config(command=load)

colm=Label(tk,text="",bg='black',fg='lawn green')
colm.place(x=75,y=85)

acclbl=Label(tk,text="",bg='black',fg='lawn green')
acclbl.place(x=100,y=300)
acclbl2=Label(tk,text="",bg='black',fg='lawn green')
acclbl2.place(x=100,y=365)

prelbl=Label(tk,text="",bg='black',fg='lawn green',font=16)
prelbl.place(x=300,y=375)
predf=Label(tk,text="",bg='black',fg='lawn green')
predf.place(x=550,y=270)
predip=Label(tk,text="",bg='black',fg='lawn green')
predip.place(x=300,y=300)


err=Label(tk,text="",bg='black',fg='red')
err.place(x=150,y=25)
err2=Label(tk,text="",bg='black',fg='red')
err2.place(x=600,y=180)
err3=Label(tk,text="",bg='black',fg='red')
err3.place(x=575,y=325)


tk.mainloop()
