from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from .forms import *
from .models import *
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pandas.io.json import json_normalize
import pandas as pd
import numpy as np
import time
import json
import re

clfa = LogisticRegression(random_state = 42)
clfb = SVC(random_state = 912, kernel = 'rbf')
clfc = xgb.XGBClassifier(seed = 2, objective = "binary:logistic")
clfd = DecisionTreeClassifier()
clff = RandomForestClassifier()
clf = LogisticRegression(random_state = 42)

# data = pd.DataFrame()
def clean_report(cl):
    c = cl.split("\n")
    c2 = [[i] for i in c if i != ""]
    cx = [re.sub(' +',' ',i[0]) for i in c2]
    c3 = ["".join(list(i)[1:]) if list(i)[0] == " " else "".join(list(i)) for i in cx]
    c4 = []
    for i in c3:
        x = i.split(" ")
        if x[0] == 'precision':
            c4.append([" "]+["P","R","F1","S"])
        elif x[0] == 'macro':
            c4.append([(x[0] + ' avg')] + x[-4:])
        elif x[1] == 'avg':
            c4.append([("WT" + ' avg')] + x[-4:])
        elif x[0] == 'accuracy':
            c4.append(["--" for i in range(5)])
            c4.append(['ACC'] + [" " for i in range(2)] + x[-2:])
        else:
            c4.append(x)
    return c4

def training(clf,xtrain, xtest, ytrain, ytest,name,name2):
    t0 = time.clock()
    cl = clf.fit(xtrain,ytrain)
    t1 = time.clock()
    yp = clf.predict(xtest)
    t2 = time.clock()
    tr_time = "{:.2f}".format((t1 - t0) * 1000)
    te_time = "{:.2f}".format((t2 - t1) * 1000)
    acc = "{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100)
    # print(acc)
    acc_score = accuracy_score(ytest, yp)
    # print(acc_score)
    d_acc = "{:.2f}".format(100-float(acc))
    # print(d_acc)
    cl = classification_report(ytest, yp)
    cl_report = clean_report(cl)
    # string = string.replace(/ +/g, ' ');
    prf_score = precision_recall_fscore_support(ytest, yp, average='macro')
    # print(prf_score)
    prfs = ["{:.2f}".format(float(i)*100) for i in prf_score[:3]]
    # print(prfs)
    report = [name,cl,acc,d_acc,tr_time,te_time,cl_report,prfs,name2]
    return report

def train_model(end, attr, classifier, train, test, data):
    x = data[attr]
    y = data[end]
    str_feat = []
    str_dict = {}
    int_feat = []
    clf = []
    accr = []
    train_time = []
    test_time = []
    scaler = MinMaxScaler()
    for c in attr:
        if type(x[c][0]) == str:
            str_feat.append(c)
            x1 = list(set(x[c]))
            di = {x1[i]:i for i in range(len(x1))}
            str_dict[c] = di
        else:
            int_feat.append(c)
    for c in str_feat:
        x[c] = x[c].map(str_dict[c])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test, random_state=0)
    print(len(xtrain), len(xtest), len(ytrain), len(ytest), xtrain)
    xtrain[int_feat] = scaler.fit_transform(xtrain[int_feat])
    xtest[int_feat] = scaler.transform(xtest[int_feat])
    classification_report = []
    for i in classifier:
        if i == 'Logistic Regression':
            report = training(clfa,xtrain, xtest, ytrain, ytest,i,"Log Reg")
            classification_report.append(report)
        elif i == 'Decision Tree':
            report = training(clfd,xtrain, xtest, ytrain, ytest,i,"D Tree")
            classification_report.append(report)
        elif i == 'Support Vector Machine':
            report = training(clfb,xtrain, xtest, ytrain, ytest,i,"SVM")
            classification_report.append(report)
        elif i == 'RandomForest':
            report = training(clff,xtrain, xtest, ytrain, ytest,i,"RF")
            classification_report.append(report)
        elif i == 'XGBoost':
            report = training(clfc,xtrain, xtest, ytrain, ytest,i,"XGB")
            classification_report.append(report)
    return classification_report

def home(request):
    if request.method == "POST":
        IP = InputForm(request.POST)
        if IP.is_valid():
            demo = IP.cleaned_data['demo']
            if demo != "" :
                demo = demo + ".csv"
                og_name = demo
                data = pd.read_csv("classifier/media/demo_data/" + demo, encoding='latin1')
                request.session['data'] = data.to_json()
                return HttpResponseRedirect("select/?file="+og_name)
            file = request.FILES['file']
            fs = FileSystemStorage(location='classifier/media/user_data/')
            name = "user_file." + file.name.split(".")[-1]
            og_name = file.name
            dir = listdir('classifier/media/user_data/')
            [fs.delete(i) for i in dir]
            filename = fs.save(name, file)
            data = pd.read_csv("classifier/media/user_data/"+filename, encoding='latin1')
            request.session['data'] = data.to_json()
            return HttpResponseRedirect("select/?file="+og_name)
    IP = InputForm()
    return render(request,'classifier/index.html')

def select(request):
    if 'data' in request.session:
        if request.method == "POST":
            SF = SelectForm(request.POST)
            if SF.is_valid():
                end = SF.cleaned_data['end']
                attr= SF.cleaned_data['attr']
                classifier = SF.cleaned_data['classifier']
                train = SF.cleaned_data['train']
                test = SF.cleaned_data['test']
                file = request.GET['file']
                start = SF.cleaned_data['start']
                redirect = SF.cleaned_data['redirect']
                if start == "data":
                    redirect = "data_page"
                    return render(request, 'classifier/loading.html', {'classifier':classifier,"end":end,"attr":attr,"train":train,"test":test,"file":file,"redirect":redirect})
                else:
                    redirect = "results_page"
                    return render(request, 'classifier/loading.html', {'classifier':classifier,"end":end,"attr":attr,"train":train,"test":test,"file":file,"redirect":redirect})

        jsondata = request.session['data']
        jdata = json.loads(jsondata)
        data = pd.DataFrame(jdata)
        file = request.GET['file']
        classifiers = ['Logistic Regression', 'Decision Tree','Support Vector Machine', 'RandomForest', 'XGBoost']
        d = {'file_name':file, 'attr':len(data.columns), 'cols':data.columns, 'classifiers':classifiers}
        SF = SelectForm()
        return render(request,'classifier/select.html', d)
    return render(request, 'classifier/index.html')

def result(request):
    if 'data' in request.session:
        SF = SelectForm()
        jsondata = request.session['data']
        jdata = json.loads(jsondata)
        data = pd.DataFrame(jdata)
        file = request.GET['file']
        if request.method == "POST":
            SF = SelectForm(request.POST)
            if SF.is_valid():
                end = SF.cleaned_data['end']
                attr= SF.cleaned_data['attr']
                attr = re.findall(r"\'(.+?)\'", attr)
                classifier = SF.cleaned_data['classifier']
                classifier = re.findall(r"\'(.+?)\'", classifier)
                train = SF.cleaned_data['train']
                test = SF.cleaned_data['test']
                start = SF.cleaned_data['start']
                redirect = SF.cleaned_data['redirect']
                if redirect == "data_page":
                    info = []
                    datal = len(data)
                    info.append(datal)                                                                                                                      #0
                    attrl = len(data.columns)
                    info.append(attrl)                                                                                                                      #1
                    trainl = int(round(datal * int(train) / 100))
                    info.append(trainl)                                                                                                                     #2
                    testl = int(round(datal * int(test) / 100))
                    info.append(testl)                                                                                                                      #3
                    info.append(len(attr))                                                                                                                  #4
                    print(datal, attrl, len(attr), trainl, testl)
                    return render(request, 'classifier/data.html', {'classifier':classifier,"end":end,"attr":attr,"train":train,"test":test,"file":file, "info":info })
                elif redirect == "results_page":
                    classification_report = train_model(end, attr, classifier, float(train)/100, float(test)/100, data)
                    acc = []
                    pre = []
                    rec = []
                    f1 = []
                    time_tr = []
                    time_te = []
                    name = []
                    graph_names = ["ACCURACY COMPARISON","PRECISION COMPARISON","RECALL COMPARISON","F1-SCORE COMPARISON"]
                    for i in classification_report:
                        name.append(i[-1])
                        acc.append(i[2])
                        pre.append(i[-2][0])
                        rec.append(i[-2][1])
                        f1.append(i[-2][2])
                        time_tr.append(float(i[4]))
                        time_te.append(float(i[5]))
                    graph = [acc,pre,rec,f1]
                    graphs = zip(graph_names,graph)
                    max_time = max(max(time_tr),max(time_te))
                    time_graph = [time_tr,time_te,max_time]
                    return render(request, 'classifier/result.html', {'classification_report':classification_report,'graphs':graphs,"time_graph":time_graph,"name":name,"file":file})
    return render(request,'classifier/index.html')
# Create your views here.
