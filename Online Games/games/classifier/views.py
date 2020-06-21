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
    for i in classifier:
        if i == 'Logistic Regression':
            t0 = time.clock()
            clf.append(clfa.fit(xtrain,ytrain))
            t1 = time.clock()
            yp = clfa.predict(xtest)
            t2 = time.clock()
            print(accuracy_score(ytest, yp))
            print(classification_report(ytest, yp))
            print(precision_recall_fscore_support(ytest, yp, average='weighted'))
            train_time.append("{:.2f}".format((t1 - t0) * 1000))
            test_time.append("{:.2f}".format((t2 - t1) * 1000))
            accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
        elif i == 'Decision Tree':
            t0 = time.clock()
            clf.append(clfd.fit(xtrain,ytrain))
            t1 = time.clock()
            yp = clfd.predict(xtest)
            t2 = time.clock()
            print(accuracy_score(ytest, yp))
            print(classification_report(ytest, yp))
            print(precision_recall_fscore_support(ytest, yp, average='weighted'))
            train_time.append("{:.2f}".format((t1 - t0) * 1000))
            test_time.append("{:.2f}".format((t2 - t1) * 1000))
            accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
        elif i == 'Support Vector Machine':
            t0 = time.clock()
            clf.append(clfb.fit(xtrain,ytrain))
            t1 = time.clock()
            yp = clfb.predict(xtest)
            t2 = time.clock()
            print(accuracy_score(ytest, yp))
            print(classification_report(ytest, yp))
            print(precision_recall_fscore_support(ytest, yp, average='weighted'))
            train_time.append("{:.2f}".format((t1 - t0) * 1000))
            test_time.append("{:.2f}".format((t2 - t1) * 1000))
            accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
        elif i == 'RandomForest':
            t0 = time.clock()
            clf.append(clff.fit(xtrain,ytrain))
            t1 = time.clock()
            yp = clff.predict(xtest)
            t2 = time.clock()
            print(accuracy_score(ytest, yp))
            print(classification_report(ytest, yp))
            print(precision_recall_fscore_support(ytest, yp, average='weighted'))
            train_time.append("{:.2f}".format((t1 - t0) * 1000))
            test_time.append("{:.2f}".format((t2 - t1) * 1000))
            accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
        elif i == 'XGBoost':
            t0 = time.clock()
            clf.append(clfc.fit(xtrain,ytrain))
            t1 = time.clock()
            yp = clfc.predict(xtest)
            t2 = time.clock()
            print(accuracy_score(ytest, yp))
            print(classification_report(ytest, yp))
            print(precision_recall_fscore_support(ytest, yp, average='weighted'))
            train_time.append("{:.2f}".format((t1 - t0) * 1000))
            test_time.append("{:.2f}".format((t2 - t1) * 1000))
            accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
    return clf, accr, train_time, test_time

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
                start = SF.cleaned_data['start']
                print(start)
                if start == "data":
                    
                file = request.GET['file']
                return render(request, 'classifier/loading.html', {'classifier':classifier,"end":end,"attr":attr,"train":train,"test":test,"file":file})
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
                clf, accr, train_time, test_time = train_model(end, attr, classifier, float(train)/100, float(test)/100, data)
                print(accr, train_time, test_time)
                # print(end, attr, classifier, train, test)
        return render(request, 'classifier/result.html', {'classifier':classifier,"end":end,"attr":attr,"train":train,"test":test,"file":file})

    return render(request,'classifier/index.html')
# Create your views here.
