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
clfc = xgb.XGBClassifier(seed = 2)
clfd = DecisionTreeClassifier()
clff = RandomForestClassifier()
clf = LogisticRegression(random_state = 42)

# data = pd.DataFrame()

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
                attr = re.findall(r"\'(.+?)\'", attr)
                classifier = SF.cleaned_data['classifier']
                classifier = re.findall(r"\'(.+?)\'", classifier)
                train = SF.cleaned_data['train']
                test = SF.cleaned_data['test']
                print(end, attr, classifier, train, test)
                return render(request, 'classifier/loading.html', {'classifier':classifier}) #LOADING
        jsondata = request.session['data']
        jdata = json.loads(jsondata)
        data = pd.DataFrame(jdata)
        file = request.GET['file']
        classifiers = ['Logistic Regression', 'Decision Tree','Support Vector Machine', 'RandomForest', 'XGBoost']
        # classify = [[i," ".join("&nbsp;" if x==" " else x for x in list(i))] for i in classifiers]
        classify = [[i," ".join("\t" if x==" " else x for x in list(i))] for i in classifiers]
        print(classify)
        d = {'file_name':file, 'attr':len(data.columns), 'cols':data.columns, 'classifiers':classifiers ,"classifier":classify}
        SF = SelectForm()
        return render(request,'classifier/select.html', d)
    return render(request, 'classifier/index.html')

def result(request):
    return render(request,'classifier/result.html')
# Create your views here.
