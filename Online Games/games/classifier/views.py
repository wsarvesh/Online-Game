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
import pandas as pd
import numpy as np

clfa = LogisticRegression(random_state = 42)
clfb = SVC(random_state = 912, kernel = 'rbf')
clfc = xgb.XGBClassifier(seed = 2)
clfd = DecisionTreeClassifier()
clff = RandomForestClassifier()
clf = LogisticRegression(random_state = 42)

data = pd.DataFrame()

def home(request):

    if request.method == "POST":
        IP = InputForm(request.POST)
        if IP.is_valid():
            print(request.FILES['file'])
            file = request.FILES['file']
            demo = IP.cleaned_data['demo']
            fs = FileSystemStorage(location='classifier/media/')
            name = "user_file." + file.name.split(".")[-1]
            og_name = file.name
            dir = listdir('classifier/media/')
            [fs.delete(i) for i in dir]
            filename = fs.save(name, file)
            uploaded_file_url = fs.url(filename)
            global data
            data = pd.read_csv("classifier/" + uploaded_file_url, encoding='latin1')
            return HttpResponseRedirect("select/?file="+og_name)

    IP = InputForm()
    return render(request,'classifier/index.html')

def select(request):
    file = request.GET['file']
    global data
    classifiers = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'RandomForest', 'XGBoost']
    d = {'file_name':file, 'attr':len(data.columns), 'cols':data.columns, 'classifiers':classifiers}
    return render(request,'classifier/select.html', d)

def result(request):
    return render(request,'classifier/result.html')
# Create your views here.
