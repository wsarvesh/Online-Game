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
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot
import time
import json
import pickle
from pickle import dump
from pickle import load
import re
import os
import zipfile
from io import StringIO, BytesIO
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session
from django.utils import timezone

clfa = LogisticRegression(random_state = 42)
clfb = SVC(random_state = 912, kernel = 'rbf')
clfc = xgb.XGBClassifier()
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

def clean_corr(corr):
    val = corr.values.tolist()
    head = list(corr.columns)
    h_small = [i[:2] for i in head]
    fin = []
    for i in val:
        x = []
        for j in i:
            k = "{:.2f}".format(j)
            if j > 0:
                k = "+"+k
            x.append(k)
        fin.append(x)

    return h_small,head,fin

def clean_skew(sk,head):
    h_small = [i[:2] for i in head]
    s = sk.values.tolist()
    skw = []
    for i in s:
        k = "{:.2f}".format(i)
        if i > 0:
            k = "+"+k
        skw.append(k)
    return zip(h_small,head,skw),skw

def graph_range(x):
    k = x//0.25
    val = 0.25 * (k+1)
    return val

def training(clf,xtrain, xtest, ytrain, ytest,name,name2,sk):
    t0 = time.clock()
    cl = clf.fit(xtrain,ytrain)
    model = cl
    path = 'Online-Game/Online Games/games/classifier/media/models/'
    filename = path + name + "_" + sk + "_" +'model.sav'
    pickle.dump(model, open(filename, 'wb'))
    t1 = time.clock()
    yp = clf.predict(xtest)
    t2 = time.clock()
    tr_time = "{:.2f}".format((t1 - t0) * 1000)
    te_time = "{:.2f}".format((t2 - t1) * 1000)
    acc = "{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100)
    acc_score = accuracy_score(ytest, yp)
    d_acc = "{:.2f}".format(100-float(acc))
    cl = classification_report(ytest, yp)
    cl_report = clean_report(cl)
    prf_score = precision_recall_fscore_support(ytest, yp, average='macro')
    prfs = ["{:.2f}".format(float(i)*100) for i in prf_score[:3]]
    report = [name,cl,acc,d_acc,tr_time,te_time,cl_report,prfs,name2]
    return report

def train_model(end, attr, classifier, train, test, data, sk):
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
    if len(int_feat) > 0 :
        xtrain[int_feat] = scaler.fit_transform(xtrain[int_feat])
        xtest[int_feat] = scaler.transform(xtest[int_feat])
    classification_report = []
    accr = []
    for i in classifier:
        if i == 'Logistic Regression':
            report = training(clfa,xtrain, xtest, ytrain, ytest,i,"Log Reg", sk)
            classification_report.append(report)
            accr.append(report[2])
        elif i == 'Decision Tree':
            report = training(clfd,xtrain, xtest, ytrain, ytest,i,"D Tree", sk)
            classification_report.append(report)
            accr.append(report[2])
        elif i == 'Support Vector Machine':
            report = training(clfb,xtrain, xtest, ytrain, ytest,i,"SVM", sk)
            classification_report.append(report)
            accr.append(report[2])
        elif i == 'RandomForest':
            report = training(clff,xtrain, xtest, ytrain, ytest,i,"RF", sk)
            classification_report.append(report)
            accr.append(report[2])
        elif i == 'XGBoost':
            report = training(clfc,xtrain, xtest, ytrain, ytest,i,"XGB", sk)
            classification_report.append(report)
            accr.append(report[2])
    path = 'Online-Game/Online Games/games/classifier/media/models/'
    filename = path + sk + "_" +'scaler.pkl'
    dump(scaler, open(filename, 'wb'))
    pred_req = []
    pred_req.append(int_feat)
    pred_req.append(str_dict)
    pred_req.append(accr)
    return classification_report, pred_req

def load_model(sk, name):
    models = []
    path = 'Online-Game/Online Games/games/classifier/media/models/'
    filename = path + sk + "_" +'scaler.pkl'
    scaler = load(open(filename, 'rb'))
    for i in name:
        path = 'Online-Game/Online Games/games/classifier/media/models/'
        filename = path + i + "_" + sk + "_" +'model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        models.append(loaded_model)
    return models, scaler

def pred_model(data,models,int_feat, str_dict, attr, scaler):
    pred = data.split(";")[:-1]
    predi = [float(i) if i.replace('.','').isdigit() else i for i in pred]
    predict = {}
    for i in range(len(predi)):
        if attr[i] in  int_feat:
            predict[attr[i]] = predi[i]
        else:
            predict[attr[i]] = str_dict[attr[i]][predi[i]]
    pre = pd.DataFrame(predict,index = [0])
    if len(int_feat) > 0:
        pre[int_feat] = scaler.transform(pre[int_feat])
    p = []
    for i in models:
        p.append(i.predict(pre))
    return p


def home(request):
    if request.method == "POST":
        IP = InputForm(request.POST)
        SP = SessionForm(request.POST)
        if SP.is_valid():
            session = SP.cleaned_data['session']
            if session == "no":
                return render(request,'classifier/index.html', {'error':1})
            elif session == "yes":
                request.session.flush()
                return render(request,'classifier/index.html', {'error':0})
        if IP.is_valid():
            demo = IP.cleaned_data['demo']
            if demo != "" :
                demo = demo + ".csv"
                og_name = demo
                data = pd.read_csv("Online-Game/Online Games/games/classifier/media/demo_data/" + demo, encoding='latin1')
                request.session['data'] = data.to_json()
                return HttpResponseRedirect("select/?file="+og_name)
            file = request.FILES['file']
            fs = FileSystemStorage(location='classifier/media/user_data/')
            name = "user_file." + file.name.split(".")[-1]
            og_name = file.name
            dir = listdir('Online-Game/Online Games/games/classifier/media/user_data/')
            [fs.delete(i) for i in dir]
            filename = fs.save(name, file)
            data = pd.read_csv("Online-Game/Online Games/games/classifier/media/user_data/"+filename, encoding='latin1')
            request.session['data'] = data.to_json()
            return HttpResponseRedirect("select/?file="+og_name)
    if 'data' in request.session:
        return render(request,'classifier/index.html', {'error':1})
    IP = InputForm()
    SP = SessionForm()
    return render(request,'classifier/index.html',)

def select(request):
    if 'data' in request.session:
        if request.method == "POST":
            SF = SelectForm(request.POST)
            if SF.is_valid():
                end = SF.cleaned_data['end']
                request.session['end'] = end
                attr= SF.cleaned_data['attr']
                attrs = re.findall(r"\'(.+?)\'", attr)
                request.session['attribute'] = attr
                classifier = SF.cleaned_data['classifier']
                request.session['classifier'] = classifier
                train = SF.cleaned_data['train']
                test = SF.cleaned_data['test']
                file = request.GET['file']
                start = SF.cleaned_data['start']
                redirect = SF.cleaned_data['redirect']
                if start == "data":
                    redirect = "data_page"
                    return render(request, 'classifier/loading.html', {'classifier':classifier,"end":end,"attr":attr,"train":train,"test":test,"file":file,"redirect":redirect})
                elif start == "start":
                    redirect = "results_page"
                    return render(request, 'classifier/loading.html', {'classifier':classifier,"end":end,"attr":attr,"train":train,"test":test,"file":file,"redirect":redirect})
                elif start == "predict":
                    redirect = "predict_page"
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
        PF = PredictForm()
        jsondata = request.session['data']
        jdata = json.loads(jsondata)
        data = pd.DataFrame(jdata)
        classification_report = []
        file = request.GET['file']
        if request.method == "POST":
            SF = SelectForm(request.POST)
            PF = PredictForm(request.POST)
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
                    dataf = pd.DataFrame(data, columns=attr)
                    unique_freq = []
                    mmm = []
                    mmu = []
                    for i in attr:
                        temp = []
                        stemp = []
                        try:
                            x = dataf[i].mean()
                            temp.append(i)
                            temp.append(dataf[i].min())
                            temp.append(dataf[i].max())
                            k = "{:.2f}".format(x)
                            temp.append(k)
                            mmm.append(temp)
                        except:
                            stemp.append(i)
                            count1 = dataf[i].value_counts()
                            min_max = count1.values.tolist()
                            min_max_a = count1.index.tolist()
                            stemp.append(min_max_a[0])
                            stemp.append(min_max_a[-1])
                            stemp.append(len(dataf[i].unique().tolist()))
                            mmu.append(stemp)

                        count  = dataf[i].value_counts()
                        unique = []
                        freq = []
                        for j,k in zip(count, count.index):
                            freq.append(j)
                            unique.append(k)
                        unique_freq.append([i, unique, freq])
                    count = data[end].value_counts()
                    unique_freq_t = [end, count.index.tolist(), count.values.tolist()]
                    ut = len(unique_freq_t[1])
                    info.append(mmm)                                                                                                    #5
                    info.append(ut)                                                                                                     #6
                    info.append(unique_freq_t)                                                                                          #7
                    info.append(train)                                                                                                  #8
                    info.append(test)
                    info.append(mmu)                                                                                                   #9
                    all_attr = [i for i in attr]
                    all_attr.append(end)
                    u_data = pd.DataFrame(data, columns=all_attr)

                    str_feat = []
                    str_dict = {}
                    int_feat = []
                    x = pd.DataFrame(data, columns=all_attr)
                    for c in all_attr:
                        if type(x[c][0]) == str:
                            str_feat.append(c)
                            x1 = list(set(x[c]))
                            di = {x1[i]:i for i in range(len(x1))}
                            str_dict[c] = di
                        else:
                            int_feat.append(c)
                    for c in str_feat:
                        x[c] = x[c].map(str_dict[c])

                    corr = x.corr(method='pearson')
                    sk= x.skew()
                    corr_head,chead,corel = clean_corr(corr)
                    corelation = zip(chead,corel,corr_head)
                    freqs = unique_freq
                    skew,skw_graph = clean_skew(sk,chead)

                    corel_num1 = [float(i) for i in corel[-1]]
                    corel_num = corel_num1[:-1]
                    if min(corel_num) < 0:
                        m = (-1)*min(corel_num)
                        corel_minmax = [round((m),1)*(-1) - 0.10,round(max(corel_num),1) + 0.10]
                    elif max(corel_num) < 0:
                        m = (-1)*min(corel_num)
                        mx = (-1)*min(corel_num)
                        corel_minmax = [round((m),1)*(-1) - 0.10,round(max(mx),1)*(-1) + 0.10]
                    else:
                        corel_minmax = [round(min(corel_num),1) - 0.10,round(max(corel_num),1) + 0.10]
                    skw_num1 = [float(i) for i in skw_graph]
                    skw_num = skw_num1[:-1]
                    if min(skw_num) < 0:
                        m = (-1)*min(skw_num)
                        skw_minmax = [round((m),1)*(-1) - 0.10,round(max(skw_num),1) + 0.10]
                    elif max(skw_num) < 0:
                        m = (-1)*min(skw_num)
                        mx = (-1)*min(skw_num)
                        skw_minmax = [round((m),1)*(-1) - 0.10,round(max(mx),1)*(-1) + 0.10]
                    else:
                        skw_minmax = [round(min(skw_num),1) - 0.10,round(max(skw_num),1) + 0.10]

                    bar_graph = [corr_head,corel[-1],corel_minmax,skw_graph,skw_minmax]


                    d =  {'classifier':classifier,"end":end,"attr":attr,"file":file, "info":info,'freqs':freqs,'corelation':corelation,'corr_head':corr_head,'skew':skew}
                    d['bar_graph'] = bar_graph
                    d['attr_dist'] = unique_freq
                    return render(request, 'classifier/data.html',d)

                elif redirect == "results_page":
                    sk = str(request.session.session_key)
                    classification_report, pred_req = train_model(end, attr, classifier, float(train)/100, float(test)/100, data, sk)
                    request.session['report'] = pred_req
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
            if PF.is_valid():
                start = PF.cleaned_data['start']
                if start == "down":
                        path = 'Online-Game/Online Games/games/classifier/media/models/'
                        classifier = request.session['classifier']
                        classifier = re.findall(r"\'(.+?)\'", classifier)
                        sk = str(request.session.session_key)
                        filenames = [path + i + "_" + sk + "_" + 'model.sav' for i in classifier]

                        zip_subdir = "Models"
                        zip_filename = zip_subdir + ".zip"

                        s = BytesIO()
                        zf = zipfile.ZipFile(s, "w")

                        for fpath in filenames:
                            fdir, fname = os.path.split(fpath)
                            zip_path = os.path.join(zip_subdir, fname)
                            zf.write(fpath, zip_path)
                        zf.close()

                        response = HttpResponse(s.getvalue(), content_type='application/zip')
                        response['Content-Disposition'] = 'attachment; filename=' + zip_filename
                        return response

                else:
                    attr = request.session['attribute']
                    attr = re.findall(r"\'(.+?)\'", attr)
                    end = request.session['end']
                    classifier = request.session['classifier']
                    classifier = re.findall(r"\'(.+?)\'", classifier)
                    mmm  = []
                    for i in attr:
                        temp = []
                        try:
                            x = data[i].mean()
                            temp.append(i)
                            temp.append(data[i].min())
                            temp.append(data[i].max())
                            k = "{:.2f}".format(x)
                            temp.append(k)
                            mmm.append(temp)
                        except:
                            temp.append(i)
                            count1 = data[i].value_counts()
                            min_max = count1.values.tolist()
                            min_max_a = count1.index.tolist()
                            temp.append(min_max_a)
                            temp.append("STR_ATTR")
                            temp.append(len(data[i].unique().tolist()))
                            mmm.append(temp)

                    pred_data = PF.cleaned_data['data'].split(";")[:-1]
                    prediction = [" " for i in classifier]
                    report = request.session['report']
                    accr = report[2]
                    if len(pred_data) != 0:
                        report = request.session['report']
                        int_feat = report[0]
                        str_dict = report[1]
                        sk = str(request.session.session_key)
                        models, scaler = load_model(sk, classifier)
                        pred = pred_model(PF.cleaned_data['data'], models, int_feat, str_dict, attr, scaler)
                        prediction = [i[0] for i in pred]
                    return render(request, 'classifier/predict.html', {'attr':zip(attr, mmm), 'end':end, 'classifier':zip(classifier,prediction,accr), 'file':file})

    return render(request,'classifier/index.html')

# Create your views here.
