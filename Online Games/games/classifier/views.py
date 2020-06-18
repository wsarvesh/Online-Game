from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from .forms import *
from .models import *
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from os import listdir
import pandas as pd

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
            data = pd.read_csv("classifier/" + uploaded_file_url)
            print(data)
            return render(request,'classifier/select.html', {'file_name':og_name})

    IP = InputForm()
    return render(request,'classifier/index.html')

def select(request):
    return render(request,'classifier/select.html')

def result(request):
    return render(request,'classifier/result.html')
# Create your views here.
