from django.shortcuts import render

def home(request):
    return render(request,'classifier/index.html')
# Create your views here.
