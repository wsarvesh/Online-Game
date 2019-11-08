
from django.urls import path
from . import views

urlpatterns = [
    path('board/',views.home,name='home'),
    path('',views.login,name='login'),
]
