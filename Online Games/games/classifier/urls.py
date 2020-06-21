from django.urls import path
from . import views

urlpatterns = [
    path('',views.home),
    path('select/',views.select),
    path('result/',views.result),
    # path('load/',views.load),
    # path('predict/',views.predict),
    # path('about/',views.about)
]
