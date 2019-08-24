from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('welcome', views.welcome, name='welcome'),
    url('annoview', views.anno_view, name='annoview'),
    url('cbir', views.cbir, name='cbir'),
    url('getskuimgstxt', views.generate_imgs_txt, name='getskuimgstxt'),
]
