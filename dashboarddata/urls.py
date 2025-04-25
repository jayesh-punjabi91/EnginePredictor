from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # path('show-engines/', views.show_possible_engines, name='show_engines'),
]
