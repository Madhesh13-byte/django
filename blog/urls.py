from django.urls import path,include
from . import views
urlpatterns = [
    path('',views.allpost,name="allpost"),
    path('<int:blog_id>',views.detail,name="detail"),
]