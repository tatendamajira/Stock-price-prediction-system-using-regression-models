from django.urls import path
from .views import fetch_stock_data,login_view, register_view, logout_view, list_symbols,home, train_model_view,news_based_prediction,how


urlpatterns = [
    path('',home,name='home'),
    path('fetch_stock_data', fetch_stock_data, name='fetch_stock_data'),
    path('register/', register_view, name='register'),
    path('login', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('symbols/', list_symbols, name='list_symbols'),
    path('train-model/', train_model_view, name='train_model'),
    path('news-prediction/', news_based_prediction, name='news_prediction'),
    path('how/',how,name='how')

  
]
