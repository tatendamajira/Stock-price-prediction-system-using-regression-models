# stock_analysis/urls.py
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views

# Customize the admin site
admin.site.site_header = "SPPS Administration"
admin.site.site_title = "SPPS Admin Portal"
admin.site.index_title = "Welcome to the SPPS Management System"

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('', include('app.urls')),
]
