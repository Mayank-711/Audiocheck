"""
URL configuration for Audiocheck project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from mainapp.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',home,name='home'),
    path('login/',loginpage,name='login'),
    path('signup/',loginpage,name='signup'),
    path('call/',call,name='call'),
    path('logout/',user_logout,name='logout'),
    path("delete_audio/<int:audio_id>/", delete_audio, name="delete_audio"),
    path("analyze_audio/<int:audio_id>/",analyze_audio,name='analyze_audio'),
    path("extract_pitch/<int:audio_id>/",extract_pitch_view,name='extract_pitch'),
    path('analyze_text/', analyze_text, name='analyze_text'),  # ✅ Add new endpoint

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)