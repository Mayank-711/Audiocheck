from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User
import re
from django.contrib.auth.decorators import login_required
from .models import AudioFile
from .forms import AudioUploadForm
import os
from django.conf import settings

def home(request):
    return render(request, "home.html")

def loginpage(request):
    if request.method == 'POST':
        form_type = request.POST.get("form_type")
        print(f"Form Type: {form_type}")  # Debugging

        if form_type == "login":
            username = request.POST.get('username')
            password = request.POST.get('password')
            print("Login Form Submitted")  # Debugging

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)  # Ensure this is correctly executed
                print("User logged in:", user)  # Debugging
                return redirect("call")
            else:
                messages.error(request, "Invalid Credentials")
                return redirect("login")

        elif form_type == "signup":
            print("Signup Form Submitted")  # Debugging
            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            print(f"Received Data - Username: {username}, Email: {email}, Password: {password}")

            # Validate if user exists
            if User.objects.filter(username=username).exists():
                messages.error(request, "User with the same username already exists.")
                return redirect("login")

            if User.objects.filter(email=email).exists():
                messages.error(request, "Email already exists.")
                return redirect("login")

            # Password validations
            if len(password) < 8:
                messages.error(request, "Password must be at least 8 characters long.")
                return redirect("login")

            if not re.search(r'[A-Za-z]', password):
                messages.error(request, "Password must contain at least one letter.")
                return redirect("login")

            if not re.search(r'[0-9]', password):
                messages.error(request, "Password must contain at least one number.")
                return redirect("login")

            # Create the user
            my_user = User.objects.create_user(username=username, email=email, password=password)
            my_user.save()
            messages.success(request, "Account created successfully. Please login to continue.")
            return redirect("login")

    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url='login')
def call(request):
    if request.method == "POST":
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES["audio"]
            if not AudioFile.objects.filter(audio="uploads/audio/" + uploaded_file.name).exists():
                form.save()

        return redirect("call")

    else:
        form = AudioUploadForm()

    audio_files = AudioFile.objects.all()
    return render(request, "call.html", {"form": form, "audio_files": audio_files})


@login_required(login_url='login')
def delete_audio(request, audio_id):
    audio = get_object_or_404(AudioFile, id=audio_id)
    
    # Get the full path of the file
    file_path = os.path.join(settings.MEDIA_ROOT, str(audio.audio))

    # Delete the file from storage
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete the database entry
    audio.delete()

    return redirect("call")

