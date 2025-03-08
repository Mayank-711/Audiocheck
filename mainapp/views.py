import tempfile
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

import os
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import AudioUploadForm
from .models import AudioFile
from pydub import AudioSegment
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

@login_required(login_url='login')
def call(request):
    if request.method == "POST":
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES["audio"]
            file_extension = uploaded_file.name.split(".")[-1].lower()

            # Define storage path
            original_path = f"uploads/audio/{uploaded_file.name}"
            wav_path = f"uploads/audio/{os.path.splitext(uploaded_file.name)[0]}.wav"

            # Check if WAV already exists
            if not AudioFile.objects.filter(audio=wav_path).exists():
                # Save uploaded file temporarily
                saved_path = default_storage.save(original_path, ContentFile(uploaded_file.read()))

                # Convert to WAV if not already WAV
                if file_extension != "wav":
                    audio = AudioSegment.from_file(default_storage.path(saved_path))
                    audio = audio.set_channels(1).set_frame_rate(16000)  # Standard settings
                    audio.export(default_storage.path(wav_path), format="wav")
                    
                    # Delete original file after conversion
                    default_storage.delete(saved_path)

                # Save WAV file path to database
                AudioFile.objects.create(audio=wav_path)

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

import os
import whisper
import librosa
import numpy as np
import spacy
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline
from .models import AudioFile  # Your AudioFile model

# Load models once for better performance
print("🔹 Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded!")

print("🔹 Loading NLP and classification models...")
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
emotion_labels = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral", "calm"]
print("✅ NLP models loaded!")

# Load ML models once
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
emotion_labels = ["happy", "sad", "angry", "neutral"]
whisper_model = whisper.load_model("base")

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="whisper")  # Suppress FP16 warning

import soundfile as sf
import numpy as np
import librosa
import soundfile as sf
import os
import tempfile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import AudioFile
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="whisper")  # Suppress FP16 warning

@csrf_exempt
def analyze_audio(request, audio_id):
    try:
        print(f"🔹 Received request for audio ID: {audio_id}")

        # ✅ Get the audio file from the database
        audio_instance = AudioFile.objects.get(id=audio_id)
        audio_path = audio_instance.audio.path  # Full path to the .wav file
        print(f"📂 Found audio file: {audio_path}")

        if not os.path.exists(audio_path):
            print("❌ Error: Audio file not found on the server.")
            return JsonResponse({"error": "Audio file not found on the server"}, status=404)

        # ✅ Get current time from request
        current_time = int(request.GET.get("time", 0))  # Default to 0 if not provided
        chunk_duration = 5  # Process 5-second chunks
        print(f"🔹 Processing chunk from {current_time}s to {current_time + chunk_duration}s")

        # ✅ Load only the required chunk of audio
        y, sr = librosa.load(audio_path, sr=None, offset=current_time, duration=chunk_duration)

        if y is None or len(y) == 0:  # 🚀 Fix: Check for empty audio chunk
            print(f"⚠️ Empty audio chunk at {current_time}s")
            return JsonResponse({"error": "Empty audio chunk"}, status=400)

        print(f"✅ Audio chunk loaded: {len(y)} samples at {sr} Hz")

        # ✅ Save chunk as temporary .wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
            tmp_audio_path = tmp_audio_file.name
            sf.write(tmp_audio_path, y, sr)

        print("🔹 Running Whisper transcription on extracted chunk...")
        result = whisper_model.transcribe(tmp_audio_path, language="en")
        os.remove(tmp_audio_path)  # Cleanup temporary file

        transcript_text = result.get("text", "").strip()
        if not transcript_text:
            print("⚠️ No transcribed text detected.")
            return JsonResponse({"error": "No transcribed text detected"}, status=400)

        print(f"✅ Transcription: {transcript_text}")

        # ✅ Return JSON Response
        print("✅ Sending response...")
        return JsonResponse({
            "transcript": transcript_text
        })

    except AudioFile.DoesNotExist:
        print(f"❌ Error: Audio file ID {audio_id} not found in database")
        return JsonResponse({"error": "Audio file not found in database"}, status=404)
    
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")  # Print full error in console
        return JsonResponse({"error": str(e)}, status=500)
