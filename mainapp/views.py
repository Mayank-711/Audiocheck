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
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline
from .models import AudioFile  # Your AudioFile model

# Load models
whisper_model = whisper.load_model("base")
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
emotion_labels = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral", "calm"]

@csrf_exempt
def analyze_audio(request, audio_id):
    try:
        audio_instance = AudioFile.objects.get(id=audio_id)
        audio_path = audio_instance.audio.path

        if not os.path.exists(audio_path):
            return JsonResponse({"error": "Audio file not found on the server"}, status=404)

        # ✅ Get current time from request
        current_time = int(request.GET.get("time", 0))  # Default 0 if no time provided
        chunk_duration = 3  # Process 3 seconds of audio at a time

        print(f"🔹 Processing audio ID {audio_id}, time: {current_time}-{current_time+chunk_duration}")

        # ✅ Load only the required chunk of audio
        y, sr = librosa.load(audio_path, sr=None, offset=current_time, duration=chunk_duration)

        if len(y) == 0:
            print(f"⚠️ Empty audio chunk at {current_time}s")
            return JsonResponse({"error": "Empty audio chunk"}, status=400)

        # ✅ Whisper transcription for this chunk
        temp_filename = f"temp_chunk_{audio_id}.wav"
        temp_path = os.path.join("media", temp_filename)
        
        # 🔹 Save chunk as WAV file
        try:
            librosa.output.write_wav(temp_path, y, sr)
        except Exception as e:
            print(f"❌ Error saving chunk: {str(e)}")
            return JsonResponse({"error": "Failed to save chunk"}, status=500)

        result = whisper_model.transcribe(temp_path)
        transcript_text = result["text"].strip()
        os.remove(temp_path)  # Delete temp file after processing

        # ✅ Sentence Segmentation and Emotion Detection
        sentences = [sent.text for sent in nlp(transcript_text).sents]
        structured_transcript = []
        for sentence in sentences:
            emotion_result = classifier(sentence, candidate_labels=emotion_labels)
            structured_transcript.append({
                "text": sentence,
                "emotion": emotion_result["labels"][0]  # Top emotion
            })

        # ✅ Pitch Analysis for the chunk
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
        f0 = np.nan_to_num(f0)

        print(f"✅ Transcription: {transcript_text}")
        print(f"✅ Pitch values: {f0[:5]}")  # Print first 5 pitch values

        # ✅ Return JSON Response
        return JsonResponse({
            "transcript": structured_transcript,
            "pitch_values": f0.tolist()
        })

    except AudioFile.DoesNotExist:
        print(f"❌ Error: Audio file ID {audio_id} not found in database")
        return JsonResponse({"error": "Audio file not found in database"}, status=404)
    
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")  # Print full error in console
        return JsonResponse({"error": str(e)}, status=500)
