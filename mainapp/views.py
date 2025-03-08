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
        #print(f"Form Type: {form_type}")  # Debugging

        if form_type == "login":
            username = request.POST.get('username')
            password = request.POST.get('password')
            #print("Login Form Submitted")  # Debugging

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)  # Ensure this is correctly executed
                #print("User logged in:", user)  # Debugging
                return redirect("call")
            else:
                messages.error(request, "Invalid Credentials")
                return redirect("login")

        elif form_type == "signup":
            #print("Signup Form Submitted")  # Debugging
            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            #print(f"Received Data - Username: {username}, Email: {email}, Password: {password}")

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
import soundfile as sf
import tempfile
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="whisper")  # Suppress FP16 warning

# Load models once for better performance
#print("🔹 Loading Whisper model...")
whisper_model = whisper.load_model("base")
#print("✅ Whisper model loaded!")

#print("🔹 Loading NLP and classification models...")
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define emotion and sentiment labels
emotion_labels = ["happy", "angry",  "calm"]
sentiment_labels = ["positive", "neutral", "negative"]

#print("✅ NLP models loaded!")
import traceback  # ✅ Import for debugging

@csrf_exempt
def analyze_audio(request, audio_id):
    try:
        audio_instance = AudioFile.objects.get(id=audio_id)
        audio_path = audio_instance.audio.path  

        if not os.path.exists(audio_path):
            return JsonResponse({"error": "Audio file not found on the server"}, status=404)

        current_time = int(request.GET.get("time", 0))  
        chunk_duration = 3  
        y, sr = librosa.load(audio_path, sr=None, offset=current_time, duration=chunk_duration)

        if y is None or len(y) == 0:  
            return JsonResponse({"error": "Empty audio chunk"}, status=400)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
            tmp_audio_path = tmp_audio_file.name
            sf.write(tmp_audio_path, y, sr)

        result = whisper_model.transcribe(tmp_audio_path, language="en")
        os.remove(tmp_audio_path)  

        transcript_text = result.get("text", "").strip()
        if not transcript_text:
            return JsonResponse({"error": "No transcribed text detected"}, status=400)

        return JsonResponse({"transcript": transcript_text})

    except AudioFile.DoesNotExist:
        return JsonResponse({"error": "Audio file not found in database"}, status=404)

    except Exception as e:
        error_details = traceback.format_exc()  # ✅ Capture full traceback
        print(f"❌ Error in analyze_audio: {error_details}")  # ✅ Print full error details
        return JsonResponse({"error": str(e)}, status=500)

def analyze_emotion(transcript_text):
    emotion_result = classifier(transcript_text, candidate_labels=emotion_labels)
    return emotion_result["labels"][0]

def analyze_sentiment(transcript_text):
    sentiment_result = sentiment_classifier(transcript_text, candidate_labels=sentiment_labels)
    return sentiment_result["labels"][0]

@csrf_exempt
def analyze_text(request):
    try:
        transcript_text = request.GET.get("text", "").strip()
        if not transcript_text:
            return JsonResponse({"error": "No text provided"}, status=400)

        detected_emotion = analyze_emotion(transcript_text)
        detected_sentiment = analyze_sentiment(transcript_text)

        return JsonResponse({
            "emotion": detected_emotion,
            "sentiment": detected_sentiment
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def extract_pitch(y, sr):
    """
    Extracts pitch (F0) values from an audio signal.
    
    :param y: Audio signal (numpy array)
    :param sr: Sample rate of the audio
    :return: A dictionary containing min, max, and average pitch or an error message
    """
    try:
        #print("🔹 Extracting pitch (F0) values...")
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_values = f0[~np.isnan(f0)].tolist()  # Remove NaN values

        if not f0_values:
            #print("⚠️ No pitch detected.")
            return {"error": "No pitch detected"}

        pitch_data = {
            "min_pitch": round(min(f0_values), 2),
            "max_pitch": round(max(f0_values), 2),
            "avg_pitch": round(np.mean(f0_values), 2),
        }
        #print(f"✅ Pitch range detected: {pitch_data}")
        return pitch_data

    except Exception as e:
        #print(f"❌ Pitch extraction error: {str(e)}")
        return {"error": "Pitch extraction error"}

@csrf_exempt
def extract_pitch_view(request, audio_id):
    """
    Django view to extract pitch (F0) from a 5-second chunk of audio.
    """
    try:
        #print(f"🔹 Received request for pitch extraction - Audio ID: {audio_id}")

        # ✅ Get the audio file from the database
        audio_instance = AudioFile.objects.get(id=audio_id)
        audio_path = audio_instance.audio.path  # Full path to the .wav file
        #print(f"📂 Found audio file: {audio_path}")

        if not os.path.exists(audio_path):
            #print("❌ Error: Audio file not found on the server.")
            return JsonResponse({"error": "Audio file not found on the server"}, status=404)

        # ✅ Get current time from request
        current_time = int(request.GET.get("time", 0))  # Default to 0 if not provided
        chunk_duration = 3  # Process 5-second chunks
        #print(f"🔹 Processing chunk from {current_time}s to {current_time + chunk_duration}s")

        # ✅ Load only the required chunk of audio
        y, sr = librosa.load(audio_path, sr=None, offset=current_time, duration=chunk_duration)

        if y is None or len(y) == 0:  # 🚀 Fix: Check for empty audio chunk
            #print(f"⚠️ Empty audio chunk at {current_time}s")
            return JsonResponse({"error": "Empty audio chunk"}, status=400)

        #print(f"✅ Audio chunk loaded: {len(y)} samples at {sr} Hz")

        # ✅ Extract pitch (F0) values
        #print("🔹 Extracting pitch (F0) values...")
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_values = f0[~np.isnan(f0)].tolist()  # Remove NaN values

        if not f0_values:
            #print("⚠️ No pitch detected.")
            return JsonResponse({"error": "No pitch detected"}, status=400)

        # ✅ Calculate pitch statistics
        min_pitch = round(min(f0_values), 2)
        max_pitch = round(max(f0_values), 2)
        avg_pitch = round(np.mean(f0_values), 2)
        pitch_summary = f"{min_pitch}Hz - {max_pitch}Hz (avg: {avg_pitch}Hz)"

        #print(f"✅ Pitch range detected: {pitch_summary}")

        # ✅ Return JSON response
        return JsonResponse({
            "min_pitch": min_pitch,
            "max_pitch": max_pitch,
            "avg_pitch": avg_pitch
        })

    except AudioFile.DoesNotExist:
        #print(f"❌ Error: Audio file ID {audio_id} not found in database")
        return JsonResponse({"error": "Audio file not found in database"}, status=404)
    
    except Exception as e:
        #print(f"❌ Unexpected Error: {str(e)}")  # #print full error in console
        return JsonResponse({"error": str(e)}, status=500)