import os
import whisper
import spacy
import nltk
import librosa
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure necessary NLTK resources are available
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Load SpaCy model for fallback tokenization
nlp = spacy.load("en_core_web_sm")

# Define input MP3 file
mp3_file = "kolu1.mp3"

# Check if file exists
if not os.path.exists(mp3_file):
    raise FileNotFoundError(f"Error: '{mp3_file}' not found! Please provide a valid file.")

# Convert MP3 to WAV (Whisper works best with WAV files)
wav_file = "converted_audio1.wav"
audio = AudioSegment.from_mp3(mp3_file)
audio.export(wav_file, format="wav")

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")

# Transcribe the audio
print("Transcribing audio...")
result = model.transcribe(wav_file)

# Check if Whisper produced text
transcribed_text = result["text"].strip()
if not transcribed_text:
    raise ValueError("Error: Whisper returned an empty transcript. Check your audio file!")

# Load audio with librosa for emotion detection
y, sr = librosa.load(wav_file)

# Extract pitch (fundamental frequency) and energy
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitch_values = pitches[pitches > 0]  # Remove zero values
energy = np.mean(librosa.feature.rms(y=y))  # Root Mean Square Energy
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Extract speech rhythm

# Emotion classification based on pitch, energy, and tempo
def classify_emotion(pitch_values, energy, tempo):
    if len(pitch_values) == 0:
        return "neutral"

    avg_pitch = np.mean(pitch_values)
    max_pitch = np.max(pitch_values)
    
    if avg_pitch > 280 and energy > 0.06:
        return "happy"
    elif avg_pitch < 150 and energy < 0.02:
        return "sad"
    elif energy > 0.08 and max_pitch > 300:
        return "angry"
    elif avg_pitch < 130 and energy < 0.015:
        return "calm"
    elif tempo > 120 and energy > 0.05:
        return "surprised"
    elif avg_pitch > 220 and energy < 0.03:
        return "fearful"
    elif avg_pitch < 140 and energy > 0.07:
        return "disgusted"
    else:
        return "neutral"

# Determine overall emotion
overall_emotion = classify_emotion(pitch_values, energy, tempo)

# Tokenize transcript into sentences (Fallback to SpaCy if NLTK fails)
try:
    sentences = sent_tokenize(transcribed_text)
except Exception:
    print("Warning: NLTK sentence tokenization failed. Using SpaCy as fallback.")
    sentences = [sent.text for sent in nlp(transcribed_text).sents]

# Extract keywords using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X = vectorizer.fit_transform(sentences)
feature_names = vectorizer.get_feature_names_out()
keywords = set(feature_names)

# Process transcript into structured format
structured_transcript = []
extracted_keywords = set()

for sentence in sentences:
    try:
        words = word_tokenize(sentence.lower())
    except LookupError:
        words = [token.text.lower() for token in nlp(sentence)]

    found_keywords = {word for word in words if word in keywords}
    extracted_keywords.update(found_keywords)

    # Speaker classification (basic heuristic)
    speaker = "Customer" if "?" in sentence or "I need" in sentence else "Call Center"

    structured_transcript.append(
        {
            "speaker": speaker,
            "text": sentence,
            "keywords": list(found_keywords),
            "emotion": overall_emotion,  # Assign detected emotion
        }
    )

# Store structured transcription in a dictionary
transcription_output = {"transcript": structured_transcript, "keywords": list(extracted_keywords)}

# Print the dictionary
print("\n📝 Transcription Output (with multi-emotion detection):\n")
print(transcription_output)
