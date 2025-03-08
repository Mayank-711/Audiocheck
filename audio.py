import os
import whisper
import spacy
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

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
wav_file = "converted_kolu1.wav"
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

# Load Zero-Shot Classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate emotion labels
emotion_labels = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral", "calm"]

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

    # Get emotion classification using Transformer
    emotion_result = classifier(sentence, candidate_labels=emotion_labels)
    detected_emotion = emotion_result["labels"][0]  # Top-ranked emotion

    structured_transcript.append(
        {
            "speaker": speaker,
            "text": sentence,
            "keywords": list(found_keywords),
            "emotion": detected_emotion,  # Assign detected emotion
        }
    )

# Store structured transcription in a dictionary
transcription_output = {"transcript": structured_transcript, "keywords": list(extracted_keywords)}

# Print the dictionary
print("\n📝 Transcription Output (with Transformer-based Emotion Detection):\n")
print(transcription_output)