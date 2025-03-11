# 🎧 AudioCheck – Real-Time QA for Call Bots  

**📅 March 2025** | **Built during HackScript 6.0**  

### 🚀 About the Project  
**AudioCheck** is a **Real-Time Quality Assurance system** for **call bots**, designed to analyze conversations and improve customer interactions. It includes features like:  
- **📊 Sentiment & Emotion Detection** – Understand caller emotions using AI.  
- **🎵 Pitch Analysis** – Detect tone variations for quality assessment.  
- **🚨 Profanity Detection** – Identify inappropriate language in real-time.  

### 🛠️ Technologies Used  
- **Django** – Backend framework for API and processing.  
- **Whisper AI** – Speech-to-text transcription.  
- **BART (facebook/bart-large-mnli)** – Emotion and sentiment analysis.  
- **NLTK & SpaCy** – Text processing and keyword extraction.  
- **PostgreSQL** – Database for storing conversation insights.  

### 📌 Features  
✅ Real-time **speech-to-text** conversion  
✅ **Sentiment & emotion analysis** from call transcripts  
✅ **Profanity detection** to flag inappropriate words  
✅ **Pitch analysis** to detect tone changes  
✅ **Django REST API** for integration with call center systems  

### ⚡ Setup & Installation  

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/AudioCheck.git
cd AudioCheck
```
  
2️⃣ **Create a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

4️⃣ **Run Migrations & Start Server**  
```bash
python manage.py migrate
python manage.py runserver
```

### 📬 API Endpoints (Example)  
- `POST /api/analyze/` – Upload an audio file for real-time analysis  
- `GET /api/results/{id}/` – Retrieve analysis results  

### 👥 Team & Contributions  
Developed during **HackScript 6.0**. Feel free to contribute by opening issues or pull requests!  

