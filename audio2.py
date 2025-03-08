import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load audio file
audio_file = "converted_kolu1.wav"  # Replace with the uploaded file path
y, sr = librosa.load(audio_file, sr=None)

# Extract pitch (fundamental frequency - F0)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)

# Replace NaN values with zero for better visualization
f0 = np.nan_to_num(f0)

# Plot pitch contour
plt.figure(figsize=(10, 4))
plt.plot(f0, label="Pitch (Hz)")
plt.xlabel("Time (frames)")
plt.ylabel("Frequency (Hz)")
plt.title("Pitch Contour")
plt.legend()
plt.show()

# Print pitch values
print("Extracted Pitch Values (Hz):")
print(f0)
