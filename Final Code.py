#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import wavfile
from scipy.io.wavfile import read as read_wav
import librosa
import librosa.display
import IPython.display as ipd

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[2]:


dataset_path = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/'

classes = []
items = os.listdir(dataset_path)
index = 0
while index < len(items):
    item = items[index]
    if os.path.isdir(os.path.join(dataset_path, item)):
        classes.append(item)
    index += 1

print(classes)


# In[3]:


train_directory = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset'

categories = [category for category in os.listdir(train_directory) if os.path.isdir(os.path.join(train_directory, category))]

file_counts = []
for category in categories:
    category_path = os.path.join(train_directory, category)
    file_counts.append(len(os.listdir(category_path)))

plt.figure(figsize=(15, 7))
plt.bar(categories, file_counts, color='skyblue', edgecolor='black')
plt.xticks(rotation=60, fontsize=10)
plt.xlabel('Labels (Categories)', fontsize=12)
plt.ylabel('Number of Audio Files', fontsize=12)
plt.title('Distribution of Audio Files Across Categories', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[4]:


audio_path = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/bed/1001.wav'
samples, sample_rate = librosa.load(audio_path, sr=16000)

print(f"Audio samples shape: {samples.shape}")
print(f"Sample rate: {sample_rate} Hz")

plt.figure(figsize=(16, 6))
plt.suptitle(f"Word = bed ({os.path.basename(audio_path)})", fontsize=16, weight='bold')

plt.subplot(1, 2, 1)
time_axis = np.arange(0, len(samples)) / sample_rate  # Calculate time axis
plt.plot(time_axis, samples, color='purple')
plt.title("Waveform", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
S = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sample_rate, x_axis="time", y_axis="mel", cmap="coolwarm")
plt.colorbar(format="%+2.0f dB", ax=plt.gca())
plt.title("Mel Spectrogram", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Mel Frequency (Hz)", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[5]:


audio_path = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/cat/1003.wav'

samples, sample_rate = librosa.load(audio_path, sr=16000)

print(f"Audio samples shape: {samples.shape}")
print(f"Sample rate: {sample_rate} Hz")

plt.figure(figsize=(16, 6))
plt.suptitle(f"Word = cat ({os.path.basename(audio_path)})", fontsize=16, weight='bold')

plt.subplot(1, 2, 1)
time_axis = np.arange(0, len(samples)) / sample_rate
plt.plot(time_axis, samples, color='purple')
plt.title("Waveform", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
S = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sample_rate, x_axis="time", y_axis="mel", cmap="coolwarm")
plt.colorbar(format="%+2.0f dB", ax=plt.gca())
plt.title("Mel Spectrogram", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Mel Frequency (Hz)", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[6]:


audio_path = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/marvel/1003.wav'

samples, sample_rate = librosa.load(audio_path, sr=16000)

print(f"Audio samples shape: {samples.shape}")
print(f"Sample rate: {sample_rate} Hz")

plt.figure(figsize=(16, 6))
plt.suptitle(f"Word = marvel ({os.path.basename(audio_path)})", fontsize=16, weight='bold')

plt.subplot(1, 2, 1)
time_axis = np.arange(0, len(samples)) / sample_rate
plt.plot(time_axis, samples, color='purple')
plt.title("Waveform", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
S = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sample_rate, x_axis="time", y_axis="mel", cmap="coolwarm")
plt.colorbar(format="%+2.0f dB", ax=plt.gca())
plt.title("Mel Spectrogram", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Mel Frequency (Hz)", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[7]:


audio_path = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/house/1003.wav'

samples, sample_rate = librosa.load(audio_path, sr=16000)

print(f"Audio samples shape: {samples.shape}")
print(f"Sample rate: {sample_rate} Hz")

plt.figure(figsize=(16, 6))
plt.suptitle(f"Word = house ({os.path.basename(audio_path)})", fontsize=16, weight='bold')

plt.subplot(1, 2, 1)
time_axis = np.arange(0, len(samples)) / sample_rate
plt.plot(time_axis, samples, color='purple')
plt.title("Waveform", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
S = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sample_rate, x_axis="time", y_axis="mel", cmap="coolwarm")
plt.colorbar(format="%+2.0f dB", ax=plt.gca())
plt.title("Mel Spectrogram", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Mel Frequency (Hz)", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[8]:


audio_path = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/tree/1002.wav'

samples, sample_rate = librosa.load(audio_path, sr=16000)

print(f"Audio samples shape: {samples.shape}")
print(f"Sample rate: {sample_rate} Hz")

plt.figure(figsize=(16, 6))
plt.suptitle(f"Word = tree ({os.path.basename(audio_path)})", fontsize=16, weight='bold')

plt.subplot(1, 2, 1)
time_axis = np.arange(0, len(samples)) / sample_rate
plt.plot(time_axis, samples, color='purple')
plt.title("Waveform", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
S = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sample_rate, x_axis="time", y_axis="mel", cmap="coolwarm")
plt.colorbar(format="%+2.0f dB", ax=plt.gca())
plt.title("Mel Spectrogram", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Mel Frequency (Hz)", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[9]:


audio_path = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/three/1003.wav'

samples, sample_rate = librosa.load(audio_path, sr=16000)

print(f"Audio samples shape: {samples.shape}")
print(f"Sample rate: {sample_rate} Hz")

plt.figure(figsize=(16, 6))
plt.suptitle(f"Word = three ({os.path.basename(audio_path)})", fontsize=16, weight='bold')

plt.subplot(1, 2, 1)
time_axis = np.arange(0, len(samples)) / sample_rate
plt.plot(time_axis, samples, color='purple')
plt.title("Waveform", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
S = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sample_rate, x_axis="time", y_axis="mel", cmap="coolwarm")
plt.colorbar(format="%+2.0f dB", ax=plt.gca())
plt.title("Mel Spectrogram", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Mel Frequency (Hz)", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[10]:


recording_durations = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    
    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            if file.endswith('.wav'):
                file_path = os.path.join(category_path, file)
                
                try:
                    sample_rate, audio_data = read_wav(file_path)
                    duration = len(audio_data) / sample_rate
                    recording_durations.append(duration)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

plt.figure(figsize=(10, 6))
plt.hist(recording_durations, bins=50, color='teal', edgecolor='black', alpha=0.7)
plt.title("Distribution of Audio File Durations", fontsize=14, fontweight='bold')
plt.xlabel("Duration (seconds)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# In[11]:


audio_data = []
audio_labels = []

for category in classes:
    category_path = os.path.join(train_directory, category)
    print(f"Processing category: {category}")
    
    for file_name in os.listdir(category_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(category_path, file_name)
            
            audio_samples, sr = librosa.load(file_path, sr=8000)
            if len(audio_samples) == 8000:
                audio_data.append(audio_samples)
                audio_labels.append(category)

print(f"Total processed audio files: {len(audio_data)}")
print(f"Unique labels: {set(audio_labels)}")


# In[12]:


le = LabelEncoder()
y = le.fit_transform(audio_labels)
classes = list(le.classes_)
y1 = to_categorical(y, num_classes=len(classes))


# In[13]:


lab = pd.get_dummies(audio_labels)


# In[14]:


lab.columns


# In[15]:


len(lab.columns)


# In[16]:


y1.shape


# In[17]:


from sklearn.model_selection import train_test_split


X = np.array(audio_data)
X_reshaped = X.reshape(X.shape[0], 8000, 1)
y = np.array(y1)

x_train, x_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, shuffle=True)

print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")


# In[18]:


import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# In[19]:


model = Sequential()

# first convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(8000, 1)))
model.add(MaxPooling1D(pool_size=2))

# second convolutional layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# third convolutional layer
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[20]:


history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


# In[22]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Test Loss')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Test Accuracy')
axes[1].set_title('Accuracy over Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
plt.tight_layout()
plt.show()


# In[29]:


model = Sequential()

# first convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(8000, 1)))

# second convolutional layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# third convolutional layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# fourth convolutional layer
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# fifth convolutional layer
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[24]:


history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


# In[25]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Test Loss')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Test Accuracy')
axes[1].set_title('Accuracy over Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
plt.tight_layout()
plt.show()


# In[26]:


model = Sequential()

# first convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(8000, 1)))

# second convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# third convolutional layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# fourth convolutional layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# fifth convolutional layer
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# sixth convolutional layer
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# seventh convolutional layer
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[27]:


history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


# In[28]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Test Loss')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Test Accuracy')
axes[1].set_title('Accuracy over Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
plt.tight_layout()
plt.show()



# In[ ]:




