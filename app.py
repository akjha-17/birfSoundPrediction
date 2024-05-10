import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import os
from itertools import groupby
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow_io as tfio
from glob import glob
#import pandas as pd
app=Flask(__name__)
# Loading the model  
model=pickle.load(open('mymodel.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')

# def preprocess_audio(audio_bytes):
#   """
#   Loads audio, resamples to 16 kHz mono, applies padding, and creates spectrograms.
#   """
#   # Load audio using tfio.audio
#   audio_tensor = tfio.audio.AudioIOTensor(audio_bytes)
#   # Convert to float tensor and combine channels
#   audio = audio_tensor.to_tensor()
#   audio = tf.math.reduce_sum(audio, axis=1) / 2
#   # Extract sample rate and cast to int64
#   sample_rate = audio_tensor.rate
#   sample_rate = tf.cast(sample_rate, dtype=tf.int64)
#   # Resample to 16 kHz
#   audio = tfio.audio.resample(audio, rate_in=sample_rate, rate_out=16000)
#   # Apply zero padding to fixed length (48000)
#   zero_padding = tf.zeros([48000] - tf.shape(audio), dtype=tf.float32)
#   audio = tf.concat([zero_padding, audio], 0)
#   # Create spectrogram with specific parameters
#   spectrogram = tf.signal.stft(audio, frame_length=320, frame_step=32)
#   spectrogram = tf.abs(spectrogram)
#   spectrogram = tf.expand_dims(spectrogram, axis=2)
#   return spectrogram

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=16)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram
def preprocess_audio(filename): 
    try:
        # Load audio using tfio.audio
        audio_tensor = tfio.audio.AudioIOTensor(filename)
        # Convert to float tensor and combine channels
        audio = audio_tensor.to_tensor()
        audio = tf.math.reduce_sum(audio, axis=1) / 2
        # Extract sample rate and cast to int64
        sample_rate = audio_tensor.rate
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        # Resample to 16 kHz
        audio = tfio.audio.resample(audio, rate_in=sample_rate, rate_out=16000)

        audio_slices = tf.keras.utils.timeseries_dataset_from_array(audio, audio, sequence_length=48000, sequence_stride=48000, batch_size=1)
        samples, index = audio_slices.as_numpy_iterator().next()
        audio_slices = audio_slices.map(preprocess_mp3)
        audio_slices = audio_slices.batch(16)
        return audio_slices 



        # # Apply zero padding to fixed length (48000)
        # zero_padding = tf.zeros([48000] - tf.shape(audio), dtype=tf.float32)
        # audio = tf.concat([zero_padding, audio], 0)
        # # Create spectrogram with specific parameters
        # spectrogram = tf.signal.stft(audio, frame_length=320, frame_step=32)
        # spectrogram = tf.abs(spectrogram)
        # spectrogram = tf.expand_dims(spectrogram, axis=2)
        # return spectrogram
    except Exception as e:
        print("Error processing audio file:", e)
        return None

# def load_mp3_16k_mono(filename):
#     """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
#     res = tfio.audio.AudioIOTensor(filename)
#     # Convert to tensor and combine channels 
#     tensor = res.to_tensor()
#     tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
#     # Extract sample rate and cast
#     sample_rate = res.rate
#     sample_rate = tf.cast(sample_rate, dtype=tf.int64)
#     # Resample to 16 kHz
#     wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
#     return wav

@app.route('/upload',methods=['POST'])
def upload():
  # Get uploaded audio file
  audio_file = request.files['audio']
  print(audio_file)
  print("\n")
  print("mmmeee")

  if audio_file:
    # Read file content
    #audio_bytes = audio_file.read()
    # Preprocess audio data
    filename = secure_filename(audio_file.filename)
    print(filename)
    preprocessed_data = preprocess_audio(filename)
    # Make predictions with the model
    print(preprocessed_data)
    print("l__________hi________") 
    predictions = model.predict(preprocessed_data)
    print(predictions)
    # Thresholding for Capuchin bird sound detection (adjust threshold as needed)
    num_sounds = tf.math.reduce_sum([1 for pred in predictions.flatten() if pred > 0.5]).numpy()
    # Return JSON response
    print(num_sounds)
    return jsonify({'filename': audio_file.filename, 'num_sounds': num_sounds})
  else:
    return jsonify({'error': 'No audio file uploaded'}), 400





# @app.route('/upload',methods=['POST'])
# def upload():
#   # Get uploaded audio file
#   audio_file = request.files['audio']
#   print(audio_file)
#   print("\n")
#   print("mmmeee")
#   if audio_file:
#     # Read file content
#     audio_bytes = audio_file.read()
#     # Preprocess audio data
#     preprocessed_data = preprocess_audio(audio_bytes)
#     # Make predictions with the model
#     print(preprocessed_data)
#     print("l__________hi________")
#     predictions = model.predict(preprocessed_data)
#     print(predictions)
#     # Thresholding for Capuchin bird sound detection (adjust threshold as needed)
#     num_sounds = tf.math.reduce_sum([1 for pred in predictions.flatten() if pred > 0.5]).numpy()
#     # Return JSON response
#     return jsonify({'filename': audio_file.filename, 'num_sounds': num_sounds})
#   else:
#     return jsonify({'error': 'No audio file uploaded'}), 400

if __name__=="__main__":
    app.run() 