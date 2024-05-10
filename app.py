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
from itertools import groupby
#import pandas as pd
app=Flask(__name__)
# Loading the model  
model=pickle.load(open('mymodel.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
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
        audio_slices = audio_slices.batch(32)
        return audio_slices 
    except Exception as e:
        print("Error processing audio file:", e)
        return None
# def preprocess_audio(filename):
#   try:
#     # Load audio using tfio.audio (read entire file)
#     audio_tensor = tfio.audio.AudioIOTensor(filename)
#     audio = audio_tensor.to_tensor()

#     # Convert to float tensor and combine channels
#     audio = tf.math.reduce_sum(audio, axis=1) / 2
#     # Resample to 16 kHz
#     sample_rate = audio_tensor.rate
#     sample_rate = tf.cast(sample_rate, dtype=tf.int64)
#     audio = tfio.audio.resample(audio, rate_in=sample_rate, rate_out=16000)

#     # Split audio into slices (adjust window and stride as needed)
#     window_size = 48000
#     stride = 48000
#     audio_slices = tf.signal.frame(audio, window_size=window_size, frame_step=stride)

#     # Apply preprocessing to each slice
#     def preprocess_slice(audio_slice):
#       # Extract features (spectrogram, etc.)
#       spectrogram = tf.signal.stft(audio_slice, frame_length=320, frame_step=16)
#       spectrogram = tf.abs(spectrogram)
#       spectrogram = tf.expand_dims(spectrogram, axis=2)
#       return spectrogram

#     preprocessed_data = tf.map(preprocess_slice, audio_slices)
    
#     # Return preprocessed data (likely a tensor)
#     return preprocessed_data

#   except Exception as e:
#     print("Error processing audio file:", e)
#     return None

# @app.route('/upload',methods=['POST'])
# def upload():
#   # Get uploaded audio file
#   audio_file = request.files['audio']
#   print(audio_file)
#   print("\n")
#   print("mmmeee")

#   if audio_file:
#     filename = secure_filename(audio_file.filename)
#     print(filename)
#     preprocessed_data = preprocess_audio(filename)
#     # Make predictions with the model
#     print(preprocessed_data)
#     #print("l__________hi________") 
#     predictions = model.predict(preprocessed_data)
#     print(predictions)
#     # Thresholding for Capuchin bird sound detection (adjust threshold as needed)
#     # num_sounds = tf.math.reduce_sum([1 for pred in predictions.flatten() if pred > 0.999]).numpy()
#     # predictions=[1 if prediction > 0.5 else 0 for prediction in predictions]
#     # predictions=[key for key, group in groupby(predictions)]
#     # calls = tf.math.reduce_sum(predictions).numpy()
#     results={}
#     results[filename] = predictions
#     class_preds = {}
#     for file, logits in results.items():
#         class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
#     postprocessed = {}
#     for file, scores in class_preds.items():
#         postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()

#     # Return JSON response
#     print(postprocessed)
#     return jsonify({'filename': filename, 'num_sounds': postprocessed[file]})
#   else:
#     return jsonify({'error': 'No audio file uploaded'}), 400

@app.route('/upload',methods=['POST'])
def upload():
  # ... your existing code ...
    # Get uploaded audio file
  audio_file = request.files['audio']
  print(audio_file)
  print("\n")
  print("mmmeee")

  if audio_file:
    filename = secure_filename(audio_file.filename)
    print(filename)
    preprocessed_data = preprocess_audio(filename)
    # Make predictions with the model
    print(preprocessed_data)
    #print("l__________hi________") 
  predictions = model.predict(preprocessed_data)
  print(predictions)
  results={}
  results[filename] = predictions
  class_preds = {}
  for file, logits in results.items():
      class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
  postprocessed = {}
  for file, scores in class_preds.items():
      # Count the number of sounds (occurrences of 1)
      #num_sounds = sum(scores)  # This is the fix
      num_sounds =tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
      postprocessed[file] = int(num_sounds)

  # Return JSON response
  print(postprocessed)
  return jsonify({'filename': filename, 'num_sounds': postprocessed[file]})


if __name__=="__main__":
    app.run() 