from flask import Flask, request, render_template, redirect, url_for
import os
import pickle
from sklearn.preprocessing import StandardScaler
import librosa as lr  # biblioteka do przetwarzania audio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from keras import models, preprocessing, Sequential, layers
import time
matplotlib.use('agg')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # folder na przesłane pliki
app.config['NUMERICAL_FEATURES_FOLDER'] = 'uploads/numerical_features'  # folder na przesłane pliki
app.config['SPECTOGRAMS_FOLDER'] = 'uploads/spectograms'  # folder na przesłane pliki
app.config['ALLOWED_EXTENSIONS'] = {'mp3'}  # dozwolone rozszerzenia plików
app.config['MODEL_FOLDER'] = 'models'

app.config['NUMERICAL_FEATURES_MODEL_FOLDER'] = 'models/numerical_features'
app.config['SPECTOGRAM_MODEL_FOLDER'] = 'models/spectograms'
app.config['PREPROCESSING_FOLDER'] = 'models/preprocessing'
app.config['TF_ENABLE_ONEDNN_OPTS'] = 0

# Wczytywanie dostępnych modeli
def load_models():
    numerical_models = {f.split('.')[0]: os.path.join(app.config['NUMERICAL_FEATURES_MODEL_FOLDER'], f)
                        for f in os.listdir(app.config['NUMERICAL_FEATURES_MODEL_FOLDER']) if f.endswith('.pkl')}
    
    spectogram_models = {f.split('.')[0]: os.path.join(app.config['SPECTOGRAM_MODEL_FOLDER'], f)
                         for f in os.listdir(app.config['SPECTOGRAM_MODEL_FOLDER']) if f.endswith('.pkl')}
    return numerical_models, spectogram_models



# Funkcja sprawdzająca rozszerzenie pliku
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Strona główna z formularzem
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # models = load_models()
    use_all_models = False
    genre_result = None
    numerical_models, spectogram_models = load_models()
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        if file:
            data = dict(request.form)
            use_all_models = data.get("all_models", False) == 'on'
            model_type = data['model_type']
            model_numerical = data['model_numerical']
            model_spectogram = data['model_spectogram']
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            if use_all_models:
                process_numerical_features(filename)
                process_spectogram(filename)
            elif model_type == "numerical_features":
                process_numerical_features(filename)
            elif model_type == "spectogram":
                process_spectogram(filename)
            
            if use_all_models:
                num_models = [("numerical_features", num_model, "") for num_model in numerical_models]
                spec_models = [("spectogram", "", spec_model) for spec_model in spectogram_models]
                all_models_params = num_models + spec_models
                genre_result = [predict_genre(filename, *params) for params in all_models_params]
            else:
                genre_result = predict_genre(filename, model_type, model_numerical, model_spectogram)
                genre_result = [genre_result]
    return render_template('index.html', numerical_models=numerical_models.keys(), spectogram_models=spectogram_models.keys(), result=genre_result, use_all_models = use_all_models)

# Funkcja do predykcji gatunku
def predict_genre(file_path, model_type, model_numerical, model_spectogram):
    # Ładowanie modelu (przykład, dostosuj ścieżkę i nazwę pliku)
    if model_type == "numerical_features":
        model_name = model_numerical
        result = predict_numerical(file_path, model_numerical)
    elif model_type == "spectogram":
        model_name = model_spectogram
        result = predict_spectogram(file_path, model_spectogram)

    return (model_name, result)


def process_numerical_features(song_path):

     # TODO: check if correct 
    seconds = 29

    colnames = ['chroma_stft', 'spectral_centorid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate']
    mfccs = [f'mfcc_{i}' for i in range(1, 21)]
    colnames.extend(mfccs)
    genre_df = pd.DataFrame(columns=colnames)

    y, sr = lr.load(song_path, mono=True, duration=seconds)
    chroma = np.mean(lr.feature.chroma_stft(y=y, sr=sr))
    centroid = np.mean(lr.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(lr.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(lr.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(lr.feature.zero_crossing_rate(y))
    mfcc = lr.feature.mfcc(y=y, sr=sr)

    row_vals = [chroma, centroid, bandwidth, rolloff, zcr]
    row_dir = {colnames[i]: row_vals[i] for i in range(len(row_vals))}

    for m in range(1, 21):
            row_dir[f'mfcc_{m}'] = np.mean(mfcc[m-1])

    genre_df = pd.concat([genre_df, pd.DataFrame([row_dir])], ignore_index=True)
    song_name = os.path.splitext(os.path.basename(song_path))[0]
    numerical_features_path = os.path.join(app.config['NUMERICAL_FEATURES_FOLDER'], song_name)
    genre_df.to_csv(f'{numerical_features_path}.csv', index=False)

def process_spectogram(song_path):

    song_name = os.path.splitext(os.path.basename(song_path))[0]
    spectogram_path = os.path.join(app.config['SPECTOGRAMS_FOLDER'], song_name)

    # spectrogram_dir = f"../../spectrograms/{genre}/{filename.replace('.mp3', '.png')}"

    if not os.path.isfile(spectogram_path):
        signal, sr = lr.load(song_path)
        mel_spectrogram = lr.feature.melspectrogram(y=signal, sr=sr)
        lr.display.specshow(lr.power_to_db(mel_spectrogram, ref=np.max), sr=sr)
        plt.axis('off')
        plt.savefig(spectogram_path, bbox_inches='tight', pad_inches=0)


def predict_numerical(song_path, numerical_model_name):

    scaler_path = os.path.join(app.config['PREPROCESSING_FOLDER'], "scaler.pkl")


    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)
   
    model_path = os.path.join(app.config['NUMERICAL_FEATURES_MODEL_FOLDER'], numerical_model_name)
    with open(f'{model_path}.pkl', 'rb') as f:
        model = pickle.load(f)
    
    song_name = os.path.splitext(os.path.basename(song_path))[0]
    numerical_features_path = os.path.join(app.config['NUMERICAL_FEATURES_FOLDER'], song_name)
    numerical_features = pd.read_csv(f"{numerical_features_path}.csv")
    numerical_features = scaler.transform(numerical_features)

    print("XSXSXS")
    print(numerical_features)
    print(numerical_features.shape)
    result = model.predict(numerical_features)[0]
    print(result)
    print(type(result))
    if isinstance(result, np.int32):
        genres = ['blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        return genres[result]
    return result


def predict_spectogram(song_path, spectogram_model_name):
    model_path = os.path.join(app.config['SPECTOGRAM_MODEL_FOLDER'], spectogram_model_name)
    with open(f'{model_path}.pkl', 'rb') as f:
        model = pickle.load(f)


    song_name = os.path.splitext(os.path.basename(song_path))[0]


    spectogram_path = os.path.join(app.config['SPECTOGRAMS_FOLDER'], song_name)
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    img = preprocessing.image.load_img(f"{spectogram_path}.png", target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    rescale = Sequential([layers.experimental.preprocessing.Rescaling(1./255)])

    img_array = rescale(img_array)
    predictions = model.predict(img_array)
    predicted_labels = np.argmax(predictions, axis=1)
    genres = ['blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    return genres[predicted_labels[0]]



if __name__ == '__main__':
    app.run(debug=True)
