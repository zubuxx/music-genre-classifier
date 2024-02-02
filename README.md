# Music Genre Classification App

This application is designed to identify the music genre of an mp3 file uploaded by the user. Utilizing the power of the librosa library, the genre is determined based on numerical values extracted from the mp3 file or from a generated spectrogram. 

## Features

- **Music Genre Detection:** Automatically identify the genre of music from an mp3 file.
- **Model Selection:** Users can select a specific model for genre detection or use all available models for a comprehensive analysis.
- **Supported Models:** The application currently supports the following models for music genre classification:
  - Random Forest
  - AdaBoost
  - K-Nearest Neighbors (KNN)
  - Convolutional Neural Network (CNN)

These models were trained using 30-second previews obtained from the Spotify API. It's important to note that the accuracy of predictions for full songs may vary significantly.

## Online Access

The application is hosted online and can be accessed at: [https://genre-classifier.azurewebsites.net/](https://genre-classifier.azurewebsites.net/)

## Usage

To use the application, simply upload an mp3 file, and select the model you wish to use for the genre detection. You also have the option to compare results across all models. The application will analyze the uploaded file and provide you with the music genre classification.

## Disclaimer

Please be aware that the prediction accuracy might be lower for full-length songs, as the models were trained on short previews. 

Enjoy exploring the musical genres of your favorite songs with our application!
