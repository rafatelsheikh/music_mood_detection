# music_mood_detection
detect the mood of music using CNN model by changing them into images (spectrograms)

DataAugmentation file is used to augment data by adding 3 other versions rather than the original one to each audio file

ConvertAudioToImages file is used to convert all the audio files into their spectrograms (images)

TrainingTheCNNModel file is used to build a CNN model and train it upon the data

DetectNewSongUsingImagesByProbability file is used to detect the mood for new songs and print the probability of each mood
