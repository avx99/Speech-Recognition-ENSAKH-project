import json
import os
import librosa

DATASET_PATH = "dataSet"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 1 
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                start = 0
                finish = start + samples_per_segment
                mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(i-1)
                print("{}".format(file_path))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

save_mfcc(DATASET_PATH, JSON_PATH)
