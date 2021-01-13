import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import librosa

DATA_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 1 
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def checkVoiceByPath(file_path,num_segments=1, num_mfcc=13, n_fft=2048, hop_length=512):
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    start = 0
    finish = start + samples_per_segment
    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    mfcc = mfcc[np.newaxis,:,:]
    pred = model.predict(mfcc)
    pred = pred[:,1]
    res = 0
    msg = ""
    if pred[0] >= 0.5 :
        res = 1
        msg = "up (lfo9)"
    else:
        res = 0
        msg = "down (ltaht)"
    print("the result of the wave file : (" + file_path + ") is : " + str(res) + "( "+ msg +" )")

    
def plot_history(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.show()

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y


def result(y_pred,y_test,j):
    y_pred = y_pred[:,j]
    r = 0
    r_pred = 0
    r_test = 0
    for i in range(len(y_test)):
        if y_pred[i] > 0.5 : 
            r_pred = 1
        else:
            r_pred = 0
        r_test = y_test[i]
        if r_pred == r_test :
            r = r + 1
    return r/len(y_test) , r , len(y_test)            
              



X, y = load_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = keras.Sequential([

keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
keras.layers.Dropout(0.3),

keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
keras.layers.Dropout(0.3),

keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
keras.layers.Dropout(0.3),

keras.layers.Dense(2, activation='softmax')
])

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

model.summary()
    
    

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=100)

y_pred = model.predict(X_test)
res , r , l= result(y_pred,y_test,1)
a = [y_test >= 0.5]
b = [y_pred >= 0.5]



print("the finale result is : " + str(res) + "("+str(r)+"/"+str(l)+")")


# checkVoiceByPath(file_path="C:\\Users\\messi\\Desktop\\tht.wav")
plot_history(history)
 

