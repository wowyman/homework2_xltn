
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

print("Starting show ing DTW results")
for index in range(1, 101):
    print("Currently at file: %d" % (index), end="\r")
    audio_file_path = './audio/19021233_MaiCongDanh/c%d.wav' % (index)
    command_timestamp_path = './audio/19021233_MaiCongDanh/c%d.txt' % (index)
    command_array = np.genfromtxt(command_timestamp_path, delimiter='\t', dtype='unicode')

    for command in command_array:
        y, sr = librosa.load(
            path=audio_file_path,
            offset=float(command[0]),
            duration=float(command[1])-float(command[0])
        )

        X = librosa.feature.chroma_cens(y=y, sr=sr)
        noise = np.random.rand(X.shape[0], 200)
        Y = np.concatenate((noise, noise, X, noise), axis=1)
        D, wp = librosa.sequence.dtw(X, Y, subseq=True)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                   ax=ax[0])
        ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
        ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
        ax[0].legend()
        fig.colorbar(img, ax=ax[0])
        ax[1].plot(D[-1, :] / wp.shape[0])
        ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
              title='Matching cost function')
        plt.show()
print("Finished the job")
