import warnings
import os
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
warnings.filterwarnings('ignore')


def MFCC_feature_from(audio_path, starting_point, end_point):
    ipd.Audio(audio_path)
    signal, sr = librosa.load(
        path=audio_path,
        offset=float(starting_point),
        duration=float(end_point)-float(starting_point)
    )
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)

    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    #mfccs_features = 
    return np.concatenate((mfccs, delta_mfccs, delta2_mfccs))



def buildDataSet(dir):
    # Filter out the wav audio files under the dir
    #fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for index in range(1, 101):
        print('Currently at file: %d' % (index), end='\r')
        audio_file_path = './audio/19021233_MaiCongDanh/c%d.wav' % (index)
        command_timestamp_path = './audio/19021233_MaiCongDanh/c%d.txt' % (index)
        command_array = np.genfromtxt(command_timestamp_path, delimiter='\t', dtype='unicode')


        for command in command_array:
            label = command[2]
            feature = MFCC_feature_from(
                audio_file_path, command[0], command[1])
            if label not in dataset.keys():
                dataset[label] = []
                dataset[label].append(feature)
            else:
                exist_feature = dataset[label]
                exist_feature.append(feature)
                dataset[label] = exist_feature
    return dataset


def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0, 0],
                              [0, tmp_p, tmp_p, tmp_p, 0],
                              [0, 0, tmp_p, tmp_p, tmp_p],
                              [0, 0, 0, 0.5, 0.5],
                              [0, 0, 0, 0, 1]], dtype=np.float)

    startprobPrior = np.array([0.5, 0.5, 0, 0, 0], dtype=np.float)

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num,
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior,
                           covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        print("-------------------TRAIN DATA---------------------------")
        print(trainData)
        # print("data[0]: ")
        # for x in trainData:
        #     print(len(x))
        # print(len(trainData[0]))
        # print("\n---------------------data[2]: -------------------------")
        # print(len(trainData[3]))
        print("-------------------END TRAIN DATA---------------------------")
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models


def main():
    trainDir = './audio/19021233_MaiCongDanh/'
    print("------------------------- Build Dataset : ----------------- ")
    trainDataSet = buildDataSet(trainDir)
    # print(trainDataSet)
    print("------------------------- END Build Dataset : ----------------- ")
    hmmModels = train_GMMHMM(trainDataSet)
    print("Finish training of the GMM_HMM models for digits 0-9")

    testDir = './audio/19021233_MaiCongDanh/'
    testDataSet = buildDataSet(testDir)

    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label,
              ": predict result label is ", predict)
        if predict == label:
            score_cnt += 1
    print("Final recognition rate is %.2f" %
          (100.0*score_cnt/len(testDataSet.keys())), "%")


if __name__ == '__main__':
    main()
