# created in 2024/04/26

# based on "Phaser: Enabling Phased Array Signal Processing on Commodity WiFi Access Points"
# the objective is to create a multi-antenna array with two cards

import read_bf_file
import numpy as np
import matplotlib.pyplot as plt

# GLOBAL PARAMETER
SIZE_OF_SUBCARRIER = 30                                     # number of subcarrier
INDEX_OF_SUBCARRIER20 = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                              1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])
INDEX_OF_SUBCARRIER40 = np.array([-58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2,
                              2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58])
                                                            # index of subcarrier
DELTA_FREQUENCY = 312.5e3                                   # subcarrier frequency interval             
                            
CHANNEL_OF_5G = 100                                         # index of 5GHz channel
BANDWIDTH_OF_CHANNEL = 40e6                                 # bandwidth
SIZE_OF_TRANSMITTING_ANTENNA = 1                            # number of tx antennas
SIZE_OF_TRANSMITTERS = 1                                    # number of Txs
SIZE_OF_RECEIVING_ANTENNA = 4                               # number of Rx antennas
SIZE_OF_RECEIVERS = 2                                       # number of Rxs
HIGH_THROUGHPUT_MODE = True                                 # HT mode

TIME_INTERVAL = 0.005                                       # sample interval
SAMPLE_RATE = 1 / TIME_INTERVAL                             # sampling frequency
CENTER_OF_FREQUENCY = 5.5e9                                 # center frequency
SPEED_OF_LIGHT = 299792458                                  # speed of electromagnetic wave
ANTDISTANCE = 2.8e-2                                        # half of the wavelength

class readCSIData:
    def __init__(self, filePath1, filepath2):
        # csi of each ants
        self.antennaPairOne = []
        self.antennaPairTwo = []
        self.antennaPairThree = []
        self.antennaPairFour = []
        self.antennaPairFive = []
        self.antennaPairSix = []
        
        # timestamp of each cards
        self.timestamp1 = []
        self.timestamp2 = []
        
        # rssi of each ants
        self.rssi_a = []
        self.rssi_b = []
        self.rssi_c = []
        self.rssi_d = []
        self.rssi_e = []
        self.rssi_f = []
        
        # other information
        self.length1 = 0
        self.length2 = 0
        self.noise1 = []
        self.noise2 = []
        self.agc1 = []
        self.agc2 = []
        
        # size: N * SIZE_OF_RECEIVING_ANTENNA * SIZE_OF_SUBCARRIER
        self.csi = None
        self.length = 0
        self.read_file(filePath1, filepath2)
        self.sequence_alignment()
        self.length = len(self.retainedIndex[0])
        self.csi = self.antennaPairOne[self.retainedIndex[0]]
        self.csi = np.append(self.csi, self.antennaPairTwo[self.retainedIndex[0]], axis=1)
        self.csi = np.append(self.csi, self.antennaPairThree[self.retainedIndex[0]], axis=1)
        self.csi = np.append(self.csi, self.antennaPairFour[self.retainedIndex[1]], axis=1)
        self.csi = np.append(self.csi, self.antennaPairFive[self.retainedIndex[1]], axis=1)
        self.csi = np.append(self.csi, self.antennaPairSix[self.retainedIndex[1]], axis=1)
        print(f'the total size of csi: {self.csi.shape}')
    
    def read_file(self, filepath1, filepath2):
        file1, self.length1 = read_bf_file.read_file(filepath1)
        file2, self.length2 = read_bf_file.read_file(filepath2)
        startTime1 = file1[0].timestamp_low
        startTime2 = file2[0].timestamp_low
        print("Length of packets1: {}".format(self.length1) + "    Start timestamp:" + str(startTime1))
        print("Length of packets2: {}".format(self.length2) + "    Start timestamp:" + str(startTime2))
        for item in file1:
            self.antennaPairOne.append(item.csi[:30])
            self.antennaPairTwo.append(item.csi[30:60])
            self.antennaPairThree.append(item.csi[60:])
            self.timestamp1.append(item.timestamp_low - startTime1)
            self.rssi_a.append(item.rssi_a)
            self.rssi_b.append(item.rssi_b)
            self.rssi_c.append(item.rssi_c)
            self.noise1.append(item.noise)
            self.agc1.append(item.agc)
            
        for item in file2:
            self.antennaPairFour.append(item.csi[:30])
            self.antennaPairFive.append(item.csi[30:60])
            self.antennaPairSix.append(item.csi[60:])
            self.timestamp2.append(item.timestamp_low - startTime2)
            self.rssi_d.append(item.rssi_a)
            self.rssi_e.append(item.rssi_b)
            self.rssi_f.append(item.rssi_c)
            self.noise2.append(item.noise)
            self.agc2.append(item.agc)
            
        self.antennaPairOne = np.expand_dims(self.antennaPairOne, axis=1)
        self.antennaPairTwo = np.expand_dims(self.antennaPairTwo, axis=1)
        self.antennaPairThree = np.expand_dims(self.antennaPairThree, axis=1)
        self.antennaPairFour = np.expand_dims(self.antennaPairFour, axis=1)
        self.antennaPairFive = np.expand_dims(self.antennaPairFive, axis=1)
        self.antennaPairSix = np.expand_dims(self.antennaPairSix, axis=1)
    
    # @staticmethod
    # TODO the algorithm simply uses the time difference to remove packets that are far apart.
    def sequence_alignment(self, timeInterval=TIME_INTERVAL, alpha=0.5):
        self.retainedIndex = [[] for _ in range(SIZE_OF_RECEIVERS)]
        tmpPoint1 = 0
        tmpPoint2 = 0
        while tmpPoint1 < self.length1 and tmpPoint2 < self.length2:
            if abs(self.timestamp1[tmpPoint1] - self.timestamp2[tmpPoint2]) <= alpha * timeInterval * 1e6:
                self.retainedIndex[0].append(tmpPoint1)
                self.retainedIndex[1].append(tmpPoint2)
                tmpPoint1 += 1
                tmpPoint2 += 1
                continue
            if self.timestamp1[tmpPoint1] - self.timestamp2[tmpPoint2] > alpha * timeInterval * 1e6:
                tmpPoint2 += 1
            else:
                tmpPoint1 += 1
    
    # Principal Components Analysis
    @staticmethod
    def principal_components_analysis_2D(data, n_components=1, dim=0):
        if n_components > data.shape[1 - dim]:
            print('n_components must be less than data.shape[1 - dim]')
            return
        if dim != 0 and dim != 1:
            print('dim must be 0 or 1')
            return
        # data deaveraging operation
        mean = np.mean(data, axis=dim, keepdims=True)
        deaveragingData = data - mean
        # estimate the covariance matrix for 2D matrix
        if dim == 0:
            covMatrix = np.dot(deaveragingData.T, np.conj(deaveragingData))
        else:
            covMatrix = np.dot(deaveragingData, np.conj(deaveragingData.T))
        # calculate the eigenvalues and the eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        eigenvalues = np.abs(eigenvalues)
        sortedIndex = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[sortedIndex]
        eigenvectors = eigenvectors[:, sortedIndex]
        # select the first n_components eigenvectors
        topEigenvectors = eigenvectors[:, :n_components]
        # project data onto the selected eigenvectors
        if dim == 0:
            projectedData = np.dot(data, topEigenvectors)
        else:
            projectedData = np.dot(topEigenvectors.T, data)
        return projectedData
    
    @staticmethod
    # TODO it can be changed to Newtonian interpolation or cubic spline interpolation
    def linear_interpolation(sequence):
        for subcarrier in range(sequence.shape[1]):
            for index, amplitude in enumerate(sequence[:, subcarrier]):
                if np.isinf(amplitude):
                    if index == 0:
                        j = index + 1
                        while (j != len(sequence[:, subcarrier]) - 1 and np.isinf(sequence[j, subcarrier])):
                            j += 1
                        sequence[index, subcarrier] = sequence[j, subcarrier]
                    elif index == len(sequence[:, subcarrier]) - 1:
                        sequence[index, subcarrier] = sequence[index - 1, subcarrier]
                    else:
                        j = index + 1
                        while (j != len(sequence[:, subcarrier]) - 1 and np.isinf(sequence[j, subcarrier])):
                            j += 1
                        if j == len(sequence[index, subcarrier]) - 1:
                            sequence[index, subcarrier] = sequence[index - 1, subcarrier]
                        else:
                            sequence[index, subcarrier] = (sequence[index - 1, subcarrier] + sequence[j, subcarrier]) / 2
        


if __name__ == '__main__':
    # plt.rcParams['savefig.dpi'] = 400  # 图片像素
    # plt.rcParams['figure.dpi'] = 100  # 分辨率
    # plt.rcParams['figure.figsize'] = (6.0, 4.0)
    # readData1 = readData(r'temp1')
    
    # readData1.SCE(490, readData1.length)
    # plt.ylim(-np.pi - 1, np.pi + 1)
    # plt.plot(np.angle(readData1.DynamicPCAret2))
    # # plt.scatter(readData1.ret2_split_index, np.angle(readData1.DynamicPCAret2[readData1.ret2_split_index]), color='red')
    # # plt.plot(range(readData1.ret2_segments_peaks_index[1][0], readData1.ret2_segments_peaks_index[1][-1]), np.angle(readData1.DynamicPCAret2[readData1.ret2_segments_peaks_index[1][0]:readData1.ret2_segments_peaks_index[1][-1]]), color='green')
    # plt.show()
    pass