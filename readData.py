# created in 2024/04/26 by Mo

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

TIME_INTERVAL = 0.01                                        # sample interval
SAMPLE_RATE = 1 // TIME_INTERVAL                            # sampling frequency
CENTER_OF_FREQUENCY = 5.5e9                                 # center frequency
SPEED_OF_LIGHT = 299792458                                  # speed of electromagnetic wave
ANTDISTANCE = SPEED_OF_LIGHT / CENTER_OF_FREQUENCY / 2      # half of the wavelength

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
        self.length1 = None
        self.length2 = None
        self.noise1 = []
        self.noise2 = []
        self.agc1 = []
        self.agc2 = []
        
        # size: SIZE_OF_RECEIVING_ANTENNA * N * SIZE_OF_SUBCARRIER
        self.csi = None
        self.length = 0
        self.read_file(filePath1, filepath2)
        self.sequence_alignment()
        self.csi = np.array(self.antennaPairOne[self.retainedIndex[0]])
        self.csi = np.append(self.csi, self.antennaPairTwo[self.retainedIndex[0]], axis=0)
        self.csi = np.append(self.csi, self.antennaPairThree[self.retainedIndex[0]], axis=0)
        self.csi = np.append(self.csi, self.antennaPairFour[self.retainedIndex[1]], axis=0)
        self.csi = np.append(self.csi, self.antennaPairFive[self.retainedIndex[1]], axis=0)
        self.csi = np.append(self.csi, self.antennaPairSix[self.retainedIndex[1]], axis=0)
    
    # read file
    def read_file(self, filepath1, filepath2):
        '''extract csi complex value for each antenna pair with shape ( len(file) * 30), i.e., packet number * subcarrier number'''
        file1, self.length1 = read_bf_file.read_file(filepath1)
        file2, self.length2 = read_bf_file.read_file(filepath2)
        startTime1 = file1[0].timestamp_low
        startTime2 = file2[0].timestamp_low
        print("Length of packets1: {}".format(length1) + "    Start timestamp:" + str(startTime1))
        print("Length of packets2: {}".format(length2) + "    Start timestamp:" + str(startTime2))
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
    
    # @staticmethod
    def sequence_alignment(self, timeInterval=TIME_INTERVAL):
        self.retainedIndex = [[] for _ in range(SIZE_OF_RECEIVERS)]
        tmpPoint1 = 0
        tmpPoint2 = 0
        while tmpPoint1 < self.length1 and tmpPoint2 < self.length2:
            if abs(self.timestamp1[tmpPoint1] - self.timestamp2[tmpPoint2]) <= 0.5 * timeInterval:
                self.retainedIndex[0].append(self.timestamp1[tmpPoint1])
                self.retainedIndex[1].append(self.timestamp2[tmpPoint2])
                tmpPoint1 += 1
                tmpPoint2 += 1
                continue
            if self.timestamp1[tmpPoint1] - self.timestamp2[tmpPoint2] <= 0.5 * timeInterval:
                tmpPoint1 += 1
            else:
                tmpPoint2 += 1
    
    # Principal Components Analysis
    @staticmethod
    def principal_components_analysis(data, n_components=1, dim=0):
        # data deaveraging operation
        mean = np.mean(data, axis=dim, keepdims=True)
        deaveragingData = data - mean

        # estimate the covariance matrix for 2D matrix
        if dim == 0:
            covMatrix = np.dot(np.conj(deaveragingData.T), deaveragingData)
        else:
            covMatrix = np.dot(deaveragingData, np.conj(deaveragingData.T))
        # calculate the eigenvalues and the eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
        
        # select the first n_components eigenvectors
        topEigenvectors = eigenvectors[:, :n_components]

        # project data onto the selected eigenvectors
        if dim == 0:
            projectedData = np.dot(data, topEigenvectors) / np.sum(np.abs(topEigenvectors), axis=0, keepdims=True)
        else:
            projectedData = np.dot(topEigenvectors.T, data) / np.sum(np.abs(topEigenvectors), axis=0, keepdims=True).T
        return projectedData
    
    @staticmethod
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