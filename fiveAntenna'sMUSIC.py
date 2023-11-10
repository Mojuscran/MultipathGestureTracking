# created in 2023/11/10

# based on <<Multiple Emitter Location and Signal Parameter - - Estimation>>
# and <<Phaser: Enabling Phased Array Signal Processing on Commodity WiFi Access Points>>
# the objective is to explore a five-antenna multi-signal classification algorithm

import read_bf_file
import numpy as np
from PCA import PCAtest
import matplotlib.pyplot as plt
import scipy.stats as sc

sampleFrequency = 200           # HertzS
centerFrequency = 5.32e9        # Hertz
speedOfLight = 299792458        # speed of electromagnetic wave
antDistance = 2.8e-2            # half of the wavelength
rxAntennaNum = 5                # nmber of rx antennas
txAntennaNum = 1                # nmber of tx antennas
subCarrierNum = 30
f_gap = 312.5e3
subCarrierIndex40 = np.array([-58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2,
                              2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58])
subCarrierIndex20 = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                              1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])

class Track(object):
    def __init__(self,
                 search_interval=(-np.pi / 2, np.pi / 2),           # angle search interval(radian measure)
                 toa_interval=(-2.5, 2.5),                          # 
                 slide_window=0.4,                                  # sliding window size(unit is s)
                 filename1=None,                                    # the first card's CSI path
                 filename2=None,                                    # the second card's CSI path
                 initial_angle=0,                                   # the initial angle
                 use_mdl=True,                                      # whether to use mdl
                 use_pca=True,                                      # whether to use PCA
                 use40mhz=False,                                    # whether to use 40MHz
                 use_trans=[5, 6],                                  # 
                 overlay=0.2,                                       # overlap ratio
                 show_image=0,                                      # whether to show pseudospectral graph
                 begin=None,                                        # the start index
                 end=None                                           # the end index
                 ):
        
        self.search_interval = search_interval
        self.toa_interval = toa_interval
        self.slide_window = slide_window
        self.filename1 = filename1
        self.filename2 = filename2
        self.initial_angle = initial_angle
        self.useMDL = use_mdl
        self.usePCA = use_pca
        self.use40MHz = use40mhz
        self.useTrans = use_trans
        self.overlay = overlay
        self.showimage = show_image
        
        self.subCarrierIndex = subCarrierIndex40 if self.use40MHz else subCarrierIndex20
        
        # [1, 90]
        a_mat = np.tile(self.subCarrierIndex, (1, rxAntennaNum))  
        # 2 * 90
        a_mat = np.append(a_mat, np.ones((1, subCarrierNum * rxAntennaNum)), axis=0)
        # 90 * 2
        a_mat = a_mat.transpose()
        # Compute the (Moore-Penrose) pseudo-inverse of a_mat
        self.a_mat_pinv = np.linalg.pinv(a_mat)
        
        
        
        self.angleStepsNum = 10000
        # angle iteration stride
        self.angleStepLen = (self.search_interval[1] - self.search_interval[0]) / self.angleStepsNum
        self.angleSteps = np.arange(self.search_interval[0], self.search_interval[1], self.angleStepLen, dtype=float)
        # convert the search start point and end point to the degree measure
        self.angleIntervalDeg = (self.search_interval[0] * 180 / np.pi, self.search_interval[1] * 180 / np.pi)
        
        self.toaStepsNum = 10000
        self.toaStepLen = (self.toa_interval[1] - self.toa_interval[0]) / self.toaStepsNum
        self.toaSteps = np.arange(self.toa_interval[0], self.toa_interval[1] + self.toaStepLen, self.toaStepLen, dtype=float)
        
        
        
        # size of the sliding windo(number of packets)
        self.slideWindowLen = int(self.slide_window * sampleFrequency)
        
        # stride of the sliding window
        self.stepLength = int((1 - self.overlay) * self.slideWindowLen)
        # length of overlap
        self.overlapLength = self.slideWindowLen - self.stepLength
        
        # read CSI file
        file_r1, file_length1 = read_bf_file.read_file(self.filename1)
        file_r2, file_length2 = read_bf_file.read_file(self.filename2)
        
        if begin != None and end != None:
            self.angle_list_index, self.angle_list = self.readFile(file_r1, file_length1, file_r2, file_length2, begin=begin, end=end)
        else:
            self.angle_list_index, self.angle_list = self.readFile(file_r1, file_length1, file_r2, file_length2, begin=None, end=None)

    def getLogGeoMean(self, x):
        """
        :param x: list of eigenvalues
        :return: Mean of log(eigenvalues) to avoid the overflow problem caused by cumulative product
        """
        n = x.size
        logGeoMean = 0
        for number in x:
            logGeoMean += np.log(number)
        res = logGeoMean / n  # compute geometric mean through formula (log(x1)+ log(x2) +...+ log(xn))/ n
        return res

    def getSteeringVector(self, angle, frequency):
        steeringVec = np.empty((rxAntennaNum, 1), dtype=complex)
        for i in range(rxAntennaNum):
            delay = antDistance * i * np.sin(angle) / speedOfLight
            steeringVec[i] = np.exp(-1j * 2 * np.pi * frequency * delay)
        return steeringVec

    def getMUSIC(self, noiseMultiply, angleSteps, fc):
        # generate MUSIC spectrum
        spectrum = np.zeros(self.angleStepsNum, dtype=float)
        for i in range(self.angleStepsNum):
            steeringVec = self.getSteeringVector(angleSteps[i], fc)
            denominator = np.sum(
                np.dot(np.dot(steeringVec.conjugate().transpose(), noiseMultiply),
                       steeringVec), axis=1)  # matrix multiplication
            spectrum[i] = 1 / abs(denominator)
        return spectrum

    def mdl_algorithm(self, eigenvalues):
        mdl = np.zeros(3)
        lambda_tot = eigenvalues
        sub_arr_size = 3
        n_segments = self.slideWindowLen
        max_multipath = len(lambda_tot)
        for k in range(0, max_multipath):
            mdl[k] = -n_segments * (sub_arr_size - k) * np.log10(sc.gmean(lambda_tot[k:]) / np.mean(lambda_tot[k:])) \
                     + 0.5 * k * (2 * sub_arr_size - k) * np.log10(n_segments)
        index = max(np.argmin(mdl), 1)
        return index

    def getNoiseMat1(self, matrix):
        matr = np.zeros([3, 3], dtype=complex)
        for i in matrix:
            mat = np.asarray(i)  # timestamp * Nrx
            cor = np.dot(mat, mat.conjugate().transpose())
            matr += cor
        # matr = matr / len(matrix)
        # 求特征值与特征向量
        eig, u_tmp = np.linalg.eig(matr)
        eig = np.abs(eig)
        # 将特征值从小到大排列，提取其在排列前对应的index(索引)输出
        un = np.argsort(-eig)
        eig = -np.sort(-eig)
        u = u_tmp[:, un[:]]
        if self.useMDL:
            index = 2
        else:
            index = 1
        qn = u[:, index:]
        noiseMultiply = np.dot(qn, qn.conjugate().transpose())
        return noiseMultiply, index

    def getNoiseMat(self, matrix):
        mat = np.asarray(matrix)  # timestamp * Nrx
        cor = np.dot(mat.transpose(), mat.conjugate())
        eig, u_tmp = np.linalg.eig(cor)
        eig = np.abs(eig)
        un = np.argsort(-eig)
        eig = -np.sort(-eig)
        u = u_tmp[:, un[:]]
        if self.useMDL:
            index = 2
        else:
            index = 1
        qn = u[:, index:]
        noiseMultiply = np.dot(qn, qn.conjugate().transpose())
        return noiseMultiply, index

    def findPeak1D(self, spectrum, sourceNum):
        peakIndexes, peakValues = [], []
        for index in range(1, len(spectrum) - 1):
            if spectrum[index - 1] < spectrum[index] > spectrum[index + 1]:
                peakIndexes.append(index)
                peakValues.append(spectrum[index])
        if len(peakIndexes) < sourceNum:
            return peakIndexes
        else:
            Z = zip(peakValues, peakIndexes)
            Zipped = sorted(Z, reverse=True)  # descend order
            valueDescend, indexDescend = zip(*Zipped)  # in type tuple
            selectIndex = list(indexDescend)[0: sourceNum]
        return selectIndex

    # populate CSI data into a matrix
    def fillCSIMatrix(self, fileToRead1, fileToRead2):
        CSIMatrix = np.zeros([len(fileToRead1), rxAntennaNum, subCarrierNum], dtype=complex)
        timestampCount = 0
        for [item1, item2] in np.array([fileToRead1, fileToRead2]).transpose():
            for EachCSI in range(0, subCarrierNum):
                # 读取的CSI原始数据的一个csi是[子载波数, 发射天线数, 接收天线数]
                CSIMatrix[timestampCount, :, EachCSI] = \
                    np.array([item.csi[EachCSI, 0, 0], item.csi[EachCSI, 0, 1],
                              item.csi[EachCSI, 0, 2]])
            # 对csi数据进行相位矫正
            CSIMatrix[timestampCount] = self.phaseCalibration(CSIMatrix[timestampCount])
            timestampCount += 1
        return CSIMatrix

    # 通过线性拟合去除STO和SFO的影响
    def phaseCalibration(self, csi):
        """
        相位校正 using SpotFi Algorithm 1 (regression)
        :param csi: uncalibrated csi matrix in shape (3, 30)
        :return: calibrated CSI
        """
        # 3 * 30
        phaseRaw = np.angle(csi)
        phaseUnwrapped = np.unwrap(phaseRaw)

        for antIndexForPhase in range(1, rxAntennaNum):
            if phaseUnwrapped[antIndexForPhase, 0] - phaseUnwrapped[0, 0] > np.pi:
                phaseUnwrapped[antIndexForPhase, :] -= 2 * np.pi
            elif phaseUnwrapped[antIndexForPhase, 0] - phaseUnwrapped[0, 0] < -np.pi:
                phaseUnwrapped[antIndexForPhase, :] += 2 * np.pi
        
        phase = phaseUnwrapped.reshape(-1)
        
        # a_mat = np.tile(subCarrierIndex, (1, rxNum))  # [1, 90]

        # # 2 * 90
        # a_mat = np.append(a_mat, np.ones((1, subCarrierNum * rxNum)), axis=0)
        # # 90 * 2
        # a_mat = a_mat.transpose()
        # # 求伪逆矩阵 2 * 90
        # a_mat_inv = np.linalg.pinv(a_mat)
        
        # 2 * 1
        x = np.dot(self.a_mat_pinv, phase)

        phaseSlope = x[0]
        # phaseCons = x[1]
        # calibration = np.exp(1j * (-phaseSlope * np.tile(self.subCarrierIndex, rxAntennaNum).reshape(3, -1) - phaseCons * np.ones((rxAntennaNum, subCarrierNum))))
        calibration = np.exp(1j * (-phaseSlope * np.tile(self.subCarrierIndex, rxAntennaNum).reshape(3, -1)))
        csi = csi * calibration
        return csi

    def signalNumEstimate(self, numList):
        num = sorted(numList, reverse=True)
        maxCount, count, est = 0, 0, num[0]
        for i in range(len(num) - 1):
            if num[i] == num[i + 1]:
                count += 1
            else:
                if count > maxCount:
                    maxCount = count
                    est = num[i]
                count = 0
        return est

    def getScaledSpectrum(self, spectrum):
        scaledSpectrum = spectrum * spectrum / sum(spectrum * spectrum)
        return scaledSpectrum

    def getAoASpectrum(self, CSIMatrix):

        if self.usePCA == 1:
            MUSICSignalNum = []
            Qn, sourceNum = self.getNoiseMat(CSIMatrix)
            eachSpectrum = self.getMUSIC(Qn, self.angleSteps, centerFrequency)  # timestamp * Nrx
            MUSICSignalNum.append(sourceNum)
            MUSICSpectrum = self.getScaledSpectrum(eachSpectrum)
            AoASpectrum = np.array(MUSICSpectrum)
            sigNumEst = self.signalNumEstimate(MUSICSignalNum)
        else:
            MUSICSignalNum = []
            Qn, sourceNum = self.getNoiseMat1(CSIMatrix)
            eachSpectrum = self.getMUSIC(Qn, self.angleSteps, centerFrequency)  # timestamp * Nrx
            MUSICSignalNum.append(sourceNum)
            MUSICSpectrum = self.getScaledSpectrum(eachSpectrum)
            AoASpectrum = np.array(MUSICSpectrum)
            sigNumEst = self.signalNumEstimate(MUSICSignalNum)

        return AoASpectrum, sigNumEst  # all subCarrier

    def widar2(self, csi):
        csi = csi.reshape(len(csi), -1)
        csi_amplitude = np.mean(abs(csi), axis=0)
        csi_variance = np.sqrt(np.var(abs(csi), axis=0))
        csi_ratio = np.divide(csi_amplitude, csi_variance)
        ant_ratio = np.mean(csi_ratio.reshape(30, 3), axis=0)
        midx = np.argmax(ant_ratio)
        csi_ref = np.tile(csi[:, midx * 30: (midx + 1) * 30], 3)

        alpha_all = 0
        for jj in range(len(csi[0])):
            alpha = np.min(abs(csi[:, jj]))
            alpha_all += alpha
            csi[:, jj] = (abs(csi[:, jj]) - alpha) * np.exp(1j * np.angle(csi[:, jj]))

        beta = alpha_all / len(csi[0]) * 1000
        for jj in range(len(csi_ref[0])):
            csi_ref[:, jj] = (abs(csi_ref[:, jj]) + beta) * np.exp(1j * np.angle(csi_ref[:, jj]))
        csi_mul = csi * csi_ref.conjugate()
        csi_mul = csi_mul.reshape(len(csi_mul), 3, 30)
        return csi_mul

    def readFile(self, *args, begin=None, end=None):
        # read file
        file1 = args[0]
        file2 = args[2]
        fileLen1 = args[1]
        fileLen2 = args[3]
        # extract the interested CSI segment
        if begin != None and end != None:
            file1 = file1[begin:end]
            file2 = file2[begin:end]
            fileLen1 = len(file1)
            fileLen2 = len(file2)
        windowNow = 0

        print("file1 len: {} and file2 len: {}".format(fileLen1, fileLen2))
        
        # the format of csi data：slideWindowLen * rxAntennaNum * subCarrierNum
        CSIMatrix1 = np.zeros([self.slideWindowLen, rxAntennaNum, subCarrierNum], dtype=complex)
        
        
        # the target angle
        angle_list = []
        angle_list_index = []
        
        
        # iteratively process the entire CSI segment
        while windowNow + self.slideWindowLen <= fileLen1:  # two Receiver file
            if windowNow == 0:  # when CSIOverlap1 is empty
                CSIMatrix1 = self.fillCSIMatrix(file1[0: self.slideWindowLen], file2[0: self.slideWindowLen])
            else:
                CSIMatrix1[:self.overlapLength, :, :] = CSIMatrix1[-self.overlapLength:, :, :]
                CSIMatrix1[self.overlapLength:, :, :] = self.fillCSIMatrix(file1[windowNow + self.overlapLength: windowNow + self.slideWindowLen])
            
            
            angle_list_index.append([windowNow, windowNow + self.slideWindowLen])
            

            if self.usePCA == 1:
                CSIMatrix = np.zeros([CSIMatrix1.shape[0], CSIMatrix1.shape[1]], dtype=complex)
                for csi in range(len(CSIMatrix1)):
                    CSIMat_temp = PCAtest(CSIMatrix1[csi, :, :], 1)
                    # CSIMat_temp = np.mean(CSIMatrix1[csi, :, :], axis=1)
                    CSIMatrix[csi, :] = CSIMat_temp.reshape(-1)

                AoASpectrumRx1, peakNumEst1 = self.getAoASpectrum(CSIMatrix)
                index_tem = self.findPeak1D(AoASpectrumRx1, peakNumEst1)
                AoA = self.angleSteps[index_tem] / np.pi * 180
                spectrumToShow = AoASpectrumRx1
            else:
                AoASpectrumRx1, peakNumEst1 = self.getAoASpectrum(CSIMatrix1)
                index_tem = self.findPeak1D(AoASpectrumRx1, peakNumEst1)
                AoA = self.angleSteps[index_tem] / np.pi * 180 
                spectrumToShow = AoASpectrumRx1

            if len(AoA) > 0:
                AoA.sort()
                angle_list.append(AoA[-1])
            else:
                angle_list.append(None)
            
            
            print("Estimating signal number on " + str((file1[int((windowNow + self.slideWindowLen) / 2)].timestamp_low
                                                        - file1[0].timestamp_low) / 1000000.0) + "s: " + str(peakNumEst1) + " " + str(len(AoA)))
            print(list(map(lambda x: float("%.1f" % x), AoA)))
            angleStepsDeg = self.angleSteps / np.pi * 180

            
            
            if self.showimage:
                plt.plot(angleStepsDeg, spectrumToShow)
                plt.xlabel('AoA / degree')
                plt.ylabel('Spectrum')
                plt.title("spectrum on RX " + str((windowNow + self.slideWindowLen) / sampleFrequency / 2) + "s")
                plt.show()

            windowNow += self.stepLength
        
        # 对于是None的角度进行插值
        for i, angle in enumerate(angle_list):
            if angle == None:
                if i == 0:
                    j = i + 1
                    while (j != len(angle_list) - 1 and angle_list[j] == None):
                        j += 1
                    angle_list[i] = angle_list[j]
                elif i == len(angle_list) - 1:
                    angle_list[i] = angle_list[i - 1]
                else:
                    j = i + 1
                    while (j != len(angle_list) - 1 and angle_list[j] == None):
                        j += 1
                    if j == len(angle_list) - 1:
                        angle_list[i] = angle_list[i - 1]
                    else:
                        angle_list[i] = (angle_list[i - 1] + angle_list[j]) / 2
                        

        angle_list = np.array(angle_list)
        angle_list_index = np.array(angle_list_index)
        plt.ylim(-31, 31)
        plt.plot(angle_list_index, angle_list)
        plt.plot(angle_list_index, [0] * len(angle_list), linestyle='dashed', color='red', linewidth=1)
        plt.show()
        
        return angle_list_index, angle_list
        
        
        


# def from_aoa_to_coordinate(aoa_rx1, aoa_rx2, dis):
#     aoa_1 = [j * np.pi / 180 for j in aoa_rx1]
#     aoa_2 = [j * np.pi / 180 for j in aoa_rx2]
#     xx, yy = [], []
#     for i in range(len(aoa_rx1)):
#         hh = dis / (1.0 / np.tan(aoa_1[i]) + 1.0 / np.tan(aoa_2[i]))
#         ww = hh / np.tan(aoa_1[i])
#         xx.append(ww)
#         yy.append(hh)
#     return xx, yy


if __name__ == '__main__':
    rx1 = Track(search_interval=(-np.pi/2, np.pi/2), slide_window=0.5,
        filename="C:/Users/13947/Desktop/static_log1.dat", initial_angle=0, use_mdl=0, use_pca=1, \
            show_image=1, overlay=0.5, begin=None, end=None)
