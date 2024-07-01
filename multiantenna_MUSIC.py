# created in 2023/11/10

# based on "Multiple Emitter Location and Signal Parameter Estimation"
# and "Phaser: Enabling Phased Array Signal Processing on Commodity WiFi Access Points"
# the objective is to explore a five-antenna multi-signal classification algorithm

from read_data import *
import numpy as np
import matplotlib.pyplot as plt

class AoATrack(object):
    def __init__(self,
                 searchInterval=(-np.pi / 2, np.pi / 2),           # angle search interval(radian measure)
                 slideWindow=0.4,                                  # sliding window size(unit is s)
                 csi=None,                                         # csi data
                 initialAngle=0,                                   # the initial angle
                 signalNum=1,                                      # whether to use mdl
                 usePCA=True,                                      # whether to use PCA
                 overlay=0.5,                                      # overlap ratio
                 showImage=False,                                  # whether to show pseudospectral graph
                 begin=None,                                       # the start index
                 end=None,                                         # the end index
                 ref=None,                                         # the reference file path                                         
                 usePhaseCalibration=False,                        # whether to calibrate phase
                 ):
        
        self.searchInterval = searchInterval
        self.slideWindow = slideWindow
        self.initialAngle = initialAngle
        self.signalNum = signalNum
        self.usePCA = usePCA
        self.overlay = overlay
        self.showImage = showImage
        self.subCarrierIndex = INDEX_OF_SUBCARRIER40
        self.refFilePath = ref
        self.usePhaseCalibration = usePhaseCalibration
        
        # shape: 1 * SIZE_OF_SUBCARRIER
        aMatrix = np.expand_dims(self.subCarrierIndex, axis=0)
        # shape: 2 * SIZE_OF_SUBCARRIER
        aMatrix = np.append(aMatrix, np.ones((1, SIZE_OF_SUBCARRIER)), axis=0)
        # Compute the (Moore-Penrose) pseudo-inverse of aMatrix 
        # shape: SIZE_OF_SUBCARRIER * 2
        self.aMatrixPinv = np.linalg.pinv(aMatrix)
        
        
        self.angleStepsNum = 10000
        # angle iteration stride
        self.angleStepLen = (self.searchInterval[1] - self.searchInterval[0]) / self.angleStepsNum
        self.angleSteps = np.arange(self.searchInterval[0], self.searchInterval[1], self.angleStepLen, dtype=float)
        # convert the search start point and end point to the degree measure
        self.angleIntervalDeg = (self.searchInterval[0] * 180 / np.pi, self.searchInterval[1] * 180 / np.pi)
        
        
        # size of the sliding window(number of packets)
        self.slideWindowLen = int(self.slideWindow * SAMPLE_RATE)
        # stride of the sliding window
        self.stepLength = int((1 - self.overlay) * self.slideWindowLen)
        # length of overlap
        self.overlapLength = self.slideWindowLen - self.stepLength
        self.start(csiData=csi, begin=begin, end=end)

    def get_steering_vector(self, angle, frequency):
        steeringVec = np.empty((SIZE_OF_RECEIVING_ANTENNA, 1), dtype=complex)
        for i in range(SIZE_OF_RECEIVING_ANTENNA):
            delay = ANTDISTANCE * i * np.sin(angle) / SPEED_OF_LIGHT
            steeringVec[i] = np.exp(-1j * 2 * np.pi * frequency * delay)
        return steeringVec

    def get_MUSIC(self, noiseMultiply, angleSteps, fc):
        # generate MUSIC spectrum
        spectrum = np.zeros(self.angleStepsNum, dtype=float)
        for i in range(self.angleStepsNum):
            steeringVec = self.get_steering_vector(angleSteps[i], fc)
            denominator = np.sum(
                np.dot(np.dot(steeringVec.conjugate().transpose(), noiseMultiply),
                       steeringVec), axis=1)  # matrix multiplication
            spectrum[i] = 1 / abs(denominator)
        return spectrum

    def getNoiseMat1(self, matrix):
        # slidelen * 3 * 30
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
        qn = u[:, self.signalNum:]
        noiseMultiply = np.dot(qn, qn.conjugate().transpose())
        return noiseMultiply

    def get_noise_mat(self, matrix):
        # shape: N * Nrx
        mat = np.asarray(matrix)  
        cor = np.dot(mat.transpose(), mat.conjugate())
        eig, u_tmp = np.linalg.eig(cor)
        eig = np.abs(eig)
        un = np.argsort(-eig)
        eig = -np.sort(-eig)
        u = u_tmp[:, un[:]]

        qn = u[:, self.signalNum:]
        noiseMultiply = np.dot(qn, qn.conjugate().transpose())
        return noiseMultiply

    # TODO modify this algorithm to be adaptive
    def find_peak_1D(self, spectrum, sourceNum):
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

    # remove the effect of STO by linear fitting
    def phase_calibration(self, csi):
        # the format of csi data: slideWindowLen * rxAntennaNum * subCarrierNum
        phaseUnwrapped = np.unwrap(np.angle(csi))
        # shape: slideWindowLen * rxAntennaNum * 2
        x = np.dot(phaseUnwrapped, self.aMatrixPinv)
        phaseSlope = x[:, :, 0]
        # phaseCons = x[:, :, 1]
        # calibration = csi * np.exp(1j * (-phaseSlope * np.tile(self.subCarrierIndex, rxAntennaNum).reshape(3, -1) - phaseCons * np.ones((rxAntennaNum, subCarrierNum))))
        calibration = csi * np.exp(-1j * (np.expand_dims(phaseSlope, axis=-1) * self.subCarrierIndex))
        return calibration

    def get_scaled_spectrum(self, spectrum):
        scaledSpectrum = spectrum * spectrum / np.sum(spectrum * spectrum)
        return scaledSpectrum

    def get_AoA_spectrum(self, CSIMatrix):
        if self.usePCA == 1:
            Qn = self.get_noise_mat(CSIMatrix)
            # shape: angleStepsNum
            eachSpectrum = self.get_MUSIC(Qn, self.angleSteps, CENTER_OF_FREQUENCY)
            MUSICSpectrum = self.get_scaled_spectrum(eachSpectrum)
        else:
            Qn = self.getNoiseMat1(CSIMatrix)
            eachSpectrum = self.get_MUSIC(Qn, self.angleSteps, CENTER_OF_FREQUENCY)  # timestamp * Nrx
            MUSICSpectrum = self.get_scaled_spectrum(eachSpectrum)
        return MUSICSpectrum

    def start(self, csiData, begin=None, end=None):
        # csiData size:  N * SIZE_OF_RECEIVING_ANTENNA * SIZE_OF_SUBCARRIER        
        # the format of temp csi data：slideWindowLen * rxAntennaNum * subCarrierNum
        CSIMatrix1 = np.zeros([self.slideWindowLen, SIZE_OF_RECEIVING_ANTENNA, SIZE_OF_SUBCARRIER], dtype=complex)
        csiData = csiData[begin:end]
        windowNow = 0
        # the target angle
        self.angleList = []
        self.angleListIndex = []
        
        if self.refFilePath is not None:
            ref = np.load(self.refFilePath)
        # iteratively process the entire CSI segment
        while windowNow + self.slideWindowLen <= csiData.shape[0]:
            CSIMatrix1 = csiData[windowNow:windowNow + self.slideWindowLen]
            if self.refFilePath is not None:
                CSIMatrix1 *= np.exp(-1j * np.angle(np.expand_dims(ref, axis=0)))
            if self.refFilePath is None and self.usePhaseCalibration:
                CSIMatrix1 = self.phase_calibration(CSIMatrix1)
            self.angleListIndex.append([windowNow, windowNow + self.slideWindowLen])
            if self.usePCA:
                CSIMatrix = np.zeros([CSIMatrix1.shape[0], CSIMatrix1.shape[1]], dtype=complex)
                for csi in range(len(CSIMatrix1)):
                    # CSIMat_temp = readCSIData.principal_components_analysis(CSIMatrix1[csi, :, :], dim=1)
                    CSIMat_temp = np.mean(CSIMatrix1[csi, :, :], axis=1)
                    CSIMatrix[csi, :] = CSIMat_temp.reshape(-1)

                AoASpectrumRx1 = self.get_AoA_spectrum(CSIMatrix)
                AoAIndex = self.find_peak_1D(AoASpectrumRx1, self.signalNum)
                AoA = self.angleSteps[AoAIndex] / np.pi * 180
                spectrumToShow = AoASpectrumRx1
            else:
                AoASpectrumRx1 = self.get_AoA_spectrum(CSIMatrix1)
                index_tem = self.find_peak_1D(AoASpectrumRx1, self.signalNum)
                AoA = self.angleSteps[index_tem] / np.pi * 180 
                spectrumToShow = AoASpectrumRx1

            if len(AoA) > 0:
                AoA.sort()
                self.angleList.append(AoA[-1])
            else:
                self.angleList.append(None)
            
            print(list(map(lambda x: float("%.1f" % x), AoA)))
            angleStepsDeg = self.angleSteps / np.pi * 180

            if self.showImage:
                plt.plot(angleStepsDeg, spectrumToShow)
                plt.xlabel('AoA / degree')
                plt.ylabel('Spectrum')
                plt.grid()
                plt.show()
            windowNow += self.stepLength
        
        # 对于是None的角度进行插值
        for i, angle in enumerate(self.angleList):
            if angle == None:
                if i == 0:
                    j = i + 1
                    while (j != len(self.angleList) - 1 and self.angleList[j] == None):
                        j += 1
                    self.angleList[i] = self.angleList[j]
                elif i == len(self.angleList) - 1:
                    self.angleList[i] = self.angleList[i - 1]
                else:
                    j = i + 1
                    while (j != len(self.angleList) - 1 and self.angleList[j] == None):
                        j += 1
                    if j == len(self.angleList) - 1:
                        self.angleList[i] = self.angleList[i - 1]
                    else:
                        self.angleList[i] = (self.angleList[i - 1] + self.angleList[j]) / 2
                        

        self.angleList = np.array(self.angleList)
        self.angleListIndex = np.array(self.angleListIndex)
        plt.ylim(-31, 31)
        plt.plot(self.angleListIndex, self.angleList)
        plt.plot(self.angleListIndex, [0] * len(self.angleList), linestyle='dashed', color='red', linewidth=1)
        plt.show()

if __name__ == '__main__':
    rx1 = Track(search_interval=(-np.pi/2, np.pi/2), slide_window=0.5,
        filename1="C:/Users/13947/Desktop/static_log1.dat", use_mdl=0, use_pca=1, \
            show_image=1, overlay=0.5)
