# created in 2024/07/01

# based on multiple signal classification algorithem to estimate tof
# calibrate To by using bi-directional transceiver element

from read_data import *
import numpy as np
import matplotlib.pyplot as plt

class ToFTrack(object):
    def __init__(self,
                 searchInterval=(1, 4),                           # ToF search interval(unit is ns)
                 slideWindow=0.4,                                  # sliding window size(unit is s)
                 csi=None,                                         # csi data
                 signalNum=1,                                      # signal source number
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
        
        
        self.ToFStepsNum = 1000
        # ToF iteration stride
        self.ToFStepLen = (self.searchInterval[1] - self.searchInterval[0]) / self.ToFStepsNum
        self.ToFSteps = np.arange(self.searchInterval[0], self.searchInterval[1], self.ToFStepLen, dtype=float)
        
        # size of the sliding window(number of packets)
        self.slideWindowLen = int(self.slideWindow * SAMPLE_RATE)
        # stride of the sliding window
        self.stepLength = int((1 - self.overlay) * self.slideWindowLen)
        # length of overlap
        self.overlapLength = self.slideWindowLen - self.stepLength
        self.start(csiData=csi, begin=begin, end=end)

    def get_steering_vector(self, delay):
        steeringVec = np.expand_dims(np.exp(-1j * 2 * np.pi * self.subCarrierIndex * DELTA_FREQUENCY * delay * 1e-9), axis=1)
        return steeringVec

    def get_MUSIC(self, noiseMultiply, ToFSteps):
        # generate MUSIC spectrum
        spectrum = np.zeros(self.ToFStepsNum, dtype=float)
        for i in range(self.ToFStepsNum):
            steeringVec = self.get_steering_vector(ToFSteps[i])
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
        # shape: N * Nsubcarrier
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

    def get_ToF_spectrum(self, CSIMatrix):
        if self.usePCA == 1:
            Qn = self.get_noise_mat(CSIMatrix)
            # shape: ToFStepsNum
            eachSpectrum = self.get_MUSIC(Qn, self.ToFSteps)
            MUSICSpectrum = self.get_scaled_spectrum(eachSpectrum)
        else:
            Qn = self.getNoiseMat1(CSIMatrix)
            eachSpectrum = self.get_MUSIC(Qn, self.ToFSteps)  # timestamp * Nrx
            MUSICSpectrum = self.get_scaled_spectrum(eachSpectrum)
        return MUSICSpectrum

    def start(self, csiData, begin=None, end=None):
        # csiData size:  N * SIZE_OF_RECEIVING_ANTENNA * SIZE_OF_SUBCARRIER        
        # the format of temp csi data：slideWindowLen * rxAntennaNum * subCarrierNum
        tmpCSIMatrix = np.zeros([self.slideWindowLen, SIZE_OF_RECEIVING_ANTENNA, SIZE_OF_SUBCARRIER], dtype=complex)
        csiData = csiData[begin:end]
        windowNow = 0
        # the target ToF
        self.ToFList = []
        self.ToFListIndex = []
        
        if self.refFilePath is not None:
            ref = np.load(self.refFilePath)
        # process the entire CSI segment iteratively
        while windowNow + self.slideWindowLen <= csiData.shape[0]:
            tmpCSIMatrix = csiData[windowNow:windowNow + self.slideWindowLen]
            if self.refFilePath is not None:
                tmpCSIMatrix *= np.exp(-1j * np.angle(np.expand_dims(ref, axis=0)))
            if self.refFilePath is None and self.usePhaseCalibration:
                tmpCSIMatrix = self.phase_calibration(tmpCSIMatrix)
            self.ToFListIndex.append([windowNow, windowNow + self.slideWindowLen])
            if self.usePCA:
                CSIMatrix = np.zeros([tmpCSIMatrix.shape[0], tmpCSIMatrix.shape[2]], dtype=complex)
                # for csi in range(len(tmpCSIMatrix)):
                #     # CSIMat_temp = readCSIData.principal_components_analysis(CSIMatrix1[csi, :, :], dim=1)
                #     CSIMat_temp = np.mean(tmpCSIMatrix[csi, :, :], axis=1)
                #     CSIMatrix[csi, :] = CSIMat_temp.reshape(-1)
                CSIMatrix = tmpCSIMatrix[:, 0]
                ToFSpectrum = self.get_ToF_spectrum(CSIMatrix)
                ToFIndex = self.find_peak_1D(ToFSpectrum, self.signalNum)
                delay = self.ToFSteps[ToFIndex]
                spectrumToShow = ToFSpectrum
            else:
                ToFSpectrum = self.get_ToF_spectrum(tmpCSIMatrix)
                index_tem = self.find_peak_1D(ToFSpectrum, self.signalNum)
                delay = self.ToFSteps[index_tem]
                spectrumToShow = ToFSpectrum

            if len(delay) > 0:
                delay.sort()
                self.ToFList.append(delay[-1])
            else:
                self.ToFList.append(None)
            
            print(list(map(lambda x: float("%.1f" % x), delay)))
            if self.showImage:
                plt.plot(self.ToFSteps, spectrumToShow)
                plt.xlabel('ToF/ns')
                plt.ylabel('Spectrum')
                plt.grid()
                plt.show()
            windowNow += self.stepLength                        

        self.ToFList = np.array(self.ToFList)
        self.ToFListIndex = np.array(self.ToFListIndex)
        plt.ylim(self.searchInterval[0], self.searchInterval[1])
        plt.plot(self.ToFListIndex, self.ToFList)
        plt.show()

if __name__ == '__main__':
    rx1 = ToFTrack(search_interval=(-np.pi/2, np.pi/2), slide_window=0.5,
        filename1="C:/Users/13947/Desktop/static_log1.dat", signalNum=1, use_pca=1, \
            show_image=1, overlay=0.5)

