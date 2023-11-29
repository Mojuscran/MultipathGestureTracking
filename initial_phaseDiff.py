# created in 2023/11/11
# correct phase ambiguity through static state


# align the antennas of the two routers one meter apart
# 
import read_bf_file
import numpy as np
import matplotlib.pyplot as plt

sampleFrequency = 200           # Hertz
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

class initial_phaseDiff:
    def __init__(self, filePath1, filePath2, begin=0, end=-1):
        self.filePath1 = filePath1
        self.filePath2 = filePath2
        self.file_len1 = 0
        self.file_len2 = 0
        # len * 1
        self.timestamp1 = []
        self.timestamp2 = []
        # len * 30
        self.antennaPair_One = []
        self.antennaPair_Two = []
        self.antennaPair_Three = []
        self.antennaPair_Four = []
        self.antennaPair_Five = []
        self.antennaPair_Six = []
        
        # 1 * 30
        self.staticPhaseDifference12 = []
        self.staticPhaseDifference23 = []
        self.staticPhaseDifference45 = []
        self.staticPhaseDifference46 = []
        # read CSI file
        self.readFile(self.filePath1, self.filePath2, begin, end)
        self.getInitialPhaseDiff()
        # print("the static phase difference of antennaPair_One and antennaPair_Two:{}".format(self.staticPhaseDifference12))
        # print("the static phase difference of antennaPair_One and antennaPair_Three:{}".format(self.staticPhaseDifference13))
        # print("the static phase difference of antennaPair_Four and antennaPair_Five:{}".format(self.staticPhaseDifference45))
        # print("the static phase difference of antennaPair_Four and antennaPair_Six:{}".format(self.staticPhaseDifference46))
        
        
        
    # read file
    def readFile(self, filePath1, filePath2, begin=0, end=-1):
        file1, self.file_len1 = read_bf_file.read_file(filePath1)
        file2, self.file_len2 = read_bf_file.read_file(filePath2)
        if begin != 0 or end != -1:
            file1 = file1[begin:end]
            file2 = file2[begin:end]
            self.file_len1 = len(file1)
            self.file_len2 = len(file2)
        startTime1 = file1[0].timestamp_low
        startTime2 = file2[0].timestamp_low
        print("Length of packets1: {}".format(self.file_len1) + "    Start timestamp:" + str(startTime1))
        print("Length of packets2: {}".format(self.file_len2) + "    Start timestamp:" + str(startTime2))
        # item: 30 * 1 * 3
        for item in file1:
            self.timestamp1.append((item.timestamp_low - startTime1) / 1000000.0)
            for eachcsi in range(0, subCarrierNum):
                self.antennaPair_One.append(item.csi[eachcsi][0][0])
                self.antennaPair_Two.append(item.csi[eachcsi][0][1])
                self.antennaPair_Three.append(item.csi[eachcsi][0][2])

        for item in file2:
            self.timestamp2.append((item.timestamp_low - startTime2) / 1000000.0)
            for eachcsi in range(0, subCarrierNum):
                self.antennaPair_Four.append(item.csi[eachcsi][0][0])
                self.antennaPair_Five.append(item.csi[eachcsi][0][1])
                self.antennaPair_Six.append(item.csi[eachcsi][0][2])


        self.antennaPair_One = np.reshape(self.antennaPair_One, (self.file_len1, subCarrierNum))
        self.antennaPair_Two = np.reshape(self.antennaPair_Two, (self.file_len1, subCarrierNum))
        self.antennaPair_Three = np.reshape(self.antennaPair_Three, (self.file_len1, subCarrierNum))
        self.antennaPair_Four = np.reshape(self.antennaPair_Four, (self.file_len2, subCarrierNum))
        self.antennaPair_Five = np.reshape(self.antennaPair_Five, (self.file_len2, subCarrierNum))
        self.antennaPair_Six = np.reshape(self.antennaPair_Six, (self.file_len2, subCarrierNum))
    
    def getInitialPhaseDiff(self):
        sixAntennasData = np.array([self.antennaPair_One, self.antennaPair_Two, self.antennaPair_Three, \
            self.antennaPair_Four, self.antennaPair_Five, self.antennaPair_Six])
        # 6 * 30
        np.save('phaseReference', np.nanmean(sixAntennasData, axis=1))
        # len * 30
        self.ret1 = np.divide(self.antennaPair_One, self.antennaPair_Three)
        self.ret2 = np.divide(self.antennaPair_Two, self.antennaPair_Three)
        self.ret3 = np.divide(self.antennaPair_Four, self.antennaPair_Five)
        self.ret4 = np.divide(self.antennaPair_Four, self.antennaPair_Six)
        # 30
        self.PCAret1 = np.nanmean(self.ret1, axis=0)
        self.PCAret2 = np.nanmean(self.ret2, axis=0)
        self.PCAret3 = np.nanmean(self.ret3, axis=0)
        self.PCAret4 = np.nanmean(self.ret4, axis=0)
        
        self.staticPhaseDifference12 = np.mean(np.angle(self.PCAret1))
        self.staticPhaseDifference13 = np.mean(np.angle(self.PCAret2))
        self.staticPhaseDifference45 = np.mean(np.angle(self.PCAret3))
        self.staticPhaseDifference46 = np.mean(np.angle(self.PCAret4))
        # 4 * 30
        sixAntennasDataDiff = np.array([self.staticPhaseDifference13, self.staticPhaseDifference23, \
            self.staticPhaseDifference45, self.staticPhaseDifference46])
        np.save('phaseDiffReference', sixAntennasDataDiff)


if __name__ == '__main__':
    pass