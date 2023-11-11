# created in 2023/11/11
# correct phase ambiguity through static state

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
    def __init__(self, filePath1, filePath2, begin=None, end=None):
        self.filePath1 = filePath1
        self.filePath2 = filePath2
        # read CSI file
        file_r1, file_length1 = read_bf_file.read_file(self.filePath1)
        file_r2, file_length2 = read_bf_file.read_file(self.filePath2)
        
    # read file
    def readFile(self, filePath1, filePath2):
        file, file_len = read_bf_file.read_file(filepath)
        self.length = file_len
        startTime = file[0].timestamp_low
        print("Length of packets: {}".format(file_len) + "    Start timestamp:" + str(startTime))
        # item: 30 * 1 * 3
        for item in file:
            self.timestamp = np.append(self.timestamp, (item.timestamp_low - startTime) / 1000000.0)
            for eachcsi in range(0, SIZE_OF_SUBCARRIER):
                ''''extract csi complex value for each antenna pair with shape ( len(file) * 30), i.e., packet number * subcarrier number'''
                self.antennaPair_One.append(item.csi[eachcsi][0][0])
                self.antennaPair_Two.append(item.csi[eachcsi][0][1])
                self.antennaPair_Three.append(item.csi[eachcsi][0][2])

        self.antennaPair_One = np.reshape(self.antennaPair_One, (self.length, 30))
        self.antennaPair_Two = np.reshape(self.antennaPair_Two, (self.length, 30))
        self.antennaPair_Three = np.reshape(self.antennaPair_Three, (self.length, 30))

        self.ret1 = np.divide(self.antennaPair_One, self.antennaPair_Two)
        self.ret2 = np.divide(self.antennaPair_Two, self.antennaPair_Three)
        
        # temp = np.array([self.antennaPair_One, self.antennaPair_Two, self.antennaPair_Three])
        # CSIMatrix = np.zeros([self.length, SIZE_OF_RECEIVING_ANTENNA], dtype=complex)
        # for eachCsi in range(self.length):
        #     CSIMat_temp = self.PCAtest(temp[:, eachCsi, :], 1)
        #     CSIMatrix[eachCsi, :] = CSIMat_temp.reshape(-1)

        # self.PCAret1 = np.divide(CSIMatrix[:, 0], CSIMatrix[:, 1])
        # self.PCAret2 = np.divide(CSIMatrix[:, 1], CSIMatrix[:, 2])
        self.PCAret1 = np.mean(self.ret1, axis=1)
        self.PCAret2 = np.mean(self.ret2, axis=1)
    
    

    def getInitialPhaseDiff(self):
        pass




if __name__ == '__main__':
    pass