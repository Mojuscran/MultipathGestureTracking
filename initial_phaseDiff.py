# created in 2023/11/11
# correct phase ambiguity through static state

import read_bf_file
import numpy as np
from PCA import PCAtest
import matplotlib.pyplot as plt

class initial_phaseDiff:
    def __init__(self, filePath1, filePath2, begin=None, end=None):
        self.filePath1 = filePath1
        self.filePath2 = filePath2
        # read CSI file
        file_r1, file_length1 = read_bf_file.read_file(self.filePath1)
        file_r2, file_length2 = read_bf_file.read_file(self.filePath2)

    def getInitialPhaseDiff(self):
        pass




if __name__ == '__main__':
    pass