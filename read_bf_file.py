import numpy as np
import os
import sys
import struct
import ctypes


unpack_buffer = ctypes.cdll.LoadLibrary(r'utils\unpack_buffer.dll')
# from .csi import WifiCsi
class WifiCsi:
    def __init__(self, args, csi):
        self.timestamp_low = args[0]
        self.rssi_a = args[1]
        self.rssi_b = args[2]
        self.rssi_c = args[3]
        self.noise = args[4]
        self.agc = args[5]
        self.csi = csi
        
def read_file(file_path, Tx=1, Rx=3):
    # 读取文件的长度
    length = os.path.getsize(file_path)
    # 当前读取位置
    cur = 0
    # 数据解析的次数
    count = 0
    # 存储解析后的数据
    csi_data = []
    
    # 数据读取
    with open(file_path, 'rb') as f:
        while(cur < (length - 3)):
            # 通过使用struct.unpack函数从文件中读取特定长度的数据，并根据读取到的数据进行不同的处理
            filed_length = struct.unpack("!H", f.read(2))[0]
            code = struct.unpack("!B", f.read(1))[0]
            cur += 3
            # 如果读取到的数据的第一个字节等于187，说明接下来的数据是要解析的数据块
            if code == 187:
                # data = []
                # for _ in range(filed_length - 1):
                    # data.append(struct.unpack("!B", f.read(1))[0])
                real = (ctypes.c_int * (30 * Tx * Rx))()
                img = (ctypes.c_int * (30 * Tx * Rx))()
                timestamp_low = (ctypes.c_uint * 1)()
                rssi = (ctypes.c_char * 5)()
                payload = ctypes.c_buffer(f.read(filed_length - 1))
                unpack_buffer.unpack_buffer(payload, real, img, timestamp_low, rssi)
                cur = cur + filed_length - 1
                args = [timestamp_low[0], struct.unpack("!B", rssi[0])[0], \
                    struct.unpack("!B", rssi[1])[0], struct.unpack("!B", rssi[2])[0], \
                        struct.unpack("!b", rssi[3])[0], struct.unpack("!B", rssi[4])[0]]
                csi_data.append(WifiCsi(args, np.array(real) + 1j * np.array(img)))
                # csi_data.append(read_bfee(data))
                count += 1
            else:
                f.seek(filed_length - 1, 1)
                cur = cur + filed_length - 1
            
    return csi_data, count

def read_csi(file_path):
    data = read_file(file_path)
    frm = data.__len__()
    ant = 3
    sbc = 30
    csi = np.zeros((frm,sbc,1,ant),dtype=complex)
    for i in range(frm):
        csi[i,:,:,:] = data[i].csi
    return csi.reshape(frm,sbc,ant).transpose(0,2,1)

def read_csi_rssi(file_path):
    data = read_file(file_path)
    frm = data.__len__()
    ant = 3
    sbc = 30
    csi = np.zeros((frm,sbc,1,ant),dtype=complex)
    rssi = np.zeros((frm,ant),dtype=float)
    for i in range(frm):
        csi[i,:,:,:] = data[i].csi
        rssi[i,:] = [data[i].rssi_a, data[i].rssi_b, data[i].rssi_c]
    return csi.reshape(frm,sbc,ant).transpose(0,2,1), rssi