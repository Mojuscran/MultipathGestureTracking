import numpy as np
import os
import sys
import struct
# from .csi import WifiCsi


class WifiCsi:
    def __init__(self, args, csi):
        self.timestamp_low = args[0]
        self.bfee_count = args[1]
        self.Nrx = args[2]
        self.Ntx = args[3]
        self.rssi_a = args[4]
        self.rssi_b = args[5]
        self.rssi_c = args[6]
        self.noise = args[7]
        self.agc = args[8]
        self.perm = args[9]
        self.rate = args[10]
        self.csi = csi
        
def get_bit_num(in_num, data_length):
    max_value = (1 << data_length - 1) - 1
    if not -max_value-1 <= in_num <= max_value:
        out_num = (in_num + (max_value + 1)) % (2 * (max_value + 1)) - max_value - 1
    else:
        out_num = in_num
    return out_num


def read_bfee(in_bytes):
    # 从输入字节中提取各种变量
    timestamp_low = in_bytes[0] + (in_bytes[1] << 8) + (in_bytes[2] << 16) + (in_bytes[3] << 24)
    bfee_count = in_bytes[4] + (in_bytes[5] << 8)
    # print(bfee_count)
    Nrx = in_bytes[8]
    Ntx = in_bytes[9]
    rssi_a = in_bytes[10]
    rssi_b = in_bytes[11]
    rssi_c = in_bytes[12]
    noise = get_bit_num(in_bytes[13],8)
    agc = in_bytes[14]
    antenna_sel = in_bytes[15]
    length = in_bytes[16] + (in_bytes[17] << 8)
    fake_rate_n_flags = in_bytes[18] + (in_bytes[19] << 8)
    calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8
    payload = in_bytes[20:]
    
    # if(length != calc_len)

    perm_size = 3
    perm = np.ndarray(perm_size, dtype=int)

    perm[0] = ((antenna_sel) & 0x3) + 1
    perm[1] = ((antenna_sel >> 2) & 0x3) + 1
    perm[2] = ((antenna_sel >> 4) & 0x3) + 1
    
    index = 0

    csi_size = (30, Ntx, Nrx)
    perm_csi = np.ndarray(csi_size, dtype=complex)

    for i in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx):
            for k in range(Ntx):
                pr = get_bit_num((payload[index // 8] >> remainder),8) | get_bit_num((payload[index // 8+1] << (8-remainder)),8)
                pi = get_bit_num((payload[(index // 8)+1] >> remainder),8) | get_bit_num((payload[(index // 8)+2] << (8-remainder)),8)
                perm_csi[i][k][perm[j] - 1] = complex(pr, pi)

                index += 16

    args = [timestamp_low, bfee_count, Nrx, Ntx, rssi_a,
            rssi_b, rssi_c, noise, agc, perm, fake_rate_n_flags]

    temp_wifi_csi = WifiCsi(args, perm_csi)
    return temp_wifi_csi

def read_file(file_path):
    # 读取文件的长度
    length = os.path.getsize(file_path)
    # 当前读取位置
    cur = 0
    # 数据解析的次数
    count = 0
    # 标志位
    broken_perm = 0
    # 固定列表
    triangle = [1, 3, 6]
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
                data = []
                for _ in range(filed_length - 1):
                    data.append(struct.unpack("!B", f.read(1))[0])


                cur = cur + filed_length - 1
                if len(data) != filed_length - 1:
                    break
                csi_data.append(read_bfee(data))
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