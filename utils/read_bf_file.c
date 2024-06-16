// 2024/04/26
// 
// unpack CSI data by c

#include <stdio.h>
#include <stdint.h>


char get_bit_num(char in_num, char data_length)
{
    char max_value = (1 << (data_length - 1)) - 1;
    char out_num = 0;
    if (!(- max_value - 1 <= in_num && in_num <= max_value))
        out_num = (in_num + max_value + 1) % (2 * (max_value + 1)) - max_value - 1;
    else
        out_num = in_num;
    return out_num;
}


void unpack_buffer(uint8_t *str, int32_t *real, int32_t *imag, uint32_t *timestamp_low, uint8_t *rssi)
{
    timestamp_low[0] = (unsigned int)str[0] + ((unsigned int)str[1] << 8) + ((unsigned int)str[2] << 16) + ((unsigned int)str[3] << 24);
    

    unsigned char Nrx = str[8];
    unsigned char Ntx = str[9];
    rssi[0] = str[10];
    rssi[1] = str[11];
    rssi[2] = str[12];
    rssi[3] = get_bit_num(str[13],8);
    rssi[4] = str[14];
    
    unsigned char antenna_sel = str[15];
    unsigned char perm[3] = {0, 0, 0};
    perm[0] = ((antenna_sel) & 0x3) + 1;
    perm[1] = ((antenna_sel >> 2) & 0x3) + 1;
    perm[2] = ((antenna_sel >> 4) & 0x3) + 1;


    int index = 0;
    int remainder = 0;
    for (int i = 0; i < 30; ++i)
    {
        index += 3;
        remainder = index % 8;
        for (int j = 0; j < Nrx; ++j)
        {
            for (int k = 0; k < Ntx; ++k)
            {
                real[i + 30 * (perm[j] - 1) + (30 * Nrx) * k] = get_bit_num(str[20 + index / 8] >> remainder, 8) | get_bit_num((str[21 + index / 8] << (8 - remainder)), 8);
                imag[i + 30 * (perm[j] - 1) + (30 * Nrx) * k] = get_bit_num(str[21 + index / 8] >> remainder, 8) | get_bit_num((str[22 + index / 8] << (8 - remainder)), 8);
                index += 16;
            }
        }
    }
}