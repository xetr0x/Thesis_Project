#include "sensirion.h"

#include <stdio.h>
#include <math.h>

uint16_t bytetou16(uint8_t *bytes){
    return (uint16_t)bytes[0] << 8 | (uint16_t)bytes[1];
}
uint32_t bytetou32(uint8_t *bytes){
    return (uint32_t)bytes[0] << 24 | (uint32_t)bytes[1] << 16 | (uint32_t)bytes[2] << 8 | (uint32_t)bytes[3];
}
int32_t byteto32(uint8_t *bytes){
    return (int32_t) bytetou32(bytes);
}

float bytetofloat(uint8_t *bytes){
    union 
    {
        uint32_t value;
        float float32;
    }temp;
    temp.value = bytetou32(bytes);
    return temp.float32;
}

void u32tobyte(uint32_t val, uint8_t *bytes){
    bytes[0] = val >> 24;
    bytes[1] = val >> 16;
    bytes[2] = val >> 8;
    bytes[3] = val;
}
void u16tobyte(uint16_t val, uint8_t *bytes){
    bytes[0] = val >> 8;
    bytes[1] = val;
}

void n32tobyte(int32_t val, uint8_t *bytes){
    bytes[0] = val >> 24;
    bytes[1] = val >> 16;
    bytes[2] = val >> 8;
    bytes[3] = val;
}
void n16tobyte(int16_t val, uint8_t *bytes){
    bytes[0] = val >> 8;
    bytes[1] = val;
}
void floattobyte(float val, uint8_t *bytes){
    union 
    {
        uint32_t val32;
        float float32;
    }temp;
    temp.float32 = val;
    u32tobyte(temp.val32, bytes);
}
void cpbyte(uint8_t *og, uint8_t *new, uint16_t len){
    uint16_t i;
    for (size_t i = 0; i < len; i++)
    {
        new[i] = og[i];
    }
}

int8_t i2c_read(i2c_inst_t *i2c,uint8_t address, uint8_t *data, uint16_t count){
    int result = i2c_read_blocking(i2c, address, data, count, false);
    return (result == count) ? 0 : -1;
}
int8_t i2c_write(i2c_inst_t *i2c,uint8_t address, uint8_t *data, uint16_t count){
    int result = i2c_write_blocking(i2c, address, data, count, false);
    return (result == count) ? 0 : -1;
}

uint8_t crc_gen(const uint8_t* data, uint16_t count) {
    uint16_t current_byte;
    uint8_t crc = CRC8_INIT;
    uint8_t crc_bit;

    /* calculates 8-Bit checksum with given polynomial */
    for (current_byte = 0; current_byte < count; ++current_byte) {
        crc ^= (data[current_byte]);
        for (crc_bit = 8; crc_bit > 0; --crc_bit) {
            if (crc & 0x80)
                crc = (crc << 1) ^ CRC8_POLYNOMIAL;
            else
                crc = (crc << 1);
        }
    }
    return crc;
}

int8_t check_crc(const uint8_t* data, uint16_t count,
                               uint8_t checksum) {
    if (crc_gen(data, count) != checksum)
        return -2;
    return 0;
}
uint16_t add_command_buffer(uint8_t* buffer, uint16_t offset,
    uint16_t command) {
    buffer[offset++] = (uint8_t)((command & 0xFF00) >> 8);
    buffer[offset++] = (uint8_t)((command & 0x00FF) >> 0);
return offset;
}
uint16_t add_uint16_t_buffer(uint8_t* buffer, uint16_t offset,
    uint16_t data) {
buffer[offset++] = (uint8_t)((data & 0xFF00) >> 8);
buffer[offset++] = (uint8_t)((data & 0x00FF) >> 0);
buffer[offset] = crc_gen(&buffer[offset - SENSIRION_WORD_SIZE], SENSIRION_WORD_SIZE);
offset++;

return offset;
}