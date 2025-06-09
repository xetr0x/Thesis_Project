#include "sgp40.h"
#include "sensirion.h"

int16_t sgp40_measure_raw_signal(uint16_t humidity, uint16_t temperature, uint16_t* sraw_voc) {
    int16_t error;
    uint8_t buffer[8];
    uint16_t offset = 0;
    offset = add_command_buffer(&buffer[0], offset, 0x260F);

    offset = add_uint16_t_buffer(&buffer[0], offset,humidity);
    offset = add_uint16_t_buffer(&buffer[0], offset, temperature);

    error =  i2c_write(i2c0,SGP40_ADDR, &buffer[0], offset);
    if (error) {
        return error;
    }

    sleep_us(30000);

    error = i2c_read(i2c0,SGP40_ADDR, &buffer[0], 2);
    if (error) {
        return error;
    }
    *sraw_voc = bytetou16(&buffer[0]);
    return 0;
}