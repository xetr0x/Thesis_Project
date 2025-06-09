#include "sth31.h"
#include "hardware/i2c.h"
#include "pico/stdlib.h"
#include <stdio.h>

bool send_command(uint16_t command){
    uint8_t msb[2] = {command >> 8, command & 0xFF};
    return (i2c_write_blocking(i2c0, SHT31_ADDR, msb, 2, false) == 2);
}
bool readtempandhum(float *temp, float *hum){
    uint8_t raw[6];
    if (!send_command(0x2C06))
    {
        return false;
    }
    sleep_ms(20);

    if (i2c_read_blocking(i2c0, SHT31_ADDR, raw, 6, false) != 6)
    {
        return false;
    }
    else{
    uint16_t raw_temp = (raw[0] << 8) | raw[1];
    *temp = -45 + (175 * (((float) raw_temp)/65535.0));

    uint16_t raw_hum = (raw[3] << 8) | raw[4];
    *hum = 100 * ((((float) raw_hum)/65535.0));
    return true;
    }
}