#include "tsl.h"
#include <stdio.h>
#include "hardware/i2c.h"
#include "pico/stdlib.h"

void tsl2591_write_register(i2c_inst_t *i2c, uint8_t reg, uint8_t value) {
    uint8_t data[2] = { TSL2591_COMMAND_BIT | reg, value };
    i2c_write_blocking(i2c, TSL2591_ADDR, data, 2, false);
}

uint16_t tsl2591_read_register_16(i2c_inst_t *i2c, uint8_t reg) {
    uint8_t data[2] = {0, 0};
    uint8_t cmd = TSL2591_COMMAND_BIT | reg;

    if (i2c_write_blocking(i2c, TSL2591_ADDR, &cmd, 1, true) < 0) {
        printf("I2C write failed!\n");
        return 0xFFFF; 
    }
    
    if (i2c_read_blocking(i2c, TSL2591_ADDR, data, 2, false) < 0) {
        printf("I2C read failed!\n");
        return 0xFFFF; 
    }

    return (data[1] << 8) | data[0];
}


void tsl2591_init(i2c_inst_t *i2c) {
    tsl2591_write_register(i2c, TSL2591_ENABLE, TSL2591_ENABLE_POWERON | TSL2591_ENABLE_AEN);
    tsl2591_write_register(i2c,TSL2591_CONFIG, TSL2591_MEDIUM_GAIN | int_time600);
}

void tsl2591_read_data(i2c_inst_t *i2c, uint16_t *full, uint16_t *ir, uint16_t *visible) {
    *full = tsl2591_read_register_16(i2c, TSL2591_C0DATAL);
    *ir = tsl2591_read_register_16(i2c, TSL2591_C1DATAL);
    *visible = (*full > *ir) ? (*full - *ir) : 0;
}
