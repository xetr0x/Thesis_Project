#ifndef tsl
#define tsl

#include "hardware/i2c.h"

// I2C address of TSL2591
#define TSL2591_ADDR 0x29

// Registers
#define TSL2591_COMMAND_BIT 0xA0
#define TSL2591_ENABLE 0x00
#define TSL2591_CONFIG 0x01
#define TSL2591_C0DATAL 0x14
#define TSL2591_C1DATAL 0x16

#define TSL2591_MEDIUM_GAIN 0x10
#define TSL2591_HIGH_GAIN 0x20


#define int_time200 0x01
#define int_time300 0x02
#define int_time400 0x03
#define int_time500 0x04
#define int_time600 0x05

// Enable settings
#define TSL2591_ENABLE_POWERON 0x01
#define TSL2591_ENABLE_AEN 0x02

void tsl2591_init(i2c_inst_t *i2c);
void tsl2591_read_data(i2c_inst_t *i2c, uint16_t *full, uint16_t *ir, uint16_t *visible);

#endif
