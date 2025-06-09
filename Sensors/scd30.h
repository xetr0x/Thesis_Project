#ifndef scd30_h
#define scd30_h

#include "hardware/i2c.h"
#include "pico/stdlib.h"
#include <stdint.h>
#include <stdbool.h>

#define SCD30_ADDR 0x61

#define SCD30_CMD_START_PERIODIC_MEASUREMENT 0x0010
#define SCD30_CMD_STOP_PERIODIC_MEASUREMENT 0x0104
#define SCD30_CMD_READ_MEASUREMENT 0x0300
#define SCD30_CMD_SET_MEASUREMENT_INTERVAL 0x4600
#define SCD30_CMD_GET_DATA_READY 0x0202
#define SCD30_CMD_SET_TEMPERATURE_OFFSET 0x5403
#define SCD30_CMD_SET_ALTITUDE 0x5102
#define SCD30_CMD_SET_FORCED_RECALIBRATION 0x5204
#define SCD30_CMD_AUTO_SELF_CALIBRATION 0x5306
#define SCD30_CMD_READ_SERIAL 0xD033
#define SCD30_CMD_GET_FIRMWARE_VERSION 0xD100
#define SCD30_CMD_SOFT_RESET 0xD304

#define SCD30_SERIAL_NUM_WORDS 16
#define SCD30_WRITE_DELAY_US 20000

int16_t scd30_read_co2(float* co2);
int16_t scd30_start_periodic_measurement(void);
int16_t scd30_get_data_ready(bool* data_ready);

#endif