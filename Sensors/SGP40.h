#ifndef SGP40_H
#define SGP40_H

#include "hardware/i2c.h"
#include "pico/stdlib.h"
#include <stdint.h>
#include <stdbool.h>

#define SGP40_ADDR 0x59

// Commands
#define SGP40_MEASUREMENT_RAW 0x260F
#define SGP40_MEASUREMENT_TEST 0x280E
#define SGP40_HEATER_OFF 0x3615
#define SGP40_RESET 0x0006
#define SGP40_GET_SERIAL_ID 0x3682
#define SGP40_GET_FEATURESET 0x202F





int16_t sgp40_measure_raw_signal(uint16_t humidity, uint16_t temperature, uint16_t* sraw_voc);
#endif