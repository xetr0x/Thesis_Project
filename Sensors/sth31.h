#ifndef STH31_H
#define STH31_H
#include "hardware/i2c.h"

#define SHT31_ADDR 0x44
bool readtempandhum(float *temp, float *hum);
#endif