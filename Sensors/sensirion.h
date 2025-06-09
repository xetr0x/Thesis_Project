#include "hardware/i2c.h"
#include "pico/stdlib.h"
#include <stdint.h>
#include <stdbool.h>

#define CRC_ERROR 1
#define I2C_BUS_ERROR 2
#define I2C_NACK_ERROR 3
#define BYTE_NUM_ERROR 4

#define CRC8_POLYNOMIAL 0x31
#define CRC8_INIT 0xFF
#define CRC8_LEN 1

#define SENSIRION_COMMAND_SIZE 2
#define SENSIRION_WORD_SIZE 2
#define SENSIRION_NUM_WORDS(x) (sizeof(x) / SENSIRION_WORD_SIZE)
#define SENSIRION_MAX_BUFFER_WORDS 32

uint16_t bytetou16(uint8_t *bytes);

uint32_t bytetou32(uint8_t *bytes);

int32_t byteto32(uint8_t *bytes);

float bytetofloat(uint8_t *bytes);

void u32tobyte(uint32_t val, uint8_t *bytes);

void u16tobyte(uint16_t val, uint8_t *bytes);

void n32tobyte(int32_t val, uint8_t *bytes);

void n16tobyte(int16_t val, uint8_t *bytes);

void floattobyte(float val, uint8_t *bytes);

void cpbyte(uint8_t *og, uint8_t *new, uint16_t len);

int8_t i2c_read(i2c_inst_t *i2c,uint8_t address, uint8_t *data, uint16_t count);

int8_t i2c_write(i2c_inst_t *i2c,uint8_t address, uint8_t *data, uint16_t count);

uint8_t crc_gen(const uint8_t* data, uint16_t count);

int8_t check_crc(const uint8_t* data, uint16_t count, uint8_t checksum);

uint16_t add_command_buffer(uint8_t* buffer, uint16_t offset, uint16_t command);

uint16_t add_uint16_t_buffer(uint8_t* buffer, uint16_t offset, uint16_t data);