#include "scd30.h"
#include "sensirion.h"

int16_t scd30_send_command(uint16_t command) {
    uint8_t buf[2];
    buf[0] = (command >> 8);
    buf[1] = (command & 0xFF);
    return i2c_write(i2c1,SCD30_ADDR,buf, 2);
}

int16_t scd30_send_command_with_args(uint16_t command, uint16_t arguments) {
    uint8_t buf[5];
    buf[0] = (command >> 8);
    buf[1] = (command & 0xFF);
    buf[2] = (arguments >> 8);
    buf[3] = (arguments & 0xFF);
    buf[4] = crc_gen(&buf[2], 2);
    return i2c_write(i2c1,SCD30_ADDR,buf, 5);
}

int16_t scd30_start_periodic_measurement(void) {
    return scd30_send_command_with_args(0x0010, 0); 
}

int16_t scd30_get_data_ready(bool* data_ready) {
    uint8_t buf[3];
    int16_t error;
    
    error = scd30_send_command(0x0202);
    if (error) return error;
    
    sleep_ms(3);
    
    error = i2c_read(i2c1,SCD30_ADDR ,buf, 3);
    if (error) return error;
    
    // Check CRC
    if (crc_gen(buf, 2) != buf[2]) return -1;
    
    *data_ready = (buf[0] << 8 | buf[1]) > 0;
    return 0;
}

int16_t scd30_read_co2(float* co2) {
    uint8_t buf[6]; // We only need first 2 words + CRC bytes for CO2
    int16_t error;
    
    // Send command to read measurement
    error = scd30_send_command(SCD30_CMD_READ_MEASUREMENT);
    if (error) return error;
    
    // SCD30 needs time to prepare data
    sleep_ms(3);
    
    // Only read the first 6 bytes (CO2 value + CRC)
    error = i2c_read(i2c1, SCD30_ADDR, buf, 6);
    if (error) return error;
    
    // Verify CRC for both words
    if (crc_gen(&buf[0], 2) != buf[2] || crc_gen(&buf[3], 2) != buf[5]) {
        return -1; // CRC error
    }
    
    // Convert bytes to float (CO2 value)
    union {
        uint32_t u32;
        float f;
    } tmp;
    
    tmp.u32 = ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) | 
              ((uint32_t)buf[3] << 8)  | ((uint32_t)buf[4]);
    *co2 = tmp.f;
    
    return 0;
    
}