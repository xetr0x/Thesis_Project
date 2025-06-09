#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "sth31.h"
#include "SGP40.h"
#include "tsl.h"
#include "scd30.h"
#include "sensirion_gas_index_algorithm.h"



#define I2C_PORTgpth i2c0
#define I2C_PORTtslscd i2c1

#define gpth_SDA 4
#define gpth_SCL 5

#define tslscd_SDA 6
#define tslscd_SCL 7


void initi2c0(){
    i2c_init(I2C_PORTgpth, 100 * 1000);
    gpio_set_function(gpth_SDA, GPIO_FUNC_I2C);
    gpio_set_function(gpth_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(gpth_SDA);
    gpio_pull_up(gpth_SCL);
}

void initi2c1(){
    i2c_init(I2C_PORTtslscd, 100 * 1000);
    gpio_set_function(tslscd_SCL, GPIO_FUNC_I2C);
    gpio_set_function(tslscd_SDA, GPIO_FUNC_I2C);
    gpio_pull_up(tslscd_SDA);
    gpio_pull_up(tslscd_SCL);
}



int main() {
    stdio_usb_init();
    printf("USB initialized\n");

    initi2c0();
    initi2c1();

    int32_t voc_index;
    float temp, hum, c02;
    uint16_t temp_scaled, hum_scaled;
    int16_t raw_voc;
    uint16_t full, ir, visible;

    GasIndexAlgorithmParams algo_params;
    GasIndexAlgorithm_init(&algo_params, GasIndexAlgorithm_ALGORITHM_TYPE_VOC);
    
    tsl2591_init(I2C_PORTtslscd);
    
    int i = 0;
    while (true) {
        if(i <100){
        
                readtempandhum(&temp, &hum);

                temp_scaled = (uint16_t)temp;
                hum_scaled = (uint16_t)hum;
                tsl2591_read_data(I2C_PORTtslscd,&full, &ir, &visible);

                sgp40_measure_raw_signal(hum_scaled,temp_scaled,&raw_voc);    
                GasIndexAlgorithm_process(&algo_params, raw_voc, &voc_index);

                bool data_ready = false;
                scd30_get_data_ready(&data_ready);
                
                scd30_read_co2(&c02);
                i++;
                printf("%d\n", i);
        sleep_ms(1000);
        }
        if (i>=100)
        {
            readtempandhum(&temp, &hum);

                temp_scaled = (uint16_t)temp;
                hum_scaled = (uint16_t)hum;
                tsl2591_read_data(I2C_PORTtslscd,&full, &ir, &visible);

                sgp40_measure_raw_signal(hum_scaled,temp_scaled,&raw_voc);    
                GasIndexAlgorithm_process(&algo_params, raw_voc, &voc_index);

                bool data_ready = false;
                scd30_get_data_ready(&data_ready);
                
                scd30_read_co2(&c02);
                
                printf("%.2f,%.2f,%d,%u,%u,%.1f\n", temp, hum, voc_index, ir, visible, c02);
                sleep_ms(10000);
        }
        
    }
    return 0;
}