#include <stdio.h>

#include "arm_neon.h"

int main () {
    /* Create custom arbitrary data. */
    const uint8_t uint8_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    /* Load our custom data into the vector register. Use function: vld1q_u8 */
    uint8x16_t data = vld1q_u8(uint8_data);
    
    /* print your data */
    printf ("%s = ", "data");
    for (int i = 0; i < 16; i++) {
        printf("%d ", data[i]);
        
    }
    printf ("\n");
    
    
    /* Create an uint8x16 vector and set all elements to 3. use function: vmovq_n_u8 */
    uint8x16_t add_val = vmovq_n_u8(250);
    
    /* add data vector and vector with all elements=3. use function: vaddq_u8 */ 
    uint8x16_t res = vaddq_u8(data, add_val);
    
    /* print the results */
    printf ("%s = ", "data_new");
    for (int i = 0; i < 16; i++) {
         printf("%d ", res[i]);
    }
    printf ("\n");
    
    return 0;
}
