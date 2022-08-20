
#include "sobel.h"
#include "arm_neon.h"

#include <iostream>

using namespace std;
using namespace cv;


int sobel_kernel_x[3][3] = {
	{ 1,  0, -1},
	{ 2,  0, -2},
	{ 1,  0, -1}};

int sobel_kernel_y[3][3] = {
	{ 1,    2,  1},
	{ 0,    0,  0},
	{ -1,  -2, -1}};

void sobel(const Mat& src, Mat& dst)
{
    const int HEIGHT = src.rows;
	const int WIDTH  = src.cols;
    
    const unsigned char* src_ptr  = src.ptr<const unsigned char>();
    unsigned char* dst_ptr  = dst.ptr<unsigned char>();
    
    for(int row = 1; row < HEIGHT-1; row++)
    {
        for(int col = 1; col < WIDTH-1; col++)
        {
            int accum_x = 0;
            int accum_y = 0;
            
            // calculate convolution with x and y kernels
            for(int kernel_row = 0; kernel_row < 3; kernel_row++)
            {
                for(int kernel_col = 0; kernel_col < 3; kernel_col++)
                {
                    accum_x += (sobel_kernel_x[kernel_row][kernel_col]*
                                src_ptr[(row+kernel_row-1) * WIDTH + (col+kernel_col-1)]);
                    accum_y += (sobel_kernel_y[kernel_row][kernel_col]*
                                src_ptr[(row+kernel_row-1) * WIDTH + (col+kernel_col-1)]);
                }
            }
            
            // calculate total magnitude and clamp to 255
            unsigned int total = sqrt(accum_x*accum_x + accum_y*accum_y);
            dst_ptr[row * WIDTH + col] = total > 255 ? 255 : total;
        }
    }
    
}


void sobel_unroll(const Mat& src, Mat& dst)
{
	const int HEIGHT = src.rows;
	const int WIDTH  = src.cols;
    
    const unsigned char* src_ptr  = src.ptr<const unsigned char>();
    unsigned char* dst_ptr  = dst.ptr<unsigned char>();
    
    for(int row = 1; row < HEIGHT-1; row++)
    {
        for(int col = 1; col < WIDTH-1; col++)
        {
            int accum_x = 0;
            int accum_y = 0;
            
            // calculate convolution with x kernel
            accum_x += (sobel_kernel_x[0][0]*
                        src_ptr[(row+0-1) * WIDTH + (col+0-1)]);
            accum_x += (sobel_kernel_x[0][1]*
                        src_ptr[(row+0-1) * WIDTH + (col+1-1)]);
            accum_x += (sobel_kernel_x[0][2]*
                        src_ptr[(row+0-1) * WIDTH + (col+2-1)]);
            accum_x += (sobel_kernel_x[1][0]*
                        src_ptr[(row+1-1) * WIDTH + (col+0-1)]);
            accum_x += (sobel_kernel_x[1][1]*
                        src_ptr[(row+1-1) * WIDTH + (col+1-1)]);
            accum_x += (sobel_kernel_x[1][2]*
                        src_ptr[(row+1-1) * WIDTH + (col+2-1)]);
            accum_x += (sobel_kernel_x[2][0]*
                        src_ptr[(row+2-1) * WIDTH + (col+0-1)]);
            accum_x += (sobel_kernel_x[2][1]*
                        src_ptr[(row+2-1) * WIDTH + (col+1-1)]);
            accum_x += (sobel_kernel_x[2][2]*
                        src_ptr[(row+2-1) * WIDTH + (col+2-1)]);
            
            // calculate convolution with y kernel
            accum_y += (sobel_kernel_y[0][0]*
                        src_ptr[(row+0-1) * WIDTH + (col+0-1)]);
            accum_y += (sobel_kernel_y[0][1]*
                        src_ptr[(row+0-1) * WIDTH + (col+1-1)]);
            accum_y += (sobel_kernel_y[0][2]*
                        src_ptr[(row+0-1) * WIDTH + (col+2-1)]);
            accum_y += (sobel_kernel_y[1][0]*
                        src_ptr[(row+1-1) * WIDTH + (col+0-1)]);
            accum_y += (sobel_kernel_y[1][1]*
                        src_ptr[(row+1-1) * WIDTH + (col+1-1)]);
            accum_y += (sobel_kernel_y[1][2]*
                        src_ptr[(row+1-1) * WIDTH + (col+2-1)]);
            accum_y += (sobel_kernel_y[2][0]*
                        src_ptr[(row+2-1) * WIDTH + (col+0-1)]);
            accum_y += (sobel_kernel_y[2][1]*
                        src_ptr[(row+2-1) * WIDTH + (col+1-1)]);
            accum_y += (sobel_kernel_y[2][2]*
                        src_ptr[(row+2-1) * WIDTH + (col+2-1)]);
            
            // calculate total magnitude and clamp to 255
            unsigned int total = sqrt(accum_x*accum_x + accum_y*accum_y);
            dst_ptr[row * WIDTH + col] = total > 255 ? 255 : total;
        }
    }
}

void sobel_neon(const Mat& src, Mat& dst)
{
	const int HEIGHT = src.rows;
	const int WIDTH  = src.cols;
    
    const unsigned char* src_ptr  = src.ptr<const unsigned char>();
    unsigned char* dst_ptr  = dst.ptr<unsigned char>();
    
    const short FILLER = 0;
    short packed_kernels [24] =
    {
        // x row 1
        1,  0, -1, FILLER,
        // x row 2
        2,  0, -2, FILLER,
        // y row 1
        1,  2,  1, FILLER,
        // y row 2
        0,  0,  0, FILLER,
        // x row 3
        1,  0, -1, FILLER,
        // y row 3
        -1,-2, -1, FILLER
    };
    
    // x kernel row 1 and 2
    int16x8_t k_x12 = vld1q_s16(packed_kernels);
    // y kernel row 1 and 2
    int16x8_t k_y12 = vld1q_s16(packed_kernels+8);
    // x kernel row 3 and y kernel row 3
    int16x8_t k_x3y3 = vld1q_s16(packed_kernels+16);
    
    for(int row = 1; row < HEIGHT-1; row++)
    {
        for(int col = 1; col < WIDTH-1; col++)
        {
            // convert input rows to int16x8_t
            uint8x8_t row1_char = vld1_u8(src_ptr + ((row-1) * WIDTH + (col-1)));
            uint8x8_t row2_char = vld1_u8(src_ptr + ((row) * WIDTH + (col-1)));
            uint8x8_t row3_char = vld1_u8(src_ptr + ((row+1) * WIDTH + (col-1)));

            int16x8_t row1 = vreinterpretq_s16_u16(vmovl_u8(row1_char));
            int16x8_t row2 = vreinterpretq_s16_u16(vmovl_u8(row2_char));
            int16x8_t row3 = vreinterpretq_s16_u16(vmovl_u8(row3_char));

            // combine rows 1 and 2 into one int16x8_t
            int16x8_t r_12 = vcombine_s16(vget_low_s16(row1), vget_low_s16(row2));
            // put two copies on row 3 in one int16x8_t to calculate x and y kernel in one operation
            int16x8_t r_33 = vcombine_s16(vget_low_s16(row3), vget_low_s16(row3));

            // calculate x kernel accumulation for rows 1 and 2
            int16x8_t accum_x12 = vmulq_s16(k_x12, r_12);
            // calculate y kernel accumulation for rows 1 and 2
            int16x8_t accum_y12 = vmulq_s16(k_y12, r_12);
            // calculate x and y kernels accumulation for row 3
            int16x8_t accum_x3y3 = vmulq_s16(k_x3y3, r_33);

            // combine accumulations for rows 1, 2, and 3
            int accum_x = vaddvq_s16(accum_x12) + vaddv_s16(vget_low_s16(accum_x3y3));
            int accum_y = vaddvq_s16(accum_y12) + vaddv_s16(vget_high_s16(accum_x3y3));
            
            // calculate total magnitude and clamp to 255
            unsigned int total = sqrt(accum_x*accum_x + accum_y*accum_y);
            dst_ptr[row * WIDTH + col] = total > 255 ? 255 : total;
        }
    }
}

