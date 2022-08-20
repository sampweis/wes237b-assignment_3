#include "main.h"

// ----------------------------------------------
// Run a FIR filter on the given input data
// ----------------------------------------------
void fir(float *coeffs, float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
    int out_idx = 0;
    for(int i = 0; i < length-filterLength; i++)
    {
        float sum = 0;
        for(int h = 0; h < filterLength; h++)
        {
            sum += input[i+h]*coeffs[h];
        }
        output[i] = sum;
    }
}

// ----------------------------------------------
// Run a FIR filter on the given input data using Loop Unrolling
// ----------------------------------------------
void fir_opt(float *coeffs, float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
    int out_idx = 0;
    for(int i = 0; i < length-filterLength; i++)
    {
        float sum = 0;
        for(int h = 0; h < filterLength; h+=4)
        {
            sum += input[i+h]*coeffs[h];
            sum += input[i+(h+1)]*coeffs[h+1];
            sum += input[i+(h+2)]*coeffs[h+2];
            sum += input[i+(h+3)]*coeffs[h+3];
        }
        output[i] = sum;
    }
}

// ----------------------------------------------
// Run a FIR filter on the given input data using NEON
// ----------------------------------------------
void fir_neon(float *coeffs, float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
    int out_idx = 0;
    for(int i = 0; i < length-filterLength; i++)
    {
        float32x4_t sum = vmovq_n_f32(0);
        for(int h = 0; h < filterLength; h+=4)
        {
            float32x4_t packed_input = vld1q_f32(input+i+h);
            float32x4_t packed_coeffs = vld1q_f32(coeffs+h);
            sum = vmlaq_f32(sum, packed_input, packed_coeffs);
        }
        output[i] = sum[0] + sum[1] + sum[2] + sum[3];
    }
}


// ----------------------------------------------
// Create filter coefficients
// ----------------------------------------------
void designLPF(float* coeffs, int filterLength, float Fs, float Fx)
// ----------------------------------------------
{
	float lambda = M_PI * Fx / (Fs/2);

	for(int n = 0; n < filterLength; n++)
	{
		float mm = n - (filterLength - 1.0) / 2.0;
		if( mm == 0.0 ) coeffs[n] = lambda / M_PI;
		else coeffs[n] = sin( mm * lambda ) / (mm * M_PI);
	}
}
