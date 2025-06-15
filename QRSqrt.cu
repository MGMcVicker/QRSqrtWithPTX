/*
===========================================================================
Copyright (C)  2025 Matthew McVicker

QRSqrt calculates the Quake III Fast Inverse Square Root Algorithm with inline PTX instructions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

A Copyright (c) notice and General Public License permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

===========================================================================
*/


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>

#include <random>

// CUDA Kernel to calculate the fast inverse square root (from Quake III)
__global__ void QRSqrtKernel(const float* input, volatile float* output) {
    //int idx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int z; // Initalize an unsighed int to use as a placeholder. Needed to balance the requested and provided asm operands

    // Inline Assembly 
    asm volatile (
        ".reg .u32 r1;\n" // Initialize registers
        ".reg .u32 r2;\n"
        ".reg .u32 r3;\n"
        ".reg .u32 r4;\n"
        "ld.global.nc.u32 r1, [%1];\n" // Load a 32 bit float from device memory into register %1 with a type of an unsigned 32 bit int
        "shr.u32 r2, r1, 1;\n" // Shift the value in r1 right by 1 bit (effectively dividing by 2) and store in r2. 
                                // The square root is a number to the half power. The error in the mantissa is handled in the next steps
        "mov.u32 r3, 0x5f3759df;" // Move the magic number to register r3, 
                                // the number is a pre-calculated constant based on the bit representation of 2^(1/2) according to the IEEE 754 floating-point format
        "sub.u32 r4, r3, r2;\n" // subtract the magic number stored in r3 from r2, and store in r4
        "st.global.u32 [%2], r4;" // store the value from r4 into memory
        : "=r"(z)  // This line is need to balance the requested and provided asm operands. The value from the destination .u32 register is written back to the C++ variable z
        : "l"(&input[idx]), // %1 (the memory address) is a long pointer to the input array element corresponding to the current thread
        "l"(output + idx)  // %2 is a long pointer to the output array element corresponding to the current thread
    );

    float x2, y;
    const float threehalfs = 1.5F;
    x2 = input[idx] * 0.5F;
    y = output[idx];
    y = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration, new y = y - f(y)/f'(y), here f(y) = y^2 - x 
    output[idx] = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration

}

int main() {

    const unsigned int arraySize = 512;
    cudaError_t cudaStatus;

    // Host memory allocation
    float* inputData = new float[arraySize];
    float* outputData = new float[arraySize];

    // Generate a random distribution between 0 and 1000 for the inputs
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1000.0);

    // Initialize inputs on the host 
    for (int i = 0; i < arraySize; ++i) {
        inputData[i] = dis(gen); // Generate a random number from the distribution
    }

    // Initialize poiniters to device memory
    float* dev_inputData = nullptr;
    float* dev_outputData = nullptr;

    // Define block dimensions: 2 blocks with 256 threads per block. TODO: test blocks of 128 and 512 threads
    unsigned int blockSize = 256;
    unsigned int numBlocks = 2;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0); 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)
    cudaStatus = cudaMalloc((void**)&dev_inputData, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_outputData, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_inputData, inputData, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    /* Launch kernal */
    QRSqrtKernel<<<numBlocks, blockSize>>>((dev_inputData), (dev_outputData));
    /*****************/

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rsqrtKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Wait for the kernel to finish and return any errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching QRSqrtKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outputData, dev_outputData, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Print the results
    std::cout << "Input, Output (Fast Inverse Square Root)\n";
    for (int i = 0; i < arraySize; ++i) {
        std::cout << inputData[i] << ", ";
        std::cout << outputData[i] << "\n";
    }
    std::cout << std::endl;

    // Free GPU/device memory
    cudaFree(dev_inputData);
    cudaFree(dev_outputData);

    // Free host memory
    delete[] inputData;
    delete[] outputData;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();

    // Return 0 if no errors
    if (cudaStatus == cudaSuccess) {
        return 0;
    }

// Report errors and return 1
Error:
    cudaFree(dev_inputData);
    cudaFree(dev_outputData);
    delete[] inputData;
    delete[] outputData;

    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
}




/*
// Exerpt from the Quake III Arena source code
// q_math.c -- stateless support routines that are included in each code module

...

/*
** float q_rsqrt( float number )
*
float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;						// evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

#ifndef Q3_VM
#ifdef __linux__
	assert( !isnan(y) ); // bk010122 - FPE?
#endif
#endif
	return y;
}
*/
