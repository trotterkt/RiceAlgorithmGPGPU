/*
 * RiceAlgorithmKernels.h
 *
 *  Created on: Feb 23, 2016
 *      Author: ktrotter
 */

#ifndef SENSORKERNELS_H_
#define SENSORKERNELS_H_

#include <stdio.h>
//#include <thrust/device_vector.h>


/**
 * CUDA kernel that identifies the winning encoding scheme for each block
 */
__global__ void encodingKernel(ushort data[32])
{
	// Operate on all samples for a given block together
	unsigned int sampleIndex = threadIdx.x;
	unsigned int dataIndex = threadIdx.x + blockIdx.x*blockDim.x;


	//printf("Block:%d data[%d]=%d\n", blockIdx.x, sampleIndex, data[dataIndex]);

	//thrust::host_vector<int> testVector;

}



#endif /* SENSORKERNELS_H_ */
