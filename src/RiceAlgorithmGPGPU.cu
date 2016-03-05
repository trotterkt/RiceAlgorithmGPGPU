/*
 ============================================================================
 Name        : RiceAlgorithmGPGPU.cu
 Author      : Keir Trotter
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <iostream>
#include <FileBasedImagePersistence.h>
#include <Sensor.h>
#include <CudaHelper.h>
#include <RiceAlgorithmKernels.h>

using namespace std;
using namespace RiceAlgorithm;



/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	return rc;
}

float *cpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	for (unsigned cnt = 0; cnt < size; ++cnt) rc[cnt] = 1.0/data[cnt];
	return rc;
}


void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5*(i+1);
}

int main(void)
{
	//=====================================================

    cout.precision(4);

    cout << "Compressing Landsat_agriculture-u16be-6x1024x1024..." << endl;

    FileBasedImagePersistence image("Landsat_agriculture-u16be-6x1024x1024", Rows, Columns, Bands);


    // This data has not been pre-processed. Need to decide if the pre-processor will be
    // placed on the GPGPU -- if so, needs to be un-associated from the Sensor type

//    ushort* hostImagePtr = image.getSampleData(1);
//
//
//    ushort *gpuRawImageData;
//
//	const int NumberOfSamples(Rows * Columns * Bands);
//
//
//	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuRawImageData, sizeof(ushort)*NumberOfSamples));
//	CUDA_CHECK_RETURN(cudaMemcpy(gpuRawImageData, hostImagePtr, sizeof(ushort)*NumberOfSamples, cudaMemcpyHostToDevice));



    // Construct my LandSat sensor, which performs the compression of the supplied
    // raw image data per the Rice algorithm
	Sensor landsat(&image, Rows, Columns, Bands);

    // Initiate the Rice algorithm compression
	landsat.process();

    //=====================================================

//	static const int WORK_SIZE = 65530;
//	float *data = new float[WORK_SIZE];
//
//	initialize (data, WORK_SIZE);
//
//	float *recCpu = cpuReciprocal(data, WORK_SIZE);
//	float *recGpu = gpuReciprocal(data, WORK_SIZE);
//	float cpuSum = std::accumulate (recCpu, recCpu+WORK_SIZE, 0.0);
//	float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);
//
//	/* Verify the results */
//	std::cout<<"gpuSum = "<<gpuSum<< " cpuSum = " <<cpuSum<<std::endl;
//
//	/* Free memory */
//	delete[] data;
//	delete[] recCpu;
//	delete[] recGpu;

	cudaDeviceReset();

	return 0;
}


