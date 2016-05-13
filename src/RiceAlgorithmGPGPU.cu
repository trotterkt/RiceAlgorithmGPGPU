 /*
 * RiceAlgorithmGPGPU.cu
 *
 *  Created by: Keir Trotter
 *  California State University, Fullerton
 *  MSE, CPSC 597, Graduate Project
 *
 *  Copyright 2016 Keir Trotter
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <iostream>
#include <FileBasedImagePersistence.h>
#include <Sensor.h>
#include <CudaHelper.h>
#include <RiceAlgorithmKernels.h>
#include <Timing.h>

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

int main(int argc, char *argv[])
{
    if (argc < 2)
    {   // Expect 2 arguments: the program name,
	    // and the raw image file for compression
		std::cerr << "\nUsage: " << argv[0] << "\nProvide basename for raw data to compress and subsequently decompress." << std::endl;
		return 1;
    }


    std::string baseImageFilename = argv[1];


	cudaDeviceReset();

	//=====================================================
    cout.precision(4);


    cout << "\n\n";
    cout << "*********************************************************************" << endl;
    cout << "*                                                                   *" << endl;
    cout << "*                         RICE ALGORITHIM                           *" << endl;
    cout << "*             General Purpose Graphics Processing Unit              *" << endl;
    cout << "*                            (GPGPU)                                *" << endl;
    cout << "*                         Parallelization                           *" << endl;
    cout << "*                                                                   *" << endl;
    cout << "*          CSU Fullerton, MSE Graduate Project, Fall 2016           *" << endl;
    cout << "*                          Keir Trotter                             *" << endl;
    cout << "*                                                                   *" << endl;
    cout << "*                                                                   *" << endl;
    cout << "*********************************************************************\n\n\n" << endl;

    // Some device memory is allocated outside of both the
    // sensor and ground system, since once created on the
    // GPGPU, it is reused, versus copying back to the host

    // Note convention of specifying host memory prefixed by 'h_' and device by 'd_'


    // Place the pre-processed data on the GPU
    //:TODO: This probably better belongs in the Predictor constructor and
    // freed in the destructor - in other words what would be expected in
    // C++ programs
    //***************************************************************************
    ushort *d_PreProcessedImageData;

    long totalSamples = Rows*Columns*Bands;


   	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_PreProcessedImageData, sizeof(ushort)*totalSamples));


    unsigned char* d_EncodedBlocks(0);
   	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_EncodedBlocks, MaximumEncodedMemory));
   	CUDA_CHECK_RETURN(cudaMemset(d_EncodedBlocks, 0, MaximumEncodedMemory));

   	// Allocate space for encoded block sizes -- the number of elements is the total samples
   	// divided by the sample block size
    ushort* d_EncodedBlockSizes(0);
   	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_EncodedBlockSizes, sizeof(ushort)*((Rows*Columns*Bands)/32)));
   	CUDA_CHECK_RETURN(cudaMemset(d_EncodedBlockSizes, 0, sizeof(ushort)*((Rows*Columns*Bands)/32)));

    //******************************************************************************
    // Rice Compression Processing
    //******************************************************************************
    FileBasedImagePersistence image(baseImageFilename.c_str(), Rows, Columns, Bands);


    // Construct my LandSat sensor, which performs the compression of the supplied
    // raw image data per the Rice algorithm
	Sensor landsat(&image, Rows, Columns, Bands);

    // this is unique to the GPGPU
    // need to write the header out before
    // the EncodedData is overridden
	landsat.sendHeader();
    image.writeEncodedData(true);

    timestamp_t t0 = getTimestamp();


    // Initiate the Rice algorithm compression
    double compressionProcessingTime = landsat.process(d_EncodedBlocks, d_EncodedBlockSizes, d_PreProcessedImageData);

    timestamp_t t1 = getTimestamp(); // investigate if time measure is in wrong place
    //******************************************************************************


    // Write out the encoded data. This is outside of the compression processing
    image.writeEncodedData();


    //******************************************************************************
    // Rice Decompression Processing
    //******************************************************************************
    cout << "\n\nDecompressing " << baseImageFilename.c_str() << "...\n" << endl;

    timestamp_t t2 = getTimestamp();

    // Kick off the associated decompression
    // The debug version launch is so that I can compare the
    // pre-processed residuals to that extracted from
    // decompression. These must match in able to have an
    // accurate decoding. To accurately perform this check
    // need to clear the existing pre-processed data created
    // by the sensor first. The space will be over written by
    // the ground process.
   	CUDA_CHECK_RETURN(cudaMemset(d_PreProcessedImageData, 0, sizeof(ushort)*totalSamples));

	#ifdef DEBUG
    	ushort* sensorResidualCompare(0);
    	sensorResidualCompare = landsat.getResiduals();
        double decompressionProcessingTime = landsat.getGround()->process(d_PreProcessedImageData, d_EncodedBlocks, d_EncodedBlockSizes, sensorResidualCompare);
	#else
        double decompressionProcessingTime = landsat.getGround()->process(d_PreProcessedImageData, d_EncodedBlocks, d_EncodedBlockSizes);
	#endif

    timestamp_t t3 = getTimestamp();
    //******************************************************************************



    cout << "=============================================================" << endl;
    cout << "Total Rice Compression processing time   ==> " << fixed << compressionProcessingTime << " seconds"<< endl;
    cout << "Total Rice Decompression processing time ==> " << fixed << decompressionProcessingTime << " seconds"<< endl;
    cout << "Total Round Trip ==> " << fixed << (compressionProcessingTime + decompressionProcessingTime)  << " seconds"<< endl;
    cout << "=============================================================" << endl;

    // Write out the decoded data. This is outside of the decompression processing
    image.writeDecodedData();




    cudaFree(d_EncodedBlocks);
    cudaFree(d_EncodedBlockSizes);

	cudaDeviceReset();

	return 0;
}


