/*
 * GroundSystem.cpp
 *
 *  Created by: Keir Trotter
 *  California State University, Fullerton
 *  MSE, CPSC 597, Graduate Project
 *
 *  Copyright 2016 Keir Trotter
 */

#include <GroundSystem.h>
#include <Sensor.h>
#include <Endian.h>
#include <iostream>
#include <stdio.h>
#include <DebuggingParameters.h>
#include "math.h" // CUDA Math library
#include <CudaHelper.h>
#include <Timing.h>


#ifdef DEBUG
#include <fstream>
#endif

using namespace std;
using namespace RiceAlgorithm;


GroundSystem::GroundSystem(ImagePersistence* image) : mySource(image), myRawSamples(0)
{
    //:TODO: There are some inconsistencies here around dimension extraction
    // that should be addressed
    memset(&myHeader, 0, sizeof(CompressedHeader));
    ushort x;
    ushort y;
    ushort z;

    image->getDimensions(x, y, z);
    myHeader.xDimension = x;
    myHeader.yDimension = y;
    myHeader.zDimension = z;
}

GroundSystem::~GroundSystem()
{
    if(myRawSamples)
    {
        delete [] myRawSamples;
        myRawSamples = 0;
    }
}

double GroundSystem::process(ushort *d_PreProcessedImageData, unsigned char* d_EncodedBlocks, ushort* d_EncodedBlockSizes, ushort* referenceResiduals)
{

	// This begins the decoding
	readHeader();

	// Having the raw sample dimensions from the header, allocate space for
	// the decoding
    ushort x(0);
    ushort y(0);
    ushort z(0);

    //:TODO: There are some inconsistencies here around dimension extraction
    // that should be addressed
	mySource->getDimensions(x, y, z);
	const long NumberOfSamples(x * y * z);

	myRawSamples = new ushort[NumberOfSamples];

	// Encoded data should begin right after the header (byte 19)

	// 1st grab the Encoded ID
	ulong totalEncodedLength(HeaderLength * BitsPerByte);

	unsigned int additionalBits(0);

	ushort* encodedBlockSizes = new ushort[NumberOfSamples / 32];
	ulong count(0);


   	// ***Block and Grid size determinations***
   	//=========================================================================
   	// Block XDim = 32, YDim = 32 (Y Dim will access 32 samples at a time),
   	// 1024 threads will access 32768 samples.
   	//
   	// Grid XDim = 6, YDim = 32 ( this is equivalent to 192 blocks )
   	// Total is then 6291456 samples processed
   	//
   	//
   	//========================================================================
   	dim3 threadsPerBlock(32, 32); // Threads per block will be limited by the
   	                              // CC of the GPGPU
   	                              // (i.e. lower CC only allow 256 threads per block)
   	dim3 gridBlocks(6, 32);

    timestamp_t t0 = getTimestamp();

   	decodingKernel<<<gridBlocks, threadsPerBlock>>> (d_PreProcessedImageData, d_EncodedBlocks, d_EncodedBlockSizes);

   	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    timestamp_t t1 = getTimestamp();


    ushort* h_PreProcessedImageData = new ushort[NumberOfSamples];
   	CUDA_CHECK_RETURN(cudaMemcpy(h_PreProcessedImageData, d_PreProcessedImageData, NumberOfSamples*sizeof(ushort), cudaMemcpyDeviceToHost));


	#ifdef DEBUG
	const long MaximumBlocks(NumberOfSamples / 32);

	for (long blockIndex = 0; blockIndex < MaximumBlocks; blockIndex++)
	{
		for(int index=0; index<32; index++)
		{
			if(referenceResiduals[blockIndex*32 + index] != h_PreProcessedImageData[blockIndex*32 + index])
			{
				cout << "Mismatch residual value at Block:" << blockIndex << " Index:" << index << endl;
			}
		}
	}
	#endif


	// Perform unprediction of the residual values
	RiceAlgorithm::Predictor unprocessor(x, y, z);

	ushort* samples = new ushort[NumberOfSamples];
	unprocessor.getSamples(h_PreProcessedImageData, samples);


	mySource->sendDecodedData(reinterpret_cast<char*>(samples), NumberOfSamples*sizeof(short));


    #ifdef DEBUG
    std::ofstream residualsStream;
    residualsStream.open("residualsGround.bin", ios::out | ios::in | ios::binary | ios::trunc);

    if (!residualsStream.is_open())
    {
        exit(EXIT_FAILURE);
    }
    //for(long index=0; index<NumberOfSamples; index++)
    for(long index=0; index<2000; index++)
    {
        cout << "residualsGround[" << index << "]=" << h_PreProcessedImageData[index] << endl;
    }
    residualsStream.write(reinterpret_cast<char*>(h_PreProcessedImageData), (1024*1024*6*2));
    residualsStream.close();
    #endif


   	delete [] h_PreProcessedImageData;


   	return(getSecondsDiff(t0, t1));
}


void GroundSystem::readHeader()
{
    // Note: Header is not completely populated for all defined parameters.
    // Only what is applicable to the selected test raw data to
    // identify identical information.

    // Since the image written to the encoded file has already
    // been materialized, just read this directly rather than
    // re-reading from the file. Just trying to perform consecutive
    // encoding, decoding here
    unsigned char* encodedData = mySource->getEncodedData();

    memcpy(&myHeader.xDimension, &encodedData[1], sizeof(myHeader.xDimension));
    memcpy(&myHeader.yDimension, &encodedData[3], sizeof(myHeader.yDimension));
    memcpy(&myHeader.zDimension, &encodedData[5], sizeof(myHeader.zDimension));
    bigEndianVersusLittleEndian(myHeader.xDimension);
    bigEndianVersusLittleEndian(myHeader.yDimension);
    bigEndianVersusLittleEndian(myHeader.zDimension);
}


