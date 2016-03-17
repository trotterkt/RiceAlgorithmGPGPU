/*
 * Sensor.cpp
 *
 *  Created by: Keir Trotter
 *  California State University, Fullerton
 *  MSE, CPSC 597, Graduate Project
 *
 *  Copyright 2016 Keir Trotter
 */


#include <Sensor.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <limits.h>
#include <Timing.h>
#include <Endian.h>
#include <CudaHelper.h>
#include <SensorKernels.h>



using namespace std;
using namespace RiceAlgorithm;


Sensor::Sensor(ImagePersistence* image, unsigned int x, unsigned int y, unsigned int z) :
        mySource(image), mySamples(0), myXDimension(x), myYDimension(y), myZDimension(z), myPreprocessor(x, y, z), myWinningEncodedLength((unsigned int)-1)
{

    mySamples = mySource->getSampleData(1); //:TODO: only one scan, need to address multiple

    // Create the encoding types
    size_t bufferSize = myXDimension*myYDimension*myZDimension;

    //AdaptiveEntropyEncoder* noComp = new AdaptiveEntropyEncoder(bufferSize);
    //SecondExtensionOption* secondExt = new SecondExtensionOption(bufferSize);
    //ZeroBlockOption* zeroBlock = new ZeroBlockOption(bufferSize);
    SplitSequence* split = new SplitSequence(bufferSize);


    //myEncoderList.push_back(noComp);  // No compression must be the first item
    // myEncoderList.push_back(secondExt);
    // myEncoderList.push_back(zeroBlock);
    //myEncoderList.push_back(split);
}

Sensor::~Sensor()
{

}

void Sensor::process()
{
	// The goal hear is to form the telemetry of the data

	sendHeader();

	// :TODO: formalize this a little more
    myEncodedBitCount = 19*BitsPerByte; // Start with header length


	// The first option must be non-compression
	// so that we can set the block to reference to
//	if(myEncoderList[0]->getSelection() != NoCompressionOpt)
//	{
//		exception wrongTypeException; //:TODO: need title
//
//		throw wrongTypeException;
//	}

	//:TODO: Nest this loop in another and iterate over the next residual block
	std::vector<AdaptiveEntropyEncoder*>::iterator winningIteration;

    CodingSelection winningSelection;

    timestamp_t t0_intermediate, t1_intermediate, t2_intermediate, t3_intermediate;

    timestamp_t t0 = getTimestamp();

    // Should only need to get the residuals once for a given raw image set
    ushort* residualsPtr = myPreprocessor.getResiduals(mySamples);

    timestamp_t t1 = getTimestamp();

    cout << "Prediction processing time ==> " << fixed << getSecondsDiff(t0, t1) << " seconds"<< endl;


    timestamp_t t2 = getTimestamp();

    int blockIndex(0);
    unsigned int encodedLength(0);

    long totalSamples = myXDimension*myYDimension*myZDimension;


    // Note convention of specifying host memory prefixed by 'h_' and device by 'd_'


    // Place the pre-processed data on the GPU
    //:TODO: This probably better belongs in the Predictor constructor and
    // freed in the destructor - in other words what would be expected in
    // C++ programs
    //***************************************************************************
    ushort *d_PreProcessedImageData;


   	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_PreProcessedImageData, sizeof(ushort)*totalSamples));
   	CUDA_CHECK_RETURN(cudaMemcpy(d_PreProcessedImageData, residualsPtr, sizeof(ushort)*totalSamples, cudaMemcpyHostToDevice));



    unsigned char* d_EncodedBlocks(0);
   	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_EncodedBlocks, MaximumEncodedMemory));
   	CUDA_CHECK_RETURN(cudaMemset(d_EncodedBlocks, 0, MaximumEncodedMemory));
    //***************************************************************************

   	// Allocate space for encoded block sizes -- the number of elements is the total samples
   	// divided by the sample block size
    unsigned int* d_EncodedBlockSizes(0);
   	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_EncodedBlockSizes, sizeof(unsigned int)*((Rows*Columns*Bands)/32)));
   	CUDA_CHECK_RETURN(cudaMemset(d_EncodedBlockSizes, 0, sizeof(unsigned int)*((Rows*Columns*Bands)/32)));
    //***************************************************************************

    //:TODO: This is one of the 1st places where we will start looking
    // at applying Amdahl's Law!!!
//   	const int NumberThreadsPerBlock(32);
//   	const int NumberOfBlocks(totalSamples/NumberThreadsPerBlock);
//   	encodingKernel<<<NumberOfBlocks, NumberThreadsPerBlock>>> (gpuPreProcessedImageData);




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

   	encodingKernel<<<gridBlocks, threadsPerBlock>>> (d_PreProcessedImageData, d_EncodedBlocks, d_EncodedBlockSizes);

   	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    timestamp_t t3 = getTimestamp();


    cout << "\nRepresentative intermediate Encoding processing times ==> " << fixed
            << "\n(intermediate t0-t1): " << fixed << getSecondsDiff(t0_intermediate, t1_intermediate) << " seconds"
            << "\n(intermediate t1-t2): " << fixed << getSecondsDiff(t1_intermediate, t2_intermediate) << " seconds"
            << "\n(intermediate t2-t3): " << fixed << getSecondsDiff(t2_intermediate, t3_intermediate) << " seconds\n" << endl;

    cout << "Encoding processing time ==> " << fixed << getSecondsDiff(t2, t3) << " seconds"<< endl;


	unsigned char* h_EncodedBlock = new unsigned char[MaximumEncodedMemory];
   	CUDA_CHECK_RETURN(cudaMemcpy(h_EncodedBlock, d_EncodedBlocks, MaximumEncodedMemory, cudaMemcpyDeviceToHost));


   	unsigned int h_EncodedBlockSize[(Rows*Columns*Bands)/32] = {0};
   	CUDA_CHECK_RETURN(cudaMemcpy(h_EncodedBlockSize, d_EncodedBlockSizes, sizeof(unsigned int) * (Rows*Columns*Bands)/32, cudaMemcpyDeviceToHost));


    // Print out the encodings. Note that every MaximumEncodedBytes byte
   	// index is a Code ID
   	//=========================================================================
    for(ulong index=0; index<=2000; index+=2)
    {
    	if(!(index%MaximumEncodedBytes))
    	{
    		cout << "\nCodeID=" << dec << ((h_EncodedBlock[index] & 0xf0) >> 4) << endl;
    		cout << "============================" << endl;
    	}

    	cout << "h_EncodedBlock[" << dec << index << "]=0x" << hex << setfill('0') << int(h_EncodedBlock[index]) << " (size:" << " h_EncodedBlock["<< dec << index+1 << "]=0x" << hex <<  int(h_EncodedBlock[index+1]) << endl;
    }

    cout << endl;
   	//=========================================================================

    for(int index=0; index<=32; index++)
     {
     	cout << "Block:" << index << "(size:" << dec << h_EncodedBlockSize[index] << ")" << endl;
     }

    cudaFree(d_EncodedBlocks);
    cudaFree(d_EncodedBlockSizes);


//    //===========================================================
//    int dataIndex(0);
//    cout << "Debug encoded stream on host - dataIndex=" << dataIndex << "==>";
//    for(int i=0; i<64; i++)
//    {
//    	boost::dynamic_bitset<unsigned char> debugEncodedStream(8, h_EncodedBlock[dataIndex*64 + i]);
//    	cout << debugEncodedStream;
//    }
//	cout << endl;
//	dataIndex = 1;
//    cout << "Debug encoded stream on host - dataIndex=" << dataIndex << "==>";
//    for(int i=0; i<64; i++)
//    {
//    	//boost::dynamic_bitset<unsigned char> debugEncodedStream(8, h_EncodedBlock[dataIndex*64 + i]);
//    	boost::dynamic_bitset<unsigned char> debugEncodedStream(8, h_EncodedBlock[ i + 64]);
//    	cout << debugEncodedStream;
//    }
//	cout << endl;
//    //===========================================================

    boost::dynamic_bitset<unsigned char> encodedStream;
    boost::dynamic_bitset<unsigned char> nextEncodedStream;
    boost::dynamic_bitset<unsigned char> packedData;

    size_t currentBitPosition(0);

		unsigned char lastByte(0);
		int differenceInBits(0);

        size_t partialBits(0);

	 //for(ulong dataIndex=0; dataIndex<20; dataIndex++) // (DEBUGGING - 196630 max)
	 for(ulong dataIndex=0; dataIndex<MaximumThreads; dataIndex++)
     {

		 ulong partitianIndex = (dataIndex)*MaximumEncodedBytes;
     	int numberOfBitsInBlock = h_EncodedBlockSize[dataIndex];


         int blockEnd = numberOfBitsInBlock/BitsInByte;
         if(numberOfBitsInBlock % BitsInByte)
         {
        	 blockEnd++;
         }


    	 // capture the first byte and combine with previous last if exist
         unsigned char firstByte(h_EncodedBlock[partitianIndex]);
         if(partialBits)
         {
        	 firstByte <<= partialBits;
         }


    	 nextEncodedStream.resize(0);


       	for(ulong j=partitianIndex; j <(partitianIndex+blockEnd); j++)
       	{
     		nextEncodedStream.append(h_EncodedBlock[j]);
       	}

       	vector<unsigned char> packedDataBlocks(nextEncodedStream.num_blocks());
       	vector<unsigned char>::iterator it;

       	//populate vector blocks
 	    boost::to_block_range(nextEncodedStream, packedDataBlocks.begin());

       	if(partialBits)
       	{
       		unsigned char firstByte = packedDataBlocks.front();
       		firstByte >>= partialBits;

       		nextEncodedStream >>= partialBits;
     	    boost::to_block_range(nextEncodedStream, packedDataBlocks.begin());


     	   it = packedDataBlocks.begin();
     	   *it = firstByte;

     	   packedData.resize(packedData.size()-BitsInByte);
       	}



 	    for (it = packedDataBlocks.begin(); it != packedDataBlocks.end(); ++it)
 	    {
 	    	packCompressedData(*it, packedData);
   	    }


        partialBits = (numberOfBitsInBlock % BitsInByte);


        nextEncodedStream.resize(0);

     }

     writeCompressedData(packedData);

    delete [] h_EncodedBlock;

}


void Sensor::sendHeader()
{
    // Note: Header is not completely populated for all defined parameters.
    // Only what is applicable to the selected test raw data to
    // identify identical information. Also, probably not the most
    // clean way to fill in fields.

    CompressedHeader header = {0};

    // Collect the structure data
    header.xDimension = myXDimension;
    header.yDimension = myYDimension;
    header.zDimension = myZDimension;
    bigEndianVersusLittleEndian(header.xDimension);
    bigEndianVersusLittleEndian(header.yDimension);
    bigEndianVersusLittleEndian(header.zDimension);

    //----------------------------------------------------------------------------
    bool signedSamples(false);
    bool bsq(true);
    header.signSampDynRangeBsq1 |= signedSamples;         header.signSampDynRangeBsq1 <<= 2; // reserved
                                                          header.signSampDynRangeBsq1 <<= 4;
    header.signSampDynRangeBsq1 |= (DynamicRange & 0xf);  header.signSampDynRangeBsq1 <<= 1;
    header.signSampDynRangeBsq1 |= bsq;
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    bool blockType(true);

    header.wordSizEncodeMethod |= blockType;  header.wordSizEncodeMethod <<= 2;
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------

    header.predictBandMode |= UserInputPredictionBands;  header.predictBandMode <<= 2;
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    bool neighborSum(true);

    header.neighborRegSize |= neighborSum;  header.neighborRegSize <<= 7;
    header.neighborRegSize |= RegisterSize;
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    header.predictWeightResInit |= (PredictionWeightResolution - 4);  header.predictWeightResInit <<= 4;

    //:TODO:
//    unsigned int scaledWeight = log2(PredictionWeightInterval);
//    header.predictWeightResInit |= (scaledWeight - 4);
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    header.predictWeightInitFinal |= (PredictionWeightInitial + 6);  header.predictWeightInitFinal <<= 4;

    header.predictWeightInitFinal |= (PredictionWeightFinal + 6);
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    header.blockSizeRefInterval |= (0x40);  // Block size 32 flag

    ushort refInterval(ReferenceInterval);
    bigEndianVersusLittleEndian(refInterval);

    header.blockSizeRefInterval |= refInterval;
    //----------------------------------------------------------------------------

    boost::dynamic_bitset<unsigned char> packedData;
    packCompressedData(header.userData, packedData);                // byte 0
    packCompressedData(header.xDimension, packedData);              // bytes 1,2
    packCompressedData(header.yDimension, packedData);              // bytes 3,4
    packCompressedData(header.zDimension, packedData);              // bytes 5,6
    packCompressedData(header.signSampDynRangeBsq1, packedData);    // byte 7
    packCompressedData(header.bsq, packedData);                     // bytes 8,9
    packCompressedData(header.wordSizEncodeMethod, packedData);     // bytes 10,11
    packCompressedData(header.predictBandMode, packedData);         // byte 12
    packCompressedData(header.neighborRegSize, packedData);         // byte 13
    packCompressedData(header.predictWeightResInit, packedData);    // byte 14
    packCompressedData(header.predictWeightInitFinal, packedData);  // byte 15
    packCompressedData(header.predictWeightTable, packedData);      // byte 16
    packCompressedData(header.blockSizeRefInterval, packedData);    // bytes 17,18


    size_t bitsPerBlock = packedData.bits_per_block;
    size_t numBlocks = packedData.num_blocks();


    writeCompressedData(packedData);

}

void Sensor::sendEncodedSamples(boost::dynamic_bitset<> &encodedStream, unsigned int encodedLength)
{
	bool endFlag(false);

	// if 0, then whatever is there can be appended and sent
	if(encodedLength == 0)
	{
		endFlag = true;
	}


	size_t bytes = encodedStream.size() / BitsPerByte;
	if (encodedStream.size() % BitsPerByte)
	{
		unsigned int previousSize = encodedStream.size();
		bytes++;
		unsigned int newSize = bytes * BitsPerByte;
		encodedStream.resize(newSize);
		encodedStream <<= (newSize - previousSize);
	}

	// this does two things - (1) changes the block size from ulong to
	// unsigned char and (2) reverses the byte order
	// Note that this algorithm was largely arrived at empirically. Looking
	// at the data to see what is correct. Keep this in mind when defining
	// architectural decisions, and when there may exist reluctance
	// after prototyping activities.

	boost::dynamic_bitset<unsigned char> convertedStream(encodedStream.size());
	for (int byteIndex = 0; byteIndex < bytes; byteIndex++)
	{
		int targetByte = bytes - byteIndex - 1;
		int sourceByte = byteIndex;

		for (int bitIndex = 0; bitIndex < BitsPerByte; bitIndex++)
		{
			int targetBit = (targetByte * BitsPerByte) + bitIndex;
			int sourceBit = (sourceByte * BitsPerByte) + bitIndex;

			convertedStream[targetBit] = encodedStream[sourceBit];
		}

	}

	writeCompressedData(convertedStream, encodedStream.size(), true);

}

void Sensor::writeCompressedData(boost::dynamic_bitset<unsigned char> &packedData, size_t bitSize, bool flag)
{
    
	// A non-default bit size might be specified, but this must be adjusted to the nearest
	// full bit
	if (!bitSize)
	{
		bitSize=packedData.size();
	}

    size_t numberOfBytes = bitSize/BitsPerByte;
    if(bitSize % BitsPerByte)
    {
    	numberOfBytes++;
    }


    vector<unsigned char> packedDataBlocks(packedData.num_blocks());

    //populate vector blocks
    boost::to_block_range(packedData, packedDataBlocks.begin());

    //write out each block
    for (vector<unsigned char>::iterator it =
            packedDataBlocks.begin(); it != packedDataBlocks.end(); ++it)
    {
        //retrieves block and converts it to a char*
        mySource->sendEncodedData(reinterpret_cast<char*>(&*it));

        // if we've written the targeted number of bytes
        // return
        numberOfBytes--;
        if(!numberOfBytes)
        {
        	break;
        }
    }
}

bool Sensor::getLastByte(unsigned char &lastByte)
{
    // Get the last byte written, and in some cases, reset the file pointer to the one previous

    bool partialByteFlag(false);

    int byteIndex = myEncodedBitCount % BitsPerByte;
    if(byteIndex)
    {
        lastByte = (mySource->getEncodedData())[byteIndex];

        partialByteFlag = true;
    }

    unsigned int putByte = myEncodedBitCount / BitsPerByte;

    mySource->setNextInsertionByte(putByte);

    return partialByteFlag;

}


