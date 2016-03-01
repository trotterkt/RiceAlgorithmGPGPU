/*
 * RiceAlgorithmKernels.h
 *
 *  Created on: Feb 23, 2016
 *      Author: ktrotter
 */

#ifndef SENSORKERNELS_H_
#define SENSORKERNELS_H_

#include <stdio.h>
#include "math.h" // CUDA Math library
#include <AdaptiveEntropyEncoder.h>

using namespace std;

const int BitsInByte(8);

// Do not have access to string library in CUDA, so need
// to create own
__device__ char * cuda_strcpy(char *dest, const char *src)
{
  int i = 0;
  do
  {
    dest[i] = src[i];
  }
  while (src[i++] != 0);

  return dest;
}

__device__ char * cuda_strcat(char *dest, const char *src)
{
  int i = 0;
  while (dest[i] != 0) i++;

  cuda_strcpy(dest+i, src);
  return dest;
}

__device__ const char *byte_to_binary(unsigned char* x, int numberOfBytes)
{
    const int MaximumBitLength(504);

    static char b[MaximumBitLength] = {0};

    b[0] = '\0';

    for(int byteIndex=0; byteIndex<numberOfBytes; byteIndex++)
    {
        int z;
        for (z = 0x80; z > 0; z >>= 1)
        {
            cuda_strcat(b, ((x[byteIndex] & z) == z) ? "1" : "0");
        }
    }

    return b;
}

//*********************************************************************
// Architectural consideration for resource allocation
// This function would be implemented by a C programmer.
// Replacement for boost::dynamic_bitset on host. Also note that
// Cuda math library is used vs. host version. (no pow(), powf() instead)
//*********************************************************************
__device__ void shiftRight(unsigned char* array, unsigned int bitSize, unsigned int arrayBitShift)
{
	unsigned int numberOfBytes(bitSize/BitsInByte);

	if(bitSize % BitsInByte)
	{
		numberOfBytes++;
	}

	// Decide where in the copy the new bytes will go
	//unsigned char* arrayCopy = new unsigned char[numberOfBytes];
	// Not allocating from global memory is significantly faster
	const int MaximumByteArray(20);
	unsigned char arrayCopy[MaximumByteArray] = {0};

	// Shift from bit to bit, and byte to byte
	unsigned int byteShift = arrayBitShift / BitsInByte;
	unsigned int bitShift = arrayBitShift % BitsInByte;

	// Include neighboring bits to transfer to next byte
	// First figure out the mask
	unsigned char mask = powf(2, bitShift) - 1;
	unsigned char previousBits(0);


	// Copy from byte to shifted byte
    for(unsigned int byteIndex=0; byteIndex<numberOfBytes; byteIndex++)
    {
    	// don't shift larger than the size of the stream
    	if((byteIndex + byteShift) >= numberOfBytes)
    	{
    		break;
    	}

		arrayCopy[byteIndex + byteShift] = (array[byteIndex]) >> bitShift;

		if (byteIndex > 0)
		{
			arrayCopy[byteIndex + byteShift] |= previousBits;
		}

		previousBits = (array[byteIndex] & mask) << (BitsInByte - bitShift);
	}

	memcpy(array, arrayCopy, numberOfBytes);

	//delete [] arrayCopy;
}


__device__ void shiftLeft(unsigned char* array, unsigned int bitSize, unsigned int arrayBitShift)
{
	unsigned int numberOfBytes(bitSize/BitsInByte);

	if(bitSize % BitsInByte)
	{
		numberOfBytes++;
	}

	// Decide where in the copy the new bytes will go
	//unsigned char* arrayCopy = new unsigned char[numberOfBytes];
	// Not allocating from global memory is significantly faster
	const int MaximumByteArray(20);
	unsigned char arrayCopy[MaximumByteArray] = {0};

	memset(arrayCopy, 0, sizeof(arrayCopy));

	// Shift from bit to bit, and byte to byte
	unsigned int byteShift = arrayBitShift / BitsInByte;
	unsigned int bitShift = arrayBitShift % BitsInByte;

	// Include neighboring bits to transfer to next byte
	// First figure out the mask
	unsigned char mask = powf(2, bitShift) - 1;
	unsigned char previousBits(0);


	// Copy from byte to shifted byte
    for(unsigned int byteIndex=byteShift; byteIndex<numberOfBytes; byteIndex++)
    {
    	// don't shift larger than the size of the stream
//    	if((byteIndex - byteShift) < 0)
//    	{
//    		break;
//    	}

    	previousBits = (array[byteIndex+1] & (mask << (BitsInByte - bitShift)));
    	previousBits >>= (BitsInByte - bitShift);

		arrayCopy[byteIndex - byteShift] = (array[byteIndex]) << bitShift;

		if (byteIndex <= (numberOfBytes-1))
		{
			arrayCopy[byteIndex - byteShift] |= previousBits;
		}

	}

	memcpy(array, arrayCopy, numberOfBytes);
//
//	delete [] arrayCopy;
}

//NOTE: CUDA does not support passing reference to kernel argument
//__device__ unsigned int splitSequenceEncoding(ushort* inputSamples, unsigned int dataIndex, RiceAlgorithm::CodingSelection &selection, unsigned char* encodedStream)
__device__ unsigned int splitSequenceEncoding(ushort* inputSamples, unsigned int dataIndex, RiceAlgorithm::CodingSelection* selection, unsigned char* encodedStream)
{
	 RiceAlgorithm::CodingSelection selection2;

	// Apply SplitSequence encoding
	//===========================================================

    unsigned int code_len = (unsigned int)-1;
    int i = 0, k = 0;
    int k_limit = 14;


    for(k = 0; k < k_limit; k++)
    {

        unsigned int code_len_temp = 0;
        for(i = dataIndex; i < (dataIndex+32); i++)
        {
        	ushort encodedSample = inputSamples[i] >> k;
            code_len_temp += (encodedSample) + 1 + k;
        }

        if(code_len_temp < code_len)
        {
            code_len = code_len_temp;
            selection2 = RiceAlgorithm::CodingSelection(k);
        }
    }

    size_t encodedSizeList[32];
    unsigned int totalEncodedSize(0);

    // Get the total encoded size first
    for(int index = dataIndex; index < (dataIndex+32); index++)
    {
        size_t encodedSize = (inputSamples[index] >> selection2) + 1;
        encodedSizeList[index-dataIndex] = encodedSize;
        totalEncodedSize += int(encodedSize);

        if(dataIndex < 32)
        printf("index=%d BlockInx=%d totalEncodedSize=%d\n", index, dataIndex, totalEncodedSize );
    }

    // include space for the  code option
//    totalEncodedSize += int(RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);

    if(dataIndex < 96)
    printf("BlockInx=%d totalEncodedSize=%d\n", dataIndex, totalEncodedSize );

    // assign each encoded sample and shift by the next one
    // at the end of the loop, we will assign the last one
    // unsigned char* localEncodedStream(0);
	// Not allocating from global memory is significantly faster
	const int MaximumByteArray(20);
    unsigned char localEncodedStream[MaximumByteArray] = {0};


    localEncodedStream[0] = 1;


    // determine number of bytes
    unsigned int numberOfBytes(totalEncodedSize/RiceAlgorithm::BitsPerByte);
    if(totalEncodedSize%RiceAlgorithm::BitsPerByte)
    {
    	numberOfBytes++;
    }

    if(dataIndex <= 96)
    printf("BlockInx=%d totalEncodedSize=%d encodedStream(size:%d)=%s", dataIndex, totalEncodedSize, totalEncodedSize, byte_to_binary(localEncodedStream, numberOfBytes));

    //localEncodedStream = new unsigned char[numberOfBytes];


    // after the zero sequence number that was split off, then we add that value to the stream
    // for each of the samples
//    boost::dynamic_bitset<> maskBits(selection, 0xffff);
//    ulong mask(0xffff);
//
    for(int index = dataIndex; index < (dataIndex+32); index++)
    {
//        ushort maskedSample = inputSamples[index] & mask;
//
//        //:TODO: this section appears to be responsible for about 8 seconds in the
//        // total encoding time
//        //===================================================================================
//        boost::dynamic_bitset<> encodedSample(selection, maskedSample);
//        size_t encodedSize = (inputSamples[index] >> selection2) + 1;
//
//        shiftLeft(localEncodedStream, totalEncodedSize, sizeof(ushort) * RiceAlgorithm::BitsPerByte);
//
//
        totalEncodedSize += selection2;
//
//        shiftLeft(localEncodedStream, totalEncodedSize, selection2);
//
//        inputSamples[index]
//        localEncodedStream[numberOfBytes-1] |= encodedSample;
//
//        totalEncodedSize += (sizeof(ushort) * RiceAlgorithm::BitsPerByte);
//        //===================================================================================
    }

    if(dataIndex <= 96)
    printf("BlockInx=%d totalEncodedSize=%d encodedStream(size:%d)=%s", dataIndex, totalEncodedSize, totalEncodedSize, byte_to_binary(localEncodedStream, numberOfBytes));

//Come back here
	//if(blockIdx.x == 0 && (threadIdx.x == 0) && (threadIdx.y == 0)  && (threadIdx.z == 0)) // otherwise I get duplicate!!! WHY????? -- Different warps?
    for(int index = dataIndex; index < (dataIndex+32); index++)
    {
    	//encodedStream[dataIndex + (numberOfBytes-1)] = 0x1;
    	localEncodedStream[numberOfBytes-1] |= 0x1;
        //encodedStream <<= encodedSizeList[index];
    	//encodedStream[index-dataIndex] |= 1;
        //shiftLeft(&encodedStream[index-dataIndex], totalEncodedSize, encodedSizeList[index]);

    	//shiftLeft(localEncodedStream, totalEncodedSize, encodedSizeList[31-index]);
        //shiftLeft(localEncodedStream, totalEncodedSize, 2);

    }
    //localEncodedStream[0] |= 1;





/*
    // place 1 in least significant bit
    localEncodedStream[numberOfBytes-1] = 0x80;
    for(int index = (numberOfBytes-1); index>=0; index--)
    {
        shiftRight(localEncodedStream, totalEncodedSize, encodedSizeList[index]);
        localEncodedStream[numberOfBytes-1] = 0x80;
    }
*/
/*
    //********************************
    unsigned char localEncodedStreamTemp[MaximumByteArray] = {0};
    //localEncodedStreamTemp[numberOfBytes-1] = 0x0f;
    //localEncodedStreamTemp[0] = 0xaf;
    //localEncodedStreamTemp[1] = 0xcc;
    localEncodedStreamTemp[10] |= 0x1;
    //shiftLeft(localEncodedStreamTemp, totalEncodedSize, 4);
    //shiftRight(localEncodedStreamTemp, totalEncodedSize, 1);
    memcpy(&encodedStream[dataIndex], localEncodedStreamTemp, numberOfBytes);
    //********************************
*/
    //memset(&encodedStream[dataIndex], 0, numberOfBytes);
    memcpy(&encodedStream[dataIndex], localEncodedStream, numberOfBytes);


//  if(dataIndex==0)
//	 printf("\nBlock:%d dataIndex=%d threadIdx.x=%d threadIdx.y=%d threadIdx.z=%d\n", blockIdx.x, dataIndex, threadIdx.x, threadIdx.y, threadIdx.z);

//	if(blockIdx.x == 0 && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
//	{
//		printf("\nBlock:%d selection:%d code_len=%d numberOfBytes=%d encoded[next to last]=%x encoded[last]=%x \n", blockIdx.x, selection, code_len, numberOfBytes, encodedStream[dataIndex+30], encodedStream[dataIndex+31]);
//
//	}

    // include space for the  code option :TODO: This only happens once
    //totalEncodedSize += RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp;

	//===========================================================

	//cudaMemcpy(encodedStream, localEncodedStream, code_len);

	//delete [] localEncodedStream;

	return code_len;
}

/**
 * CUDA kernel that identifies the winning encoding scheme for each block
 */
__global__ void encodingKernel(ushort inputSamples[32], unsigned char* gpuEncodedBlocks)
{
	// Operate on all samples for a given block together
	unsigned int sampleIndex = threadIdx.x;
	//unsigned int dataIndex = (threadIdx.x + blockIdx.y*blockDim.x + blockIdx.z*blockDim.y*blockDim.z);
	unsigned int dataIndex = blockIdx.x * 32;


//	if(!dataIndex)
//	//if(blockIdx.x <= 2 && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
//	{
//		for(int index=0; index<32; index++)
//		{
//			printf("Block:%d data[%d]=%d\n", blockIdx.x, index, inputSamples[dataIndex+index]);
//		}
//	}




	RiceAlgorithm::CodingSelection selection;
	unsigned int encodedLength(0);
	unsigned int winningEncodedLength = -1;
	RiceAlgorithm::CodingSelection winningSelection;

	// Apply SplitSequence encoding
	encodedLength = splitSequenceEncoding(inputSamples, dataIndex, &selection, gpuEncodedBlocks);

	// Find the winning encoding for all encoding types
    // This basically determines the winner
    if (encodedLength < winningEncodedLength)
    {
        //*this = *(*iteration);
        winningEncodedLength = encodedLength;
        winningSelection = selection;

        //encodedSize = (*iteration)->getEncodedBlockSize();
    }

	if(!dataIndex)
	{
		//memcpy(gpuEncodedBlocks, gpuEncodedBlocks, winningEncodedLength);
	   // unsigned char array[] = { 0xAC, 0xFF, 0xCC, 0x55, 0xAC, 0xFF, 0xCC, 0x55, 0xAC, 0xFF, 0xCC, 0x55 };
       // memcpy(&gpuEncodedBlocks[dataIndex], array,  12);


        //cudaMemcpy(gpuEncodedBlocks, array, sizeof(array), cudaMemcpyDeviceToDevice);
	    //gpuEncodedBlocks[0] = array[0];
	    //gpuEncodedBlocks[1] = array[1];
	    //gpuEncodedBlocks[2] = array[2];
	}
    //*************************************************************
    // Once here, synchronization among all threads should happen
    // Note that his is only applicable, for threads within a given
    // block. But we do not want to return until all are available.
    //*************************************************************
    __syncthreads();


}



#endif /* SENSORKERNELS_H_ */
