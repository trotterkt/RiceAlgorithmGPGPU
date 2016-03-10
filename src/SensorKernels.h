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

  } while (src[i++] != 0);

  return dest;
}

__device__ char * cuda_strcat(char *dest, const char *src)
{
  int i = 0;

  while (dest[i] != 0) i++;

  cuda_strcpy(dest+i, src);
  return dest;
}

__device__ const char *byte_to_binary(unsigned char* x, int numberOfBits)
{
    const int MaximumBitLength(504);

    char b[MaximumBitLength];

    b[0] = '\0';


    for(int bitIndex=0; bitIndex<numberOfBits; )
    {
        unsigned int z;
        for (z = 0x80; z >= 0x1; z >>= 1)
        {
            cuda_strcat(b, ((x[bitIndex/BitsInByte] & z) == z) ? "1" : "0");
            bitIndex++;
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
	const int MaximumByteArray(56);
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

    	//***************************************************
    	// do some index checking
    	if((byteIndex + byteShift) >= MaximumByteArray)
    	{
    		printf("We have an error  in shiftRight-- (byteIndex + byteShift)=%d\n", (byteIndex + byteShift));
    		return;
    	}
    	//***************************************************

    	arrayCopy[byteIndex + byteShift] = (array[byteIndex]) >> bitShift;

		if (byteIndex > 0)
		{
			arrayCopy[byteIndex + byteShift] |= previousBits;
		}

		previousBits = (array[byteIndex] & mask) << (BitsInByte - bitShift);
	}

	//***************************************************
	// do more index checking
	if((numberOfBytes) >= MaximumByteArray)
	{
		printf("We have an error  in shiftRight-- (numberOfBytes)=%d\n", numberOfBytes);
		return;
	}
	//***************************************************

	memcpy(array, arrayCopy, numberOfBytes);

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
	const int MaximumByteArray(56);
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
    	if((byteIndex - byteShift) < 0)
    	{
    		break;
    	}

    	previousBits = (array[byteIndex+1] & (mask << (BitsInByte - bitShift)));
    	previousBits >>= (BitsInByte - bitShift);

    	//***************************************************
    	// do some index checking
    	if((byteIndex - byteShift) >= MaximumByteArray)
    	{
    		printf("We have an error  in shiftLeft -- (byteIndex - byteShift)=%d\n", (byteIndex + byteShift));
    		return;
    	}
    	//***************************************************

		arrayCopy[byteIndex - byteShift] = (array[byteIndex]) << bitShift;

		if (byteIndex <= (numberOfBytes-1))
		{
			arrayCopy[byteIndex - byteShift] |= previousBits;
		}

	}

	//***************************************************
	// do more index checking
	if((numberOfBytes) >= MaximumByteArray)
	{
		printf("We have an error in shiftLeft -- (numberOfBytes)=%d\n", numberOfBytes);
		return;
	}
	//***************************************************

	memcpy(array, arrayCopy, numberOfBytes);
}

__device__ void bitwiseOr(unsigned char* byteFirst, unsigned char* byteSecond, unsigned int numberOfBytes, unsigned char* outByte)
{
    for(int i=0; i<numberOfBytes; i++)
    {
       outByte[i] =  byteFirst[i] | byteSecond[i];
    }
}

__device__ void bitwiseAnd(unsigned char* byteFirst, unsigned char* byteSecond, unsigned int numberOfBytes, unsigned char* outByte)
{
    for(int i=0; i<numberOfBytes; i++)
    {
       outByte[i] =  byteFirst[i] & byteSecond[i];
    }
}

//NOTE: CUDA does not support passing reference to kernel argument
//__device__ unsigned int splitSequenceEncoding(ushort* inputSamples, unsigned int dataIndex, RiceAlgorithm::CodingSelection &selection, unsigned char* encodedStream)
__device__ unsigned int splitSequenceEncoding(ushort* inputSamples, unsigned int dataIndex, RiceAlgorithm::CodingSelection* selection, unsigned char* encodedStream)
{
	// Apply SplitSequence encoding
	//=========================================================================================================

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
            *selection = RiceAlgorithm::CodingSelection(k);
        }
    }

    size_t encodedSizeList[32];
    unsigned int totalEncodedSize(0);

    // Get the total encoded size first
    for(int index = dataIndex; index < (dataIndex+32); index++)
    {
        size_t encodedSize = (inputSamples[index] >> *selection) + 1;
        encodedSizeList[index-dataIndex] = encodedSize;
        totalEncodedSize += int(encodedSize);

    }

    // include space for the  code option
    totalEncodedSize += int(RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);


	// Not allocating from global memory is significantly faster
	const int MaximumByteArray(50);
    unsigned char localEncodedStream[MaximumByteArray];
    //=========================================================================================================

    memset(localEncodedStream, 0, MaximumByteArray);


    // assign each encoded sample and shift by the next one
    // at the end of the loop, we will assign the last one
    // unsigned char* localEncodedStream(0);
    for(int index = 31; index >= 0; index--)
    {

    	localEncodedStream[0] |= 0x1;

    	int shift = encodedSizeList[index];

        shiftRight(localEncodedStream, totalEncodedSize, shift);

    }
    //localEncodedStream[0] |= 1;



	// see Lossless Data Compression, Blue Book, sec 5.1.2
    // place the code encoding selection
    //shiftRight(localEncodedStream, totalEncodedSize, int(RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp));
    unsigned char selectionEncoding[MaximumByteArray] = {0};
    selectionEncoding[0] = *selection + 1;
    //shiftLeft(selectionEncoding, MaximumByteArray, (sizeof(unsigned char)*BitsInByte) - int(RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp));
    bitwiseOr(localEncodedStream, selectionEncoding, MaximumByteArray, localEncodedStream);



    if(dataIndex <= 96)
    {
        printf("Line #306, BlockInx=%d totalEncodedSize=%d encodedStream(size:%d)=%s\n", dataIndex, totalEncodedSize, totalEncodedSize, byte_to_binary(localEncodedStream, 81));
    }

    // determine number of bytes
    unsigned int numberOfBytes(totalEncodedSize/RiceAlgorithm::BitsPerByte);
    if(totalEncodedSize%RiceAlgorithm::BitsPerByte)
    {
    	numberOfBytes++;
    }


    //localEncodedStream[0] = 1;

    //=========================================================================================================


    //=========================================================================================================
    // after the zero sequence number that was split off, then we add that value to the stream
    // for each of the samples
    unsigned short mask = powf(2, *selection) - 1;

    const unsigned int MaximumByteAdditionalArray(56); // 14*32/BitsInByte
    const unsigned int additionalEncodedSize(*selection * 32 * BitsInByte);


    unsigned char encodedSample[MaximumByteAdditionalArray] = {0};
    unsigned char individualEncodedSample[MaximumByteAdditionalArray];


    for(int index = 0; index < 32; index++)
    {
        unsigned short maskedSample = inputSamples[index+dataIndex] & mask;
        unsigned char byteConvert[2] = {((maskedSample&0xff00)>>8), (maskedSample&0xff)}; //:KLUDGE: need to change the number into
                                                                                          // a byte form for printing only -- endian issue?

        memset(individualEncodedSample, 0, sizeof(individualEncodedSample));

        memcpy(individualEncodedSample, &byteConvert, sizeof(byteConvert));

        // This shift aligns the encoding at the beginning of the array
        shiftLeft(individualEncodedSample, *selection, ((sizeof(byteConvert)*BitsInByte) - *selection));


    	//***************************************************
        // This shift aligns the encoding at the proper relative position in the array
        shiftRight(individualEncodedSample, (*selection * index), (*selection * index));


        // Finally merge the individual sample into this segment of the encoded stream
        bitwiseOr(encodedSample, individualEncodedSample, MaximumByteAdditionalArray, encodedSample);

    }

    //=========================================================================================================

    shiftRight(encodedSample, MaximumByteAdditionalArray, totalEncodedSize);
    totalEncodedSize += (32 * *selection);

    bitwiseOr(localEncodedStream, encodedSample, MaximumByteArray, localEncodedStream);

    // :TODO: Unclear why offset still exists
    shiftLeft(localEncodedStream, MaximumByteArray, RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);


    //:TODO: Another possible source of device to host transfer problem
    // I think the data is already segmented in 32 sample blocks
    // memcpy(&encodedDataPtr[dataIndex], localEncodedStream, numberOfBytes);

    unsigned char* encodedDataPtr = &encodedStream[dataIndex];

    memcpy(encodedDataPtr, localEncodedStream, numberOfBytes);


	return code_len;
}

/**
 * CUDA kernel that identifies the winning encoding scheme for each block
 */
__global__ void encodingKernel(ushort* inputSamples, unsigned char* gpuEncodedBlocks)
{
	// Operate on all samples for a given block together

	unsigned int dataIndex(0);

	// http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

//***************************************
//if(threadId > 4) return; // DEBUGGING
//***************************************

	dataIndex = threadId;


//	if(threadId)
//	{
//		//dataIndex = (threadId % 196608);
//	}
//	else if (threadId >= 196607) // temporary debugging
//	{
//		printf("VERY BAD!!!!! threadId=%d\n", threadId);
//		return;
//	}


	RiceAlgorithm::CodingSelection selection;
	unsigned int encodedLength(0);
	unsigned int winningEncodedLength = -1;
	RiceAlgorithm::CodingSelection winningSelection;

	// Apply SplitSequence encoding
	encodedLength = splitSequenceEncoding(inputSamples, dataIndex*32, &selection, gpuEncodedBlocks);  // index by the block size or 32

	// Find the winning encoding for all encoding types
    // This basically determines the winner
    if (encodedLength < winningEncodedLength)
    {
        //*this = *(*iteration);
        winningEncodedLength = encodedLength;
        winningSelection = selection;

        //encodedSize = (*iteration)->getEncodedBlockSize();
    }

	if(dataIndex <= 7)
	{
	    printf("Line #430, (encodedLength=%d) dataIndex=%d gpuEncodedBlocks=%s\n",
	    		encodedLength, dataIndex, byte_to_binary(&gpuEncodedBlocks[dataIndex*32], encodedLength));
	}
    //*************************************************************
    // Once here, synchronization among all threads should happen
    // Note that this is only applicable, for threads within a given
    // block. But we do not want to return until all are available.
    // Need to sync on the host.
    //*************************************************************
    __syncthreads();
}



#endif /* SENSORKERNELS_H_ */
