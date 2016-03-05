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

    __shared__ static char b[MaximumBitLength];

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
	 RiceAlgorithm::CodingSelection selection2;

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
    //for(int index = (dataIndex+31); index >= 0; index--)
    for(int index = 31; index >= 0; index--)
    {
//    	//localEncodedStream[totalEncodedSize/BitsInByte] |= 0x1;
    	localEncodedStream[0] |= 0x1;
//    	//localEncodedStream <<= encodedSizeList[index];
//        //shiftLeft(&localEncodedStream[index-dataIndex], totalEncodedSize, encodedSizeList[index]);
    	int shift = encodedSizeList[index];

        shiftRight(localEncodedStream, totalEncodedSize, shift);
      	  //localEncodedStream[0] |= 0x1;

        //if(dataIndex <= 0)
        //    printf("Line #237, BlockInx=%d totalEncodedSize=%d shift=%d encodedStream(size:%d)=%s\n", dataIndex, totalEncodedSize, shift, totalEncodedSize, byte_to_binary(localEncodedStream, 81));

    }
    //localEncodedStream[0] |= 1;

    if(dataIndex <= 0)
        printf("Line #237, BlockInx=%d totalEncodedSize=%d encodedStream(size:%d)=%s\n", dataIndex, totalEncodedSize, totalEncodedSize, byte_to_binary(localEncodedStream, 81));


    // determine number of bytes
    unsigned int numberOfBytes(totalEncodedSize/RiceAlgorithm::BitsPerByte);
    if(totalEncodedSize%RiceAlgorithm::BitsPerByte)
    {
    	numberOfBytes++;
    }


    //if(dataIndex <= 96)
    //printf("Line #247, BlockInx=%d totalEncodedSize=%d encodedStream(size:%d)=%s\n", dataIndex, totalEncodedSize, totalEncodedSize, byte_to_binary(localEncodedStream, numberOfBytes));

    localEncodedStream[0] = 1;



    //=========================================================================================================


    //=========================================================================================================
    // after the zero sequence number that was split off, then we add that value to the stream
    // for each of the samples
    unsigned short mask = powf(2, selection2) - 1;

    const unsigned int MaximumByteAdditionalArray(56); // 14*32/BitsInByte
    const unsigned int additionalEncodedSize(selection2 * 32 * BitsInByte);


    unsigned char encodedSample[MaximumByteAdditionalArray] = {0};
    unsigned char individualEncodedSample[MaximumByteAdditionalArray];


    for(int index = 0; index < 32; index++)
    {
        unsigned short maskedSample = inputSamples[index+dataIndex] & mask;
        unsigned char byteConvert[2] = {((maskedSample&0xff00)>>8), (maskedSample&0xff)}; //:KLUDGE: need to change the number into
                                                                                          // a byte form for printing only -- endian issue?
        //(*(reinterpret_cast<unsigned short*>(&byteConvert))) <<= ((sizeof(byteConvert)*BitsInByte)-selection);
        //==========================================================================================================
//        if(dataIndex <= 0)
//            printf("Line #298, maskedSample(%d)(%d)(0x%0x)=%s\n",
//                    index, maskedSample, inputSamples[index+dataIndex], byte_to_binary(byteConvert, 16));
//        cout << "Line #298, maskedSample(" << dec << index << ")(" << maskedSample << ")(0x" << hex << inputSamples[index+dataIndex] << ")=" << endl;
        //==========================================================================================================
        memset(individualEncodedSample, 0, sizeof(individualEncodedSample));

        memcpy(individualEncodedSample, &byteConvert, sizeof(byteConvert));

        // This shift aligns the encoding at the beginning of the array
        shiftLeft(individualEncodedSample, selection2, ((sizeof(byteConvert)*BitsInByte)-selection2));


    	//***************************************************
        // This shift aligns the encoding at the proper relative position in the array
        shiftRight(individualEncodedSample, (selection2*index), (selection2*index));


        // Finally merge the individual sample into this segment of the encoded stream
        bitwiseOr(encodedSample, individualEncodedSample, MaximumByteAdditionalArray, encodedSample);

    }

    //=========================================================================================================


    memcpy(&encodedStream[dataIndex], localEncodedStream, numberOfBytes);






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
	//unsigned int sampleIndex = threadIdx.x;
	//unsigned int dataIndex = (threadIdx.x + blockIdx.y*blockDim.x + blockIdx.z*blockDim.y*blockDim.z);
	//unsigned int dataIndex = blockIdx.x * 32;


//	if(!dataIndex)
//	//if(blockIdx.x <= 2 && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
//	{
//		for(int index=0; index<32; index++)
//		{
//			printf("Block:%d data[%d]=%d\n", blockIdx.x, index, inputSamples[dataIndex+index]);
//		}
//	}

	unsigned int dataIndex(0);

//	unsigned int blockId = blockIdx.x + (blockIdx.y * gridDim.x);
//	unsigned int threadId =  threadIdx.x + (threadIdx.y * blockDim.x);
//	unsigned int threadsPerBlock = (blockDim.x * blockDim.y);
//
//	unsigned int globalIdx = (blockId * threadsPerBlock) + threadId;
//
//
//	//:TODO: calculation of dataIndex still issue -- bad address
//	if(globalIdx)
//	{
//		dataIndex = (globalIdx * 32) - 1;
//	}

	// http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	//:TODO: calculation of dataIndex still issue -- bad address
	if(threadId)
	{
		dataIndex = (threadId * 32) - 1;
	}
	else if (threadId >= 196607)
	{
		printf("VERY BAD!!!!! threadId=%d\n", threadId);
		return;
	}







//if(dataIndex >= 0 && dataIndex < 64)
//    printf("dataIndex=%d, blockId=%d, threadId=%d globalIdx=%d\n", dataIndex, blockId, threadId, globalIdx);

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

	//if(!dataIndex)
	//{
		//memcpy(gpuEncodedBlocks, gpuEncodedBlocks, winningEncodedLength);
	   // unsigned char array[] = { 0xAC, 0xFF, 0xCC, 0x55, 0xAC, 0xFF, 0xCC, 0x55, 0xAC, 0xFF, 0xCC, 0x55 };
       // memcpy(&gpuEncodedBlocks[dataIndex], array,  12);


        //cudaMemcpy(gpuEncodedBlocks, array, sizeof(array), cudaMemcpyDeviceToDevice);
	    //gpuEncodedBlocks[0] = array[0];
	    //gpuEncodedBlocks[1] = array[1];
	    //gpuEncodedBlocks[2] = array[2];
	//}
    //*************************************************************
    // Once here, synchronization among all threads should happen
    // Note that his is only applicable, for threads within a given
    // block. But we do not want to return until all are available.
    //*************************************************************
    __syncthreads();


}



#endif /* SENSORKERNELS_H_ */
