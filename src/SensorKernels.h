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
const ulong MaximumThreads(Rows*Columns*Bands/BlockSize);
const int MaximumEncodedBytes(77);            // Observed maximum number of encoded bytes
const ulong MaximumEncodedMemory(MaximumThreads*MaximumEncodedBytes*BlockSize);

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
	const int MaximumByteArray(80);
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
	const int MaximumByteArray(80);
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

__device__ unsigned int getWinningEncodedLength(ushort* inputSamples, ulong dataIndex, RiceAlgorithm::CodingSelection* selection, size_t* encodedSizeList)
{
    unsigned int code_len = (unsigned int)-1;
    int i = 0, k = 0;
    int k_limit = 14;


    for(k = 0; k < k_limit; k++)
    {

        unsigned int code_len_temp = 0;
        for(i = dataIndex*32; i < (dataIndex*32)+32; i++)
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

    //size_t encodedSizeList[32];
    unsigned int totalEncodedSize(0);

    // Get the total encoded size first
    for(int index = dataIndex*32; index < (dataIndex*32)+32; index++)
    {
        size_t encodedSize = (inputSamples[index] >> *selection) + 1;
        encodedSizeList[index-(dataIndex*32)] = encodedSize; // Store Fundamental Sequence values
        totalEncodedSize += int(encodedSize);

    }

    // include space for the  code option
    totalEncodedSize += int(RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);

    code_len = totalEncodedSize;

    return code_len;

}

//NOTE: CUDA does not support passing reference to kernel argument
__device__ void splitSequenceEncoding(ushort* inputSamples, ulong dataIndex, RiceAlgorithm::CodingSelection* selection,
		                              unsigned char* d_EncodedBlocks, unsigned int totalEncodedSize, size_t* encodedSizeList)
{
	// Returning immediately if selection not within range
	// helps prevent thread divergence in warp
	if(*selection > RiceAlgorithm::K14)
	{
		return;
	}





	// Apply SplitSequence encoding
	//=========================================================================================================

//    unsigned int code_len = (unsigned int)-1;
//    int i = 0, k = 0;
//    int k_limit = 14;


//    for(k = 0; k < k_limit; k++)
//    {
//
//        unsigned int code_len_temp = 0;
//        for(i = dataIndex*32; i < (dataIndex*32)+32; i++)
//        {
//        	ushort encodedSample = inputSamples[i] >> k;
//            code_len_temp += (encodedSample) + 1 + k;
//        }
//
//        if(code_len_temp < code_len)
//        {
//            code_len = code_len_temp;
//            *selection = RiceAlgorithm::CodingSelection(k);
//        }
//    }

    //size_t encodedSizeList[32];
    //unsigned int totalEncodedSize(0);

    // Assemble number of zeros for encoding
//    for(int index = dataIndex*32; index < (dataIndex*32)+32; index++)
//    {
//        size_t encodedSize = (inputSamples[index] >> *selection) + 1;
//        encodedSizeList[index-(dataIndex*32)] = encodedSize;
//        //totalEncodedSize += int(encodedSize);
//
//    }

    // include space for the  code option
    //totalEncodedSize += int(RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);


	// Not allocating from global memory is significantly faster
	const int MaximumByteArray(80);
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



	// see Lossless Data Compression, Blue Book, sec 5.1.2
    // place the code encoding selection
    unsigned char selectionEncoding[MaximumByteArray] = {0};
    selectionEncoding[0] = *selection + 1;
    bitwiseOr(localEncodedStream, selectionEncoding, MaximumByteArray, localEncodedStream);



//    if(dataIndex <= 96)
//    {
//        printf("Line #306, BlockInx=%u d_EncodedBlocks(size:%d)=%s\n", dataIndex, totalEncodedSize, byte_to_binary(localEncodedStream, totalEncodedSize));
//    }



    //=========================================================================================================


    //=========================================================================================================
    // after the zero sequence number that was split off, then we add that value to the stream
    // for each of the samples
    unsigned short mask = powf(2, *selection) - 1;

    const unsigned int MaximumByteAdditionalArray(56); // 14*32/BitsInByte
    const unsigned int additionalEncodedSize(*selection * 32 * BitsInByte);


    //unsigned char encodedSample[MaximumByteAdditionalArray] = {0};
    unsigned char encodedSample[MaximumByteArray] = {0};
    unsigned char individualEncodedSample[MaximumByteArray];

    totalEncodedSize += (32 * *selection);


    for(int index = 0; index < 32; index++)
    {
        unsigned short maskedSample = inputSamples[index+dataIndex] & mask;
        unsigned char byteConvert[2] = {((maskedSample&0xff00)>>8), (maskedSample&0xff)}; //:KLUDGE: need to change the number into
                                                                                          // a byte form for printing only -- endian issue?

        memset(individualEncodedSample, 0, sizeof(individualEncodedSample));

    	//***************************************************
        // This shift aligns the encoding at the proper relative position in the array
        memcpy(individualEncodedSample, byteConvert, sizeof(byteConvert));

        shiftRight(individualEncodedSample, totalEncodedSize, ((*selection * index)));
        shiftLeft(individualEncodedSample, totalEncodedSize,  (BitsInByte*sizeof(byteConvert)) - *selection);


//        if(dataIndex <= 1)
//        {
//             printf("index=%3d,  maskedSample=%3x byteConvert[0]=%2x byteConvert[1]=%2x totalEncodedSize=%d   individualEncodedSample=%s\n", index, maskedSample, byteConvert[0], byteConvert[1], (totalEncodedSize), byte_to_binary(individualEncodedSample, totalEncodedSize));
//        }


        // Finally merge the individual sample into this segment of the encoded stream
        bitwiseOr(encodedSample, individualEncodedSample, MaximumByteArray, encodedSample);

//        if(dataIndex <= 0)
//         {
//             printf("index=%3d, shifRight=%2d maskedSample=0x%3x totalEncodedSize=%d                                  encodedSample=%s\n", index, (*selection * index), maskedSample, (totalEncodedSize), byte_to_binary(encodedSample, totalEncodedSize));
//         }

    }

    //=========================================================================================================

    // determine number of bytes
    unsigned int numberOfBytes(totalEncodedSize/RiceAlgorithm::BitsPerByte);
    if(totalEncodedSize%RiceAlgorithm::BitsPerByte)
    {
    	numberOfBytes++;
    }

    // :TODO: Unclear why offset still exists
    shiftLeft(localEncodedStream, numberOfBytes*BitsInByte, RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);

    shiftRight(encodedSample, totalEncodedSize, (totalEncodedSize - (32 * *selection)));

    //if(dataIndex <= 1)
    // {
    //     printf("LAST:  totalEncodedSize=%d       encodedSample=%s\n", (totalEncodedSize), byte_to_binary(encodedSample, totalEncodedSize));
    // }

    //if(dataIndex <= 1)
    //printf("BEFORE: totalEncodedSize=%d localEncodedStream=%s\n", (totalEncodedSize), byte_to_binary(localEncodedStream, totalEncodedSize));

    bitwiseOr(localEncodedStream, encodedSample, numberOfBytes, localEncodedStream);

    ulong partitianIndex = (dataIndex*MaximumEncodedBytes);

    //printf("partitianIndex=%u dataIndex=%u localEncodedStream[0]=0x%4x numberOfBytes=%d\n", partitianIndex, dataIndex, localEncodedStream[0], numberOfBytes );

    memcpy(&d_EncodedBlocks[partitianIndex], localEncodedStream, numberOfBytes);


//    d_EncodedBlocks[partitianIndex] = localEncodedStream[0];
//    d_EncodedBlocks[partitianIndex+1] =  localEncodedStream[1];
//    d_EncodedBlocks[partitianIndex+2] =  localEncodedStream[2];


//    for(int i=0; i<64; i++)
//    {
//    	//encodedDataPtr[partitianIndex+i] = localEncodedStream[i];
//    	d_EncodedBlocks[partitianIndex+i] = localEncodedStream[i];
//    }

 //   code_len = totalEncodedSize;

//	return code_len;
}


__device__ void zeroBlockEncoding(ushort* inputSamples, ulong dataIndex, RiceAlgorithm::CodingSelection* selection,
		                          unsigned char* d_EncodedBlocks, unsigned int totalEncodedSize, size_t* encodedSizeList)
{
	// Returning immediately if selection not within range
	// helps prevent thread divergence in warp
	if(*selection != RiceAlgorithm::ZeroBlockOpt)
	{
		return;
	}


    //*** TODO: Right now -- does not seem to be applicable to test image ***//


}

__device__ void secondExtensionEncoding(ushort* inputSamples, ulong dataIndex, RiceAlgorithm::CodingSelection* selection,
		                                unsigned char* d_EncodedBlocks, unsigned int totalEncodedSize, size_t* encodedSizeList)
{
	// Returning immediately if selection not within range
	// helps prevent thread divergence in warp
	if(*selection != RiceAlgorithm::SecondExtensionOpt)
	{
		return;
	}


    //*** TODO: Right now -- does not seem to be applicable to test image ***//
    // When it does may need to reassess how encoded samples are read back out before
    // sending encodedStream
    unsigned int secondExtentionOption;
    size_t byteLocation(0);

    // This will make the entire encoding 4 bits too long :TODO: Fix this
    d_EncodedBlocks[byteLocation] = RiceAlgorithm::SecondExtensionOpt + 1;


    int i = 0;
    for(i = 0; i < 32; i+=2)
    {
        secondExtentionOption = (((unsigned int)inputSamples[i] + inputSamples[i + 1])*((unsigned int)inputSamples[i] + inputSamples[i + 1] + 1))/2 + inputSamples[i + 1];

        memcpy(&d_EncodedBlocks[byteLocation], &secondExtentionOption, sizeof(secondExtentionOption));

        byteLocation += sizeof(secondExtentionOption);
    }

}

/**
 * CUDA kernel that identifies the winning encoding scheme for each block
 */
__global__ void encodingKernel(ushort* inputSamples, unsigned char* d_EncodedBlocks, unsigned int* d_EncodedBlockSizes)
{
	// Operate on all samples for a given block together

	ulong dataIndex(0);

	// http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
	ulong blockId = blockIdx.x + blockIdx.y * gridDim.x;
	ulong threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

//***************************************
//if(threadId > 4) return; // DEBUGGING 196608 max
//***************************************

	dataIndex = threadId;

//	printf("threadIdx.x=%d threadIdx.y=%d threadIdx.z=%d blockIdx.x=%d blockIdx.y=%d blockIdx.z=%d blockIdx.y dataIndex=%d,  threadId=%d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, (196607-dataIndex), threadId);


	RiceAlgorithm::CodingSelection selection;
	unsigned int encodedLength(0);
	unsigned int winningEncodedLength = -1;
	RiceAlgorithm::CodingSelection winningSelection;


    unsigned int code_len = (unsigned int)-1;
    size_t encodedSizeList[32];
    unsigned int totalEncodedSize(0);

    //===============================================================================================
    totalEncodedSize = getWinningEncodedLength(inputSamples, dataIndex, &selection, encodedSizeList);

//    switch (selection)
//    {
//		case RiceAlgorithm::K0:
//		case RiceAlgorithm::K1:
//		case RiceAlgorithm::K2:
//		case RiceAlgorithm::K3:
//		case RiceAlgorithm::K4:
//		case RiceAlgorithm::K5:
//		case RiceAlgorithm::K6:
//		case RiceAlgorithm::K7:
//		case RiceAlgorithm::K8:
//		case RiceAlgorithm::K9:
//		case RiceAlgorithm::K10:
//		case RiceAlgorithm::K11:
//		case RiceAlgorithm::K12:
//		case RiceAlgorithm::K13:
//		case RiceAlgorithm::K14:
//			// Apply SplitSequence encoding  ===> Result is performance slow down (~0.5 sec).
//            // Apparent that the switch statement is causing Thread Divergence
//			splitSequenceEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, totalEncodedSize, encodedSizeList);
//			break;
//
//		case RiceAlgorithm::ZeroBlockOpt:
//
//			break;
//
//		case RiceAlgorithm::SecondExtensionOpt:
//
//			break;
//
//		case RiceAlgorithm::NoCompressionOpt:
//
//			break;
//
//
//    }

    // Call will exit immediately return if Selection is out of range ==> prevents thread Divergence
	splitSequenceEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, totalEncodedSize, encodedSizeList);  // index by the block size or 32

	zeroBlockEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, totalEncodedSize, encodedSizeList);  // index by the block size or 32

	secondExtensionEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, totalEncodedSize, encodedSizeList);  // index by the block size or 32

	//:TODO: need No Comp Opt

    //===============================================================================================

	// Apply SplitSequence encoding
	//encodedLength = splitSequenceEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks);  // index by the block size or 32


	// Keep the encoded length for later
	d_EncodedBlockSizes[dataIndex] = encodedLength;

	// Find the winning encoding for all encoding types
    // This basically determines the winner
//    if (encodedLength < winningEncodedLength)
//    {
//        //*this = *(*iteration);
//        winningEncodedLength = encodedLength;
//        winningSelection = selection;
//
//        //encodedSize = (*iteration)->getEncodedBlockSize();
//    }

//	if(dataIndex <= 2)
//	{
//	    printf("Line #430, (encodedLength=%d) dataIndex=%d d_EncodedBlocks=%s\n",
//	    		encodedLength, dataIndex, byte_to_binary(&d_EncodedBlocks[dataIndex*64*2], encodedLength));
//	}
    //*************************************************************
    // Once here, synchronization among all threads should happen
    // Note that this is only applicable, for threads within a given
    // block. But we do not want to return until all are available.
    // Need to sync on the host.
    //*************************************************************
    __syncthreads();
}



#endif /* SENSORKERNELS_H_ */
