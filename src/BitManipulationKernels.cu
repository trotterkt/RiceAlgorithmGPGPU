/*
 * BitManipulationKernels.cpp
 *
 *  Created by: Keir Trotter
 *  California State University, Fullerton
 *  MSE, CPSC 597, Graduate Project
 *
 *  Copyright 2016 Keir Trotter
 */

#include <BitManipulationKernels.h>

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
__device__ __host__ void shiftRight(unsigned char* array, unsigned int bitSize, unsigned int arrayBitShift)
{
    unsigned int numberOfBytes(bitSize/RiceAlgorithm::BitsPerByte);

    if(bitSize % RiceAlgorithm::BitsPerByte)
    {
        numberOfBytes++;
    }

    // Decide where in the copy the new bytes will go
    //unsigned char* arrayCopy = new unsigned char[numberOfBytes];
    // Not allocating from global memory is significantly faster
    const int MaximumByteArray(80);
    unsigned char arrayCopy[MaximumByteArray] = {0};

    // Shift from bit to bit, and byte to byte
    unsigned int byteShift = arrayBitShift / RiceAlgorithm::BitsPerByte;
    unsigned int bitShift = arrayBitShift % RiceAlgorithm::BitsPerByte;

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

        previousBits = (array[byteIndex] & mask) << (RiceAlgorithm::BitsPerByte - bitShift);
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


__device__ __host__ void shiftLeft(unsigned char* array, unsigned int bitSize, unsigned int arrayBitShift)
{
    unsigned int numberOfBytes(bitSize/RiceAlgorithm::BitsPerByte);

    if(bitSize % RiceAlgorithm::BitsPerByte)
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
    unsigned int byteShift = arrayBitShift / RiceAlgorithm::BitsPerByte;
    unsigned int bitShift = arrayBitShift % RiceAlgorithm::BitsPerByte;

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

        previousBits = (array[byteIndex+1] & (mask << (RiceAlgorithm::BitsPerByte - bitShift)));
        previousBits >>= (RiceAlgorithm::BitsPerByte - bitShift);

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
    //unsigned int totalEncodedSize(0);
    unsigned int totalEncodedSize_NoComp(0);
    unsigned int totalEncodedSize_Split(0);

    unsigned int code_len = (unsigned int)-1;
    int i = 0, k = 0;
    int k_limit = 14;


    // No Compression check is first
    code_len = (32 * sizeof(ushort) * BitsInByte) + int(RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);
    *selection = RiceAlgorithm::NoCompressionOpt;
    totalEncodedSize_NoComp = code_len;


    // Then split sequence
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

    // Get the total encoded size first
    for(int index = dataIndex*32; index < (dataIndex*32)+32; index++)
    {
        size_t encodedSize = (inputSamples[index] >> *selection) + 1;
        encodedSizeList[index-(dataIndex*32)] = encodedSize; // Store Fundamental Sequence values
        totalEncodedSize_Split += int(encodedSize);

    }

    // Include length for split samples
    totalEncodedSize_Split += (32 * *selection);


    // include space for the  code option
    totalEncodedSize_Split += int(RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);


    if(totalEncodedSize_Split < totalEncodedSize_NoComp)
    {
    	code_len = totalEncodedSize_Split;

    }
    else
    {
        code_len = totalEncodedSize_NoComp;
    }


    return code_len;
}

__device__ unsigned int getSplitValueLocations(ulong dataIndex, RiceAlgorithm::CodingSelection selection, unsigned char* d_EncodedBlocks, ushort* d_EncodedBlockSizes, ushort splitValue[32])
{
	// Account for encoded value not being on a byte boundary
    const unsigned int CopySize(32 * sizeof(ushort) + 1); // Encoded data will be no larger than this

	unsigned char encodedDataAnotherCopy[CopySize] = {0};

    //Packet size adjustment
    float actualPacketBytes = ceil(float(d_EncodedBlockSizes[dataIndex])/
    		                       float(BitsInByte));

	size_t roundedBytes = actualPacketBytes;

	memcpy(encodedDataAnotherCopy, &d_EncodedBlocks[dataIndex*MaximumEncodedBytes], roundedBytes);
    shiftLeft(encodedDataAnotherCopy, (roundedBytes)*BitsInByte, // get rid of the selection id
              RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);


	// Assuming k-split type -
	// - count bits until 32-ones have been counted
	int encodeCount(0);

	int encodedLength(0);

	int splitCount(1);     // count within byte
	int index(0);          // location in split array
	int copyIndex(0);      // current byte


	while (encodeCount < 32)
	{
		for(int bitIndex=(BitsInByte-1); bitIndex>=0; bitIndex--)
		{
			if(encodeCount >= 32)
			{
				break;
			}

			encodedLength++;

			// Capture the encoded value
			//=====================================================
			// Count the bit if its '1'
			if ((encodedDataAnotherCopy[copyIndex] >> bitIndex) & 1)
			{
				encodeCount++;

				splitValue[index] = splitCount;

				#ifdef DEBUG
					//printf("dataIndex%lu ===> splitValue[%d]=%u\n", dataIndex, index, splitValue[index]);
					//printf("dataIndex%lu ===> encodeCount=%d, splitCount=%d, bitIndex=%d encodedLength=%d copyIndex=%d\n", dataIndex, encodeCount, splitCount, bitIndex, encodedLength, copyIndex);
				#endif

				index++;
				splitCount = 0;
			}

			splitCount++;

		}

		copyIndex++;

	}


	encodedLength += RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp;

	return encodedLength;
}


//NOTE: CUDA does not support passing reference to kernel argument
__device__ void splitSequenceEncoding(ushort* inputSamples, ulong dataIndex, RiceAlgorithm::CodingSelection* selection,
		                              unsigned char* d_EncodedBlocks, unsigned int* totalEncodedSize, size_t* encodedSizeList)
{
	// Returning immediately if selection not within range
	// helps prevent thread divergence in warp
	if(*selection > RiceAlgorithm::K13)
	{
		return;
	}

	// Apply SplitSequence encoding
	//=========================================================================================================

	// Not allocating from global memory is significantly faster
    unsigned char localEncodedStream[MaximumEncodedBytes];
    //=========================================================================================================

    memset(localEncodedStream, 0, MaximumEncodedBytes);


    // assign each encoded sample and shift by the next one
    // at the end of the loop, we will assign the last one
    // unsigned char* localEncodedStream(0);
    for(int index = 31; index >= 0; index--)
    {
    	localEncodedStream[0] |= 0x1;

    	int shift = encodedSizeList[index];

        shiftRight(localEncodedStream, *totalEncodedSize, shift);
    }



	// see Lossless Data Compression, Blue Book, sec 5.1.2
    // place the code encoding selection
    unsigned char selectionEncoding[MaximumEncodedBytes] = {0};
    selectionEncoding[0] = *selection + 1;
    bitwiseOr(localEncodedStream, selectionEncoding, MaximumEncodedBytes, localEncodedStream);

    //=========================================================================================================


    //=========================================================================================================
    // after the zero sequence number that was split off, then we add that value to the stream
    // for each of the samples
    unsigned short mask = powf(2, *selection) - 1;


    unsigned char encodedSample[MaximumEncodedBytes] = {0};
    unsigned char individualEncodedSample[MaximumEncodedBytes];



    for(int index = 0; index < 32; index++)
    {
        unsigned short maskedSample = inputSamples[index+dataIndex*32] & mask;
        unsigned char byteConvert[2] = {((maskedSample&0xff00)>>8), (maskedSample&0xff)}; //:KLUDGE: need to change the number into
                                                                                          // a byte form for printing only -- endian issue?

        memset(individualEncodedSample, 0, sizeof(individualEncodedSample));

    	//***************************************************
        // This shift aligns the encoding at the proper relative position in the array
        memcpy(individualEncodedSample, byteConvert, sizeof(byteConvert));

        shiftRight(individualEncodedSample, *totalEncodedSize, ((*selection * index)));
        shiftLeft(individualEncodedSample, *totalEncodedSize,  (BitsInByte*sizeof(byteConvert)) - *selection);


        // Finally merge the individual sample into this segment of the encoded stream
        bitwiseOr(encodedSample, individualEncodedSample, MaximumEncodedBytes, encodedSample);
    }

    //=========================================================================================================

    // determine number of bytes
    unsigned int numberOfBytes(*totalEncodedSize/RiceAlgorithm::BitsPerByte);
    if(*totalEncodedSize%RiceAlgorithm::BitsPerByte)
    {
    	numberOfBytes++;
    }

    // :TODO: Unclear why offset still exists
    shiftLeft(localEncodedStream, numberOfBytes*BitsInByte, RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);

    shiftRight(encodedSample, *totalEncodedSize, (*totalEncodedSize - (32 * *selection)));


    bitwiseOr(localEncodedStream, encodedSample, numberOfBytes, localEncodedStream);

    ulong partitianIndex = (dataIndex*MaximumEncodedBytes);


    memcpy(&d_EncodedBlocks[partitianIndex], localEncodedStream, numberOfBytes);

}

__device__ void splitSequenceDecoding(ulong dataIndex, unsigned char* d_EncodedBlocks, ushort* d_EncodedBlockSizes, ushort* d_PreProcessedImageData)
{
	unsigned char selectionBuffer = (d_EncodedBlocks[dataIndex*MaximumEncodedBytes] >> 4);

	RiceAlgorithm::CodingSelection selection = RiceAlgorithm::CodingSelection(int(selectionBuffer));

	// Returning immediately if selection not within range
	// helps prevent thread divergence in warp
	if(selection > RiceAlgorithm::K13)
	{
		return;
	}

	ushort splitValue[32] = {0};
	unsigned int bitLocation = getSplitValueLocations(dataIndex, selection, d_EncodedBlocks, d_EncodedBlockSizes, splitValue);


    // Make a new array for the 32 split values
	unsigned char encodedDataCopy[MaximumEncodedBytes] = {0};
	memcpy(encodedDataCopy, &d_EncodedBlocks[dataIndex*MaximumEncodedBytes], MaximumEncodedBytes);

    // Combine the individual values per the split sequence method
    // and save in the preprocessed array
    size_t bufferSize(MaximumEncodedBytes * BitsInByte);

	shiftLeft(encodedDataCopy, bufferSize, bitLocation);


	for(int index=0; index<32; index++)
	{
		ushort value(0);

		memcpy(&value, encodedDataCopy, sizeof(ushort));

        unsigned char byteConvert[2] = {((value&0xff00)>>8), (value&0xff)}; // since host function bigEndianVersusLittleEndian()
                                                                            // is not available
        memcpy(&value, byteConvert, sizeof(ushort));

        value >>= (sizeof(ushort) * BitsInByte - (selection - 1));

        d_PreProcessedImageData[index + dataIndex*32] = ((splitValue[index]-1) << (selection - 1)) | value;
		shiftLeft(encodedDataCopy, bufferSize, (selection - 1));
	}

}

__device__ void noCompEncoding(ushort* inputSamples, ulong dataIndex, RiceAlgorithm::CodingSelection* selection,
		                          unsigned char* d_EncodedBlocks, unsigned int* totalEncodedSize, size_t* encodedSizeList)
{
//	// Returning immediately if selection not within range
//	// helps prevent thread divergence in warp
//	if(*selection != RiceAlgorithm::NoCompressionOpt)
//	{
//		return;
//	}

	//if(threadId > 4) return; // DEBUGGING 196608 max


	// Apply No Compression encoding
	//=========================================================================================================

	// Not allocating from global memory is significantly faster
    unsigned char localEncodedStream[MaximumEncodedBytes] = {0};
    //=========================================================================================================


	// see Lossless Data Compression, Blue Book, sec 5.1.2
    // place the code encoding selection
    unsigned char selectionEncoding[MaximumEncodedBytes] = {0};
    selectionEncoding[0] = RiceAlgorithm::NoCompressionOpt;

    bitwiseOr(localEncodedStream, selectionEncoding, MaximumEncodedBytes, localEncodedStream);

    shiftRight(localEncodedStream, MaximumEncodedBytes*BitsInByte,  (MaximumEncodedBytes*BitsInByte)-RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp - 4);
    shiftLeft(localEncodedStream, MaximumEncodedBytes*BitsInByte, BitsInByte*sizeof(ushort)); // make room for next sample. insertion is in reverse


    //=========================================================================================================

    unsigned char individualEncodedSample[MaximumEncodedBytes] = {0};

	*totalEncodedSize = 32 * sizeof(ushort) * BitsInByte + RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp;

    for(int index = 0; index < 32; index++)
    {
        unsigned short sample = inputSamples[index+dataIndex*32];
        unsigned char byteConvert[2] = {((sample&0xff00)>>8), (sample&0xff)}; //:KLUDGE: need to change the number into
                                                                              // a byte form for printing only -- endian issue?

        memset(individualEncodedSample, 0, MaximumEncodedBytes);

        memcpy(individualEncodedSample, byteConvert, sizeof(sample));

        shiftRight(individualEncodedSample, MaximumEncodedBytes*BitsInByte,  (MaximumEncodedBytes*BitsInByte)-(sizeof(sample)*BitsInByte));

        bitwiseOr(localEncodedStream, individualEncodedSample, MaximumEncodedBytes, localEncodedStream);

        // prepare space for each sample
        if(index != 31)
        {
        	shiftLeft(localEncodedStream, MaximumEncodedBytes*BitsInByte, BitsInByte*sizeof(ushort)); // make room for next sample. insertion is in reverse
        }
    }

    // determine number of bytes
    unsigned int numberOfBytes(*totalEncodedSize/RiceAlgorithm::BitsPerByte);
    if(*totalEncodedSize%RiceAlgorithm::BitsPerByte)
    {
    	numberOfBytes++;
    }

    // :TODO: Unclear why offset still exists
    shiftLeft(localEncodedStream, numberOfBytes*BitsInByte, RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp);

    ulong partitianIndex = (dataIndex*MaximumEncodedBytes);

    memcpy(&d_EncodedBlocks[partitianIndex], localEncodedStream, numberOfBytes);
}

__device__ void noCompDecoding(ulong dataIndex, unsigned char* d_EncodedBlocks, ushort* d_EncodedBlockSizes, ushort* d_PreProcessedImageData)
{
	unsigned char selectionBuffer = (d_EncodedBlocks[dataIndex*MaximumEncodedBytes] >> 4);

	RiceAlgorithm::CodingSelection selection = RiceAlgorithm::CodingSelection(int(selectionBuffer));

	// Returning immediately if selection not within range
	// helps prevent thread divergence in warp
	if(selection != RiceAlgorithm::NoCompressionOpt)
	{
		return;
	}


    // Make a new array for the 32 split values
	unsigned char encodedDataCopy[MaximumEncodedBytes] = {0};
	memcpy(encodedDataCopy, &d_EncodedBlocks[dataIndex*MaximumEncodedBytes], MaximumEncodedBytes);

    size_t bufferSize(MaximumEncodedBytes * BitsInByte);

	shiftLeft(encodedDataCopy, bufferSize, RiceAlgorithm::CodeOptionBitFieldFundamentalOrNoComp); // get rid of the selection id

	for(int index=0; index<32; index++)
	{
		ushort value(0);

		memcpy(&value, encodedDataCopy, sizeof(ushort));

        unsigned char byteConvert[2] = {((value&0xff00)>>8), (value&0xff)}; // since host function bigEndianVersusLittleEndian()
                                                                            // is not available
        memcpy(&value, byteConvert, sizeof(ushort));

        d_PreProcessedImageData[index + dataIndex*32] = value;
		shiftLeft(encodedDataCopy, bufferSize, sizeof(ushort)*BitsInByte);
	}

}

__device__ void zeroBlockEncoding(ushort* inputSamples, ulong dataIndex, RiceAlgorithm::CodingSelection* selection,
		                          unsigned char* d_EncodedBlocks, unsigned int* totalEncodedSize, size_t* encodedSizeList)
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
		                                unsigned char* d_EncodedBlocks, unsigned int* totalEncodedSize, size_t* encodedSizeList)
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
__global__ void encodingKernel(ushort* inputSamples, unsigned char* d_EncodedBlocks, ushort* d_EncodedBlockSizes)
{
	// Operate on all samples for a given block together

	ulong dataIndex(0);

	// http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
	ulong blockId = blockIdx.x + blockIdx.y * gridDim.x;
	ulong threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// Provides ability to debug individual threads
	//***************************************
	//if(threadId != 196509 && threadId != 196508  && threadId != 196510 ) return; // DEBUGGING 196608 max
	//if(threadId != 196509) return; // DEBUGGING 196608 max
	//if(threadId != 196507 && threadId != 196508 && threadId != 196509) return; // DEBUGGING 196608 max
	//if(threadId >= 1) return;
	//if((threadId != 16562) && (threadId != 16563) && (threadId != 16564)) return; // DEBUGGING 196608 max
	//***************************************
	dataIndex = threadId;


	RiceAlgorithm::CodingSelection selection;
	unsigned int encodedLength(0);
	unsigned int winningEncodedLength = -1;
	RiceAlgorithm::CodingSelection winningSelection;


    unsigned int code_len = (unsigned int)-1;
    size_t encodedSizeList[32];
    unsigned int totalEncodedSize(0);

    //===============================================================================================
    totalEncodedSize = getWinningEncodedLength(inputSamples, dataIndex, &selection, encodedSizeList);

    // Don't do this as in the sequential versions -
    // prone to thread divergence
    //***************************************************
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
    //***************************************************

    //:TODO: THis seems like a bad thing as far as branch divergence,
    // but for now appears to be the only way to produce the correct
    // encoding that is consistent reading back on decompression
    if(totalEncodedSize < 516)
    {
    	// Call will exit immediately return if Selection is out of range ==> prevents thread Divergence
    	splitSequenceEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, &totalEncodedSize, encodedSizeList);  // index by the block size or 32
    }
    else
    {
    	noCompEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, &totalEncodedSize, encodedSizeList);  // index by the block size or 32
    }
	//zeroBlockEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, &totalEncodedSize, encodedSizeList);  // index by the block size or 32

	//secondExtensionEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, &totalEncodedSize, encodedSizeList);  // index by the block size or 32
    //===============================================================================================

	// Apply SplitSequence encoding
	//encodedLength = splitSequenceEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks);  // index by the block size or 32


	// Keep the encoded length for later
	d_EncodedBlockSizes[dataIndex] = totalEncodedSize;


    //*************************************************************
    // Once here, synchronization among all threads should happen
    // Note that this is only applicable, for threads within a given
    // block. But we do not want to return until all are available.
    // Need to sync on the host.
    //*************************************************************
    __syncthreads();
}


/**
 * CUDA kernel that identifies the winning encoding scheme for each block
 */
__global__ void decodingKernel(ushort* d_PreProcessedImageData, unsigned char* d_EncodedBlocks, ushort* d_EncodedBlockSizes)
{
	// Operate on all samples for a given block together

	ulong dataIndex(0);

	// http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
	ulong blockId = blockIdx.x + blockIdx.y * gridDim.x;
	ulong threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	//***************************************
	//if(threadId != 16563) return; // DEBUGGING 196608 max
	//if(threadId >= 20) return; // DEBUGGING 196608 max
	//***************************************

	dataIndex = threadId;



	RiceAlgorithm::CodingSelection selection;
	unsigned int encodedLength(0);
	unsigned int winningEncodedLength = -1;
	RiceAlgorithm::CodingSelection winningSelection;


    unsigned int code_len = (unsigned int)-1;
    size_t encodedSizeList[32];
    unsigned int totalEncodedSize(0);

    //===============================================================================================
//    totalEncodedSize = getWinningEncodedLength(inputSamples, dataIndex, &selection, encodedSizeList);

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
    splitSequenceDecoding(dataIndex, d_EncodedBlocks, d_EncodedBlockSizes, d_PreProcessedImageData);

    noCompDecoding(dataIndex, d_EncodedBlocks, d_EncodedBlockSizes, d_PreProcessedImageData);



//	splitSequenceEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, &totalEncodedSize, encodedSizeList);  // index by the block size or 32
//
//	zeroBlockEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, &totalEncodedSize, encodedSizeList);  // index by the block size or 32
//
//	secondExtensionEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks, &totalEncodedSize, encodedSizeList);  // index by the block size or 32

	//:TODO: need No Comp Opt

    //===============================================================================================

	// Apply SplitSequence encoding
	//encodedLength = splitSequenceEncoding(inputSamples, dataIndex, &selection, d_EncodedBlocks);  // index by the block size or 32


	// Keep the encoded length for later
	d_EncodedBlockSizes[dataIndex] = totalEncodedSize;

    //*************************************************************
    // Once here, synchronization among all threads should happen
    // Note that this is only applicable, for threads within a given
    // block. But we do not want to return until all are available.
    // Need to sync on the host.
    //*************************************************************
    __syncthreads();
}
