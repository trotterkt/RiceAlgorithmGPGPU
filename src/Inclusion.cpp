//============================================================================
// Name        : Hello.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>

using namespace std;

const unsigned short BitsInByte(8);

void shiftRight(unsigned char* array, unsigned int bitSize, unsigned int arrayBitShift)
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

}

unsigned int shiftLeft(unsigned char* array, unsigned int bitSize, unsigned int arrayBitShift)
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
        if((byteIndex - byteShift) < 0)
        {
            break;
        }

        previousBits = (array[byteIndex+1] & (mask << (BitsInByte - bitShift)));
        previousBits >>= (BitsInByte - bitShift);

        arrayCopy[byteIndex - byteShift] = (array[byteIndex]) << bitShift;

        if (byteIndex <= (numberOfBytes-1))
        {
            arrayCopy[byteIndex - byteShift] |= previousBits;
        }

    }

    memcpy(array, arrayCopy, numberOfBytes);
}

void bitwiseOr(unsigned char* byteFirst, unsigned char* byteSecond, unsigned int numberOfBytes, unsigned char* outByte)
{
    for(int i=0; i<numberOfBytes; i++)
    {
       outByte[i] =  byteFirst[i] | byteSecond[i];
    }
}

void bitwiseAnd(unsigned char* byteFirst, unsigned char* byteSecond, unsigned int numberOfBytes, unsigned char* outByte)
{
    for(int i=0; i<numberOfBytes; i++)
    {
       outByte[i] =  byteFirst[i] & byteSecond[i];
    }
}

const char *byte_to_binary(unsigned char* x, int numberOfBytes)
{
    const int MaximumBitLength(504);
    
    static char b[MaximumBitLength] = {0};
    
    b[0] = '\0';

    for(int byteIndex=0; byteIndex<numberOfBytes; byteIndex++)
    {
        int z;
        for (z = 0x80; z > 0; z >>= 1)
        {
            strcat(b, ((x[byteIndex] & z) == z) ? "1" : "0");
        }
    }
        
    return b;
}



int main()
{
    cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

    int* pInt = new int[50];




    unsigned char array[] = { 0xF0, 0xCC, 0xAA, 0xF0 };

    printf("%s\n", byte_to_binary(array, sizeof(array)));
    //printf("%s\n", byte_to_binary(15));
    //printf("%s\n", byte_to_binary(8));

    //shiftRight(array, sizeof(array), 1);


    // Additional encoding:
    // This basically works - to be included in GPGPU version.
    //===============================================================================
    int dataIndex(0);

    unsigned short inputSamples[] =
    {
         22015,
         4096,
         3071,
         2560,
         2048,
         512,
         511,
         512,
         512,
         512,
         511,
         2047,
         2047,
         3583,
         2047,
         1023,
         511,
         0,
         1023,
         4096,
         0,
         0,
         512,
         512,
         512,
         3584,
         1023,
         511,
         1536,
         512,
         511,
         0
    };

    unsigned int selection(10); // assume just K10 for now


    unsigned short mask = powf(2, selection) - 1;

    const unsigned int MaximumByteAdditionalArray(56); // 14*32/BitsInByte
    const unsigned int additionalEncodedSize(selection * 32 * BitsInByte);


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
        shiftLeft(individualEncodedSample, selection, ((sizeof(byteConvert)*BitsInByte)-selection));

        // This shift aligns the encoding at the proper relative position in the array
        shiftRight(individualEncodedSample, (selection*32), (selection*index));

        // Finally merge the individule sample into this segment of the encoded stream 
        bitwiseOr(encodedSample, individualEncodedSample, MaximumByteAdditionalArray, encodedSample);

    }


    return 0;
}
