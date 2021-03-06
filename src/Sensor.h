/*
 * Sensor.h
 *
 *  Created by: Keir Trotter
 *  California State University, Fullerton
 *  MSE, CPSC 597, Graduate Project
 *
 *  Copyright 2016 Keir Trotter
 */


#ifndef SENSOR_H_
#define SENSOR_H_

#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <thrust/device_vector.h>

#include <sstream>
#include <Predictor.h>
#include <AdaptiveEntropyEncoder.h>
#include <SplitSequence.h>
#include <SecondExtensionOption.h>
#include <ZeroBlockOption.h>
#include <ImagePersistence.h>
#include <boost/dynamic_bitset.hpp>

const double LandsatDownlinkRate(384);

const ushort MaximumEncodedBlockSize(RiceAlgorithm::BlockSize);

// These parameters are what is utilized for LandSat
const int Rows(1024);
const int Columns(1024);
const int Bands(6);
const int BlockSize(32);




// Note that if member types are not defined as being of similar size
// there can be an alignment problem. See Annotated  C++ Ref Manual,
// Sec 5.3.2. This will not be an issue for this structure, since
// I will not be writing in out directly as a whole.

struct CompressedHeader
{
    char userData;
    short xDimension;
    short yDimension;
    short zDimension;

    char signSampDynRangeBsq1;   // sample type, reserved, dyn range, bsq(1)
    short bsq;
    short wordSizEncodeMethod; // reserved, out word size, encoding method,
                                 // reserved
    char predictBandMode;        // user input predictor band,
                                 // predictor full, reserved,
    char neighborRegSize;        // neighbor sum,
                                 //reserve, register size
    char predictWeightResInit;   // weight resolution, weight interval, initial weight,
                                 // final weight, reserved, initial weight table,
                                  // weight init resolution
    char predictWeightInitFinal;    // reserved, block size flag, restricted, ref interval
    char predictWeightTable;
    short blockSizeRefInterval;
};

class Sensor
{
	public:
		Sensor(RiceAlgorithm::ImagePersistence* image, unsigned int x, unsigned int y, unsigned int z);
		virtual ~Sensor();

		void process();

		 void operator=(RiceAlgorithm::AdaptiveEntropyEncoder& right){
			 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 memcpy(myEncodedBlock,
			 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	        right.getEncodedBlock(),
			 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	        right.getEncodedBlockSize());
			 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 myWinningEncodedLength = right.getEncodedBlockSize();
		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 }
		 bool operator>(RiceAlgorithm::AdaptiveEntropyEncoder& right){  return false; }

		 void setImageSource(RiceAlgorithm::ImagePersistence* image) {mySource = image; }

	private:


		 RiceAlgorithm::ImagePersistence* mySource;


	    ushort* mySamples;
	    unsigned int myEncodedBlock[MaximumEncodedBlockSize];

	    // append all encoded blocks before sending at once
	    boost::dynamic_bitset<> myFullEncodedStream;

	    void sendHeader();

        unsigned int myXDimension;
        unsigned int myYDimension;
        unsigned int myZDimension;

        RiceAlgorithm::Predictor myPreprocessor;
		//std::vector<class RiceAlgorithm::AdaptiveEntropyEncoder*> myEncoderList;
		//thrust::device_vector<class RiceAlgorithm::AdaptiveEntropyEncoder*> myEncoderList;

		unsigned int myWinningEncodedLength;

		//:TODO: These should instead be declared in the implementation file
		//RiceAlgorithm::AdaptiveEntropyEncoder* noComp;
		//RiceAlgorithm::SecondExtensionOption* secondExt;
		//RiceAlgorithm::ZeroBlockOption* zeroBlock;
		//RiceAlgorithm::SplitSequence* split; // this will become more specific

		void sendEncodedSamples(boost::dynamic_bitset<> &encodedStream, unsigned int encodedLength=0);


		template<typename T> void packCompressedData(T data, boost::dynamic_bitset<unsigned char> &packedData, ulong bitSize=sizeof(T)*RiceAlgorithm::BitsPerByte)
        {

		    size_t numberOfBytes = sizeof(data);

		    // whatever type it is, see it as a collection of bytes
		    char* ptrData = reinterpret_cast<char*>(&data);

		    size_t currentSize = packedData.size();

		    // Since the data is of uneven types, all are read in as single bytes
		    for(int index=0; index<numberOfBytes; index++)
		    {
		        packedData.append(ptrData[index]);
		    }

		    packedData.resize(currentSize+bitSize);
        }

        void writeCompressedData(boost::dynamic_bitset<unsigned char> &packedData, size_t bitSize=0, bool flag=false);

        bool getLastByte(unsigned char &lastByte);

        unsigned int myEncodedBitCount;

};

#endif /* SENSOR_H_ */
