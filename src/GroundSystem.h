/*
 * GroundSystem.h
 *
 *  Created by: Keir Trotter
 *  California State University, Fullerton
 *  MSE, CPSC 597, Graduate Project
 *
 *  Copyright 2016 Keir Trotter
 */

#ifndef GROUNDSYSTEM_H_
#define GROUNDSYSTEM_H_

#include <ImagePersistence.h>
#include <AdaptiveEntropyEncoder.h>
#include <BitManipulationKernels.h>

namespace RiceAlgorithm
{

class GroundSystem
{
	public:
		GroundSystem(ImagePersistence* image);
		virtual ~GroundSystem();

		double process(ushort *d_PreProcessedImageData, unsigned char* d_EncodedBlocks, ushort* d_EncodedBlockSizes, ushort* referenceResiduals = 0);

		// For validation
		ushort* getSamples(){ return myRawSamples; }

	private:
		void readHeader();
		CompressedHeader myHeader;

		RiceAlgorithm::ImagePersistence* mySource;

		ushort* myRawSamples;
};

} /* namespace RiceAlgorithm */

#endif /* GROUNDSYSTEM_H_ */
