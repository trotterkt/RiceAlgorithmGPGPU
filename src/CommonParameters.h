/*
 * CommonParameters.h
 *
 *  Created by: Keir Trotter
 *  California State University, Fullerton
 *  MSE, CPSC 597, Graduate Project
 *
 *  Copyright 2016 Keir Trotter
 */

#ifndef COMMONPARAMETERS_H_
#define COMMONPARAMETERS_H_


// These parameters are what is utilized for LandSat
const int Rows(1024);
const int Columns(1024);
const int Bands(6);
const int BlockSize(32);

const int BitsInByte(8);
const int MaximumEncodedBytes(32*sizeof(ushort) + 1);            // Observed maximum number of encoded bytes
const int NumberEncodedPackets(Rows*Columns*Bands/32);
const ulong MaximumEncodedMemory(MaximumEncodedBytes*NumberEncodedPackets);




#endif /* COMMONPARAMETERS_H_ */
