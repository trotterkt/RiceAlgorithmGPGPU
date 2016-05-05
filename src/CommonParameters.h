/*
 * CommonParameters.h
 *
 *  Created on: May 3, 2016
 *      Author: ktrotter
 */

#ifndef COMMONPARAMETERS_H_
#define COMMONPARAMETERS_H_


// These parameters are what is utilized for LandSat
const int Rows(1024);
const int Columns(1024);
const int Bands(6);
const int BlockSize(32);

const int BitsInByte(8);
//const int MaximumEncodedBytes(77);            // Observed maximum number of encoded bytes
const int MaximumEncodedBytes(32*sizeof(ushort) + 1);            // Observed maximum number of encoded bytes
const int NumberEncodedPackets(Rows*Columns*Bands/32);




#endif /* COMMONPARAMETERS_H_ */
