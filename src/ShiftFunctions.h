/*
 * ShiftFunctions.h
 *
 *  Created by: Keir Trotter
 *  California State University, Fullerton
 *  MSE, CPSC 597, Graduate Project
 *
 *  Copyright 2016 Keir Trotter
 */

#ifndef SHIFTFUNCTIONS_H_
#define SHIFTFUNCTIONS_H_
#include <sys/types.h>


// Taken from the CUDA implementation
//***************************************************************
void shiftRight(unsigned char* array, unsigned int bitSize, unsigned int arrayBitShift);

void shiftLeft(unsigned char* array, unsigned int bitSize, unsigned int arrayBitShift);
//***************************************************************



#endif /* SHIFTFUNCTIONS_H_ */
