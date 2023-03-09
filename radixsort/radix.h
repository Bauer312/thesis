#ifndef radix_h
#define radix_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
	#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
	#include <arm_neon.h>
#endif

typedef struct radixStruct {
	int32_t numElements;
	int32_t radixSize;
	uint32_t mask;
	uint32_t numSlots;
	uint32_t numRadixValues;

	uint32_t *primaryArr;
	uint32_t *secondaryArr;
} radixStruct;

typedef struct histogramRadix {
	uint32_t radixValue;
	uint32_t numElements;
	uint32_t startIndex;
	uint32_t currentIndex;
} histogramRadix;

typedef struct threadStruct {
	histogramRadix **radiiToProcess;
	uint32_t radixCount;
	uint32_t totalElements;
	uint32_t threadNumber;
	uint32_t slotNumber;
	pthread_t thread;
	radixStruct *rs;
	void (*histFunc)(radixStruct*, histogramRadix*, int32_t, int32_t, int32_t);
	void (*sortFunc)(radixStruct*, histogramRadix*, int32_t, int32_t, int32_t);
} threadStruct;

#define STYLE_BASIC 1
#define STYLE_SIMD 2

int32_t initRadixStruct(radixStruct *rs, int32_t numElements, int32_t radixSize);
void freeRadixStruct(radixStruct *rs);
void populateRadixStruct(radixStruct *rs);

void execute(radixStruct *rs, int32_t numThreads, int32_t style);
void populateHistogram(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end);
void populateHistogramSIMD(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end);
void sortSlot(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end);
void sortSlotSIMD(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end);

void printRadix(radixStruct *rs, int32_t whichSlot, int32_t swapped);
int testSorted(radixStruct *rs);

int availableCores();

void qs(unsigned int *arr, int32_t low, int32_t high);
int32_t qsp(unsigned int *arr, int32_t low, int32_t high);
void executeQS(radixStruct *rs);
#endif
