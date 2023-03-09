#ifndef blur_h
#define blur_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
	#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
	#include <arm_neon.h>
#endif

typedef struct blurStruct {
	int32_t convolutionDistance;
	int32_t imageSize;

	double standardDeviation;

	float *convolutionMatrix;
	int32_t *image;
	int32_t *newImage;
	int32_t *simdImage;
} blurStruct;

typedef struct threadStruct {
	int32_t start;
	int32_t end;
	pthread_t thread;
	blurStruct *bs;
	void (*blurFunc)(blurStruct*, int32_t, int32_t);
} threadStruct;

#define STYLE_BASIC 1
#define STYLE_SIMD 2

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

int matrixSize(double stdDev);
int initBlurStruct(blurStruct *bs, double stdDev, int32_t imageSize);
void populateBlurStruct(blurStruct *bs);
void freeBlurStruct(blurStruct *bs);

void blurImage(blurStruct *bs, int32_t startingRow, int32_t endingRow);
void blurImageSIMD(blurStruct *bs, int32_t startingRow, int32_t endingRow);
void blurPixel(blurStruct *bs, int32_t x, int32_t y);
void blurPixelSIMD(blurStruct *bs, int32_t x, int32_t y);

void execute(blurStruct *bs, int numThreads, int style);
void *invokeBlurWithThread(void *arguments);
int availableCores();

void printConvolutionMatrix(blurStruct *bs);

#endif