#include "blur.h"

int matrixSize(double stdDev) {
	return ceil(6.0 * stdDev);
}

int initBlurStruct(blurStruct *bs, double stdDev, int32_t imageSize) {
	if(bs == NULL) {
		return -1;
	}

  bs->standardDeviation = stdDev;
	bs->convolutionDistance = ceil(ceil(6.0 * stdDev) / 2);
	int convolutionElementSide = bs->convolutionDistance * 2 + 1;
	bs->convolutionMatrix = (float*)malloc(sizeof(float) * convolutionElementSide * convolutionElementSide);
	if(bs->convolutionMatrix) {
		int y = -1 * bs->convolutionDistance;
		int matrixOffsetY = 0;
		while(y <= bs->convolutionDistance) {
			int x = -1 * bs->convolutionDistance;
			int matrixOffsetX = 0;
			while(x <= bs->convolutionDistance) {
				bs->convolutionMatrix[matrixOffsetY + matrixOffsetX] = (1.0 / (2.0 * M_PI * pow(stdDev, 2.0)) * exp(-1.0 * ((pow(x, 2.0) + pow(y, 2.0)) / (2.0 * pow(stdDev, 2.0)))));
				//printf(" %10.8f ", bs->convolutionMatrix[matrixOffsetY + matrixOffsetX]);
				x++;
				matrixOffsetX++;
			}
			//printf("\n");
			y++;
			matrixOffsetY += convolutionElementSide;
		}
	} else {
		fprintf(stderr, "Unable to allocate the convolution matrix\n");
		return -1;
	}

	bs->imageSize = imageSize;
	bs->image = (int32_t*)malloc(sizeof(int32_t) * imageSize * imageSize);
	if(bs->image) {
		memset(bs->image, 0, sizeof(int32_t) * imageSize * imageSize);
	} else {
		fprintf(stderr, "Unable to allocate the image data\n");
		return -1;
	}
	bs->newImage = (int32_t*)malloc(sizeof(int32_t) * imageSize * imageSize);
	if(bs->newImage) {
		memset(bs->newImage, 0, sizeof(int32_t) * imageSize * imageSize);
	} else {
		fprintf(stderr, "Unable to allocate the blurred image data\n");
		return -1;
	}
	bs->simdImage = (int32_t*)malloc(sizeof(int32_t) * imageSize * imageSize);
	if(bs->simdImage) {
		memset(bs->simdImage, 0, sizeof(int32_t) * imageSize * imageSize);
	} else {
		fprintf(stderr, "Unable to allocate the blurred image data\n");
		return -1;
	}

	return 0;
}

void populateBlurStruct(blurStruct *bs) {
	if(bs) {
		time_t t;
		srand((unsigned) time(&t));

		for(int i = 0; i < bs->imageSize * bs->imageSize; i++) {
			bs->image[i] = rand() % (INT16_MAX / 4);
		}
	}
}

void freeBlurStruct(blurStruct *bs) {
	if(bs) {
		if(bs->convolutionMatrix) {
			free(bs->convolutionMatrix);
			bs->convolutionMatrix = NULL;
		}
		if(bs->image) {
			free(bs->image);
			bs->image = NULL;
		}
		if(bs->newImage) {
			free(bs->newImage);
			bs->newImage = NULL;
		}
		if(bs->simdImage) {
			free(bs->simdImage);
			bs->simdImage = NULL;
		}
	}
}

void blurPixel(blurStruct *bs, int32_t x, int32_t y) {
	int32_t imageXmin = x - bs->convolutionDistance;
	int32_t imageXmax = x + bs->convolutionDistance;
	int32_t imageYmin = y - bs->convolutionDistance;
	int32_t imageYmax = y + bs->convolutionDistance;
	int32_t cmXmin = 0;
	int32_t cmYmin = 0;

	size_t rowOffset;
	size_t cmRowOffset;
	int32_t elementCount = 0;
	double accumulator = 0.0;
	for(int32_t row = imageYmin, cmRow = cmYmin; row <= imageYmax; row++, cmRow++) {
		rowOffset = row * bs->imageSize;
		cmRowOffset = cmRow * (bs->convolutionDistance * 2 + 1);
		for(int32_t column = imageXmin, cmColumn = cmXmin; column <= imageXmax; column++, cmColumn++) {
			accumulator += bs->image[rowOffset + column] * bs->convolutionMatrix[cmRowOffset + cmColumn];
			elementCount++;
		}
	}
	bs->newImage[y * bs->imageSize + x] = ceil(accumulator / elementCount);
}

void blurPixelSIMD(blurStruct *bs, int32_t x, int32_t y) {
	int32_t imageXmin = x - bs->convolutionDistance;
	//int32_t imageXmax = x + bs->convolutionDistance;
	int32_t imageYmin = y - bs->convolutionDistance;
	int32_t imageYmax = y + bs->convolutionDistance;

	size_t rowOffset;
	size_t cmRowOffset;
	int32_t elementCount = 0;
	double accumulator = 0.0;

	#if defined(__x86_64__) || defined(_M_X64)
		__m256i imageValues;
	__m256 imgFloatValues;
		__m256 convoValues;
	__m256 temp;
	int32_t loadMaskArr[] = {1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 0};
		if(bs->convolutionDistance < 3) { loadMaskArr[5] = 0; loadMaskArr[6] = 0; }
	if(bs->convolutionDistance < 2) { loadMaskArr[3] = 0; loadMaskArr[4] = 0; }
	__m256i loadMask = _mm256_loadu_si256((__m256i*)&(loadMaskArr[0]));

	for(int32_t row = imageYmin, cmRow = 0; row <= imageYmax; row++, cmRow++) {
		rowOffset = row * bs->imageSize;
		cmRowOffset = cmRow * (bs->convolutionDistance * 2 + 1);

		//clusterValues = _mm256_loadu_si256((__m256i*)&(ds->clusters[cluster * ds->actualDimensions + i]));
		imageValues = _mm256_maskload_epi32(&(bs->image[rowOffset + imageXmin]), loadMask);
		imgFloatValues = _mm256_cvtepi32_ps(imageValues);
		convoValues = _mm256_maskload_ps(&(bs->convolutionMatrix[cmRowOffset]), loadMask);

		temp = _mm256_mul_ps(imgFloatValues, convoValues);

		// Sum all the elements together
		accumulator += (temp[0] + temp[1]) + (temp[2] + temp[3]) + (temp[4] + temp[5]) + (temp[6] + temp[7]);
		elementCount += bs->convolutionDistance * 2 + 1;
	}
	//fprintf(stdout, "[%d,%d] %f %d\n", x, y, accumulator, elementCount);
	bs->simdImage[y * bs->imageSize + x] = ceil(accumulator / elementCount); 
	#elif defined(__aarch64__) || defined(_M_ARM64)
	int32_t imageXmax = x + bs->convolutionDistance;
		int32x4_t imageValues;
		float32x4_t imgFloatVal;
		float32x4_t convoValues;
		float32x4_t temp;
		int32_t simdSize = 4;
		for(int32_t row = imageYmin, cmRow = 0; row <= imageYmax; row++, cmRow++) {
			int32_t idx = imageXmin;
			int32_t cIdx = 0;
			rowOffset = row * bs->imageSize;
			cmRowOffset = cmRow * (bs->convolutionDistance * 2 + 1);

			while(imageXmax - idx >= simdSize) {
				// Load image data
				imageValues = vld1q_s32(&(bs->image[rowOffset + idx]));
				// Convert to floating point
				imgFloatVal = vcvtq_f32_s32(imageValues);
				// Load convo data
				convoValues = vld1q_f32(&(bs->convolutionMatrix[cmRowOffset + cIdx]));
				// Get the dot product of the two vectors
				temp = vmulq_f32(imgFloatVal, convoValues);
				accumulator += vaddvq_f32(temp);

				idx += simdSize;
				cIdx += simdSize;
				elementCount += simdSize;
			}
			// Take care of whatever elements are left over
			for(; idx <= imageXmax; idx++, cIdx++) {
				accumulator += bs->image[rowOffset + idx] * bs->convolutionMatrix[cmRowOffset + cIdx];
				elementCount++;
			}
		}
		// Store the mean as the result
		bs->simdImage[y * bs->imageSize + x] = ceil(accumulator / elementCount);
	#endif
}

void blurImage(blurStruct *bs, int32_t startingRow, int32_t endingRow) {
	if(startingRow < bs->convolutionDistance) { startingRow = bs->convolutionDistance; }
	if(endingRow > bs->imageSize - bs->convolutionDistance) { endingRow = bs->imageSize - bs->convolutionDistance; }

	//fprintf(stdout, "Start %d End %d (%d)\n", startingRow, endingRow, bs->convolutionDistance);
	for(int32_t row = startingRow; row < endingRow; row++) {
		for(int32_t column = bs->convolutionDistance; column < bs->imageSize - bs->convolutionDistance; column++) {
			blurPixel(bs, column, row);
		}
	}
}

void blurImageSIMD(blurStruct *bs, int32_t startingRow, int32_t endingRow) {
	if(startingRow < bs->convolutionDistance) { startingRow = bs->convolutionDistance; }
	if(endingRow > bs->imageSize - bs->convolutionDistance) { endingRow = bs->imageSize - bs->convolutionDistance; }

	for(int32_t row = startingRow; row < endingRow; row++) {
		for(int32_t column = bs->convolutionDistance; column < bs->imageSize - bs->convolutionDistance; column++) {
			blurPixelSIMD(bs, column, row);
		}
	}
}

void execute(blurStruct *bs, int numThreads, int style) {
	struct timespec startTime;
	struct timespec endTime;
	timespec_get(&startTime, TIME_UTC);

	threadStruct *threadArr = (threadStruct*)malloc(sizeof(threadStruct) * numThreads);
	if(threadArr == NULL) {
		fprintf(stderr, "Unable to allocate memory for %d thread structures\n", numThreads);
	}

	int rowsPerThread = (bs->imageSize - bs->convolutionDistance * 2) / numThreads;
	int leftovers = (bs->imageSize - bs->convolutionDistance * 2) % numThreads;

	threadArr[0].start = bs->convolutionDistance;
	threadArr[0].end = threadArr[0].start + rowsPerThread + leftovers;
	threadArr[0].bs = bs;
	if(style == STYLE_BASIC) {
		threadArr[0].blurFunc = &blurImage;
	} else if(style == STYLE_SIMD) {
		threadArr[0].blurFunc = &blurImageSIMD;
	}

	for(int i = 0; i < numThreads; i++) {
		if(i < numThreads - 1) {
			threadArr[i+1].start = threadArr[i].end;
			threadArr[i+1].end = threadArr[i].end + rowsPerThread;
			threadArr[i+1].bs = bs;
			if(style == STYLE_BASIC) {
				threadArr[i+1].blurFunc = &blurImage;
			} else if(style == STYLE_SIMD) {
				threadArr[i+1].blurFunc = &blurImageSIMD;
			}
		}
		if(pthread_create(&(threadArr[i].thread), NULL, invokeBlurWithThread, &threadArr[i]) != 0) {
			fprintf(stderr, "Unable to start thread %d\n", i);
			exit(EXIT_FAILURE);
		}
	}

	// Wait for all threads to finish
	for (int i = 0; i < numThreads; i++) {
		if(pthread_join(threadArr[i].thread, NULL) != 0) {
			fprintf(stderr, "Unable to properly join thread %d\n", i);
		}
	}

	timespec_get(&endTime, TIME_UTC);

	int64_t startNanoseconds = (int64_t)startTime.tv_sec * 1000000000L + startTime.tv_nsec;
	int64_t endNanoseconds = (int64_t)endTime.tv_sec * 1000000000L + endTime.tv_nsec;

	char startBuff[100];
	char endBuff[100];
	strftime(startBuff, sizeof startBuff, "%FT%T", gmtime(&startTime.tv_sec));
	strftime(endBuff, sizeof endBuff, "%FT%T", gmtime(&endTime.tv_sec));

	fprintf(stdout, "%s|%d threads|%f stdDev|%d imageSize|%s.%09ld UTC|%s.%09ld UTC|%f (milliseconds)\n",
		style == STYLE_BASIC ? "BASIC" : "SIMD", numThreads,
    	bs->standardDeviation, bs->imageSize,
		startBuff, startTime.tv_nsec, endBuff, endTime.tv_nsec,
		((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	free(threadArr);
}

void *invokeBlurWithThread(void *arguments) {
	threadStruct *threadInfo = (threadStruct*)arguments;

	threadInfo->blurFunc(threadInfo->bs, threadInfo->start, threadInfo->end);

	return NULL;
}

int availableCores() {
	return sysconf(_SC_NPROCESSORS_CONF);
}

void printConvolutionMatrix(blurStruct *bs) {
	if(bs) {
		int convolutionElementSide = bs->convolutionDistance * 2 + 1;
		int y = -1 * bs->convolutionDistance;
		int matrixOffsetY = 0;
		while(y <= bs->convolutionDistance) {
			int x = -1 * bs->convolutionDistance;
			int matrixOffsetX = 0;
			while(x <= bs->convolutionDistance) {
				//bs->convolutionMatrix[matrixOffsetY + matrixOffsetX] = (1.0 / (2.0 * M_PI * pow(stdDev, 2.0)) * exp(-1.0 * ((pow(x, 2.0) + pow(y, 2.0)) / (2.0 * pow(stdDev, 2.0)))));
				printf(" %10.8f ", bs->convolutionMatrix[matrixOffsetY + matrixOffsetX]);
				x++;
				matrixOffsetX++;
			}
			printf("\n");
			y++;
			matrixOffsetY += convolutionElementSide;
		}
	}
}