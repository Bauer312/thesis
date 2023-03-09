#include "blur.h"

int main(int argc, char **argv) {
	if(argc < 3) {
		fprintf(stderr, "Usage: %s <standardDeviation> <imageSize> <optional numthreads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	double stdDev = atof(argv[1]);
	int32_t imgSize = (int32_t)atoi(argv[2]);
	int32_t numThreads = 1;

	if(argc >= 4) {
		numThreads = (int32_t)atoi(argv[3]);
	}

	if(numThreads < 0) {
		// If passed a negative number of threads, use the total number of cores/SMT for the system
		numThreads = availableCores();
	} else if(numThreads == 0) {
		// If passed 0 for number of threads, use 1 thread
		numThreads = 1;
	}

	//fprintf(stdout, "Creating data structure\n%f standard deviation\n%d image Size\n", stdDev, imgSize);

	blurStruct bs;

	if(initBlurStruct(&bs, stdDev, imgSize) != 0) {
		fprintf(stderr, "Unable to initialize blur structure\n");
		exit(-1);
	}

	//fprintf(stdout, "\nPopulating data structure with random image data\n\n");
	populateBlurStruct(&bs);

	sleep(2);
	execute(&bs, numThreads, STYLE_BASIC);
	sleep(2);
	execute(&bs, numThreads, STYLE_SIMD);
	sleep(2);

	/*
	for(int threadCount = 1; threadCount <= numThreads; threadCount++) {
		execute(&bs, threadCount, STYLE_BASIC);
		sleep(1);
		execute(&bs, threadCount, STYLE_SIMD);
		sleep(1);

		// Compare newImage to simdImage
		for(int row = bs.convolutionDistance; row < bs.imageSize - bs.convolutionDistance; row++) {
			size_t rowOffset = row * bs.imageSize;
			for(int col = bs.convolutionDistance; col < bs.imageSize - bs.convolutionDistance; col++) {
				if(abs(bs.newImage[rowOffset + col] - bs.simdImage[rowOffset + col]) > 1) {
					fprintf(stderr, "[%d,%d] %d vs %d\n", row, col, bs.newImage[rowOffset + col], bs.simdImage[rowOffset + col]);
				}
			}
		}
	}
	*/

	freeBlurStruct(&bs);

	return 0;
}
