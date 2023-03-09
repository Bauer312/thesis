#include "blur.h"

int testBasic() {
	int side = matrixSize(0.123);

	if(side != 1) {
		fprintf(stderr, "Invalid side length: %d vs 1\n", side);
		return -1;
	}

	return 0;
}

int testSmallStdDev() {
	double stdDev = 0.25;
	int32_t imageSize = 16;

	blurStruct bs;
	if(initBlurStruct(&bs, stdDev, imageSize) != 0) {
		fprintf(stderr, "Unable to initialize structure\n");
		return -1;
	}

	printConvolutionMatrix(&bs);

	freeBlurStruct(&bs);

	return 0;
}

int testMediumStdDev() {
	double stdDev = 0.5;
	int32_t imageSize = 16;

	blurStruct bs;
	if(initBlurStruct(&bs, stdDev, imageSize) != 0) {
		fprintf(stderr, "Unable to initialize structure\n");
		return -1;
	}

	printConvolutionMatrix(&bs);

	freeBlurStruct(&bs);

	return 0;
}

int getSideLength() {
	double stdDev = 0.84089642;
	int32_t imageSize = 16;

	blurStruct bs;
	if(initBlurStruct(&bs, stdDev, imageSize) != 0) {
		fprintf(stderr, "Unable to initialize structure\n");
		return -1;
	}

	populateBlurStruct(&bs);

	if(bs.convolutionDistance != 3) {
		fprintf(stderr, "Incorrect convolution distance: %d vs 3\n", bs.convolutionDistance);
		return -1;
	}

	blurImage(&bs, 0, bs.imageSize);

	freeBlurStruct(&bs);

	return 0;
}

int main() {
	int testStatus = 0;

	if(testBasic() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "testBasic");
	}

	if(testSmallStdDev() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "testSmallStdDev");
	}

	if(testMediumStdDev() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "testMediumStdDev");
	}

	if(getSideLength() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "getSideLength");
	}

	if(testStatus == 0) {
		fprintf(stdout, "All tests successful\n");
	} else {
		fprintf(stdout, "Some tests failed\n");
	}
}