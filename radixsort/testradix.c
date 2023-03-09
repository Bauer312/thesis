#include "radix.h"

int testExecuteOptions(int32_t numElements, int32_t radixSize, int32_t numThreads, int32_t style) {
	radixStruct rs;
	memset(&rs, 0, sizeof(radixStruct));

	if(initRadixStruct(&rs, numElements, radixSize) != 0) {
		fprintf(stderr, "Unable to initialize structure\n");
		return -1;
	}

	populateRadixStruct(&rs);

	execute(&rs, numThreads, style);

	if(testSorted(&rs) != 0) {
		fprintf(stderr, "The array is not sorted\n");
		return -1;
	}

	freeRadixStruct(&rs);

	return 0;
}

int testQS(int32_t numElements) {
	radixStruct rs;
	memset(&rs, 0, sizeof(radixStruct));

	if(initRadixStruct(&rs, numElements, 8) != 0) {
		fprintf(stderr, "Unable to initialize structure\n");
		return -1;
	}

	populateRadixStruct(&rs);
	qs(rs.primaryArr, 0, rs.numElements - 1);

	if(testSorted(&rs) != 0) {
		fprintf(stderr, "The array is not sorted\n");
		return -1;
	}

	freeRadixStruct(&rs);

	return 0;
}

int main() {
	int testStatus = 0;

	if(testQS(10000) != 0) {
			testStatus++;
			fprintf(stderr, "FAIL: testQS 10000\n");
	}

	int32_t maxThreads = sysconf(_SC_NPROCESSORS_CONF);
	for(int32_t i = maxThreads; i > 0; i--) {
		/*
		if(testExecuteOptions(500000, 4, i, STYLE_BASIC) != 0) {
			testStatus++;
			fprintf(stderr, "FAIL: %s %d\n", "testExecuteOptions 500000 4", i);
		}
		*/
		if(testExecuteOptions(50000, 8, i, STYLE_BASIC) != 0) {
			testStatus++;
			fprintf(stderr, "FAIL: %s %d\n", "testExecuteOptions 50000 8", i);
		}
		if(testExecuteOptions(50000, 8, i, STYLE_SIMD) != 0) {
			testStatus++;
			fprintf(stderr, "FAIL: %s %d\n", "testExecuteOptions 50000 8", i);
		}
	}

	if(testStatus == 0) {
		fprintf(stdout, "All tests successful\n");
	} else {
		fprintf(stdout, "Some tests failed\n");
	}
}
