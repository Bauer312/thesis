#include "radix.h"

int main(int argc, char **argv) {
	if(argc < 2) {
		fprintf(stderr, "Usage: %s <numElements> <optional numthreads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	int32_t numElements = (int32_t)atoi(argv[1]);
	int32_t numThreads = 1;

	if(argc >= 3) {
		numThreads = (int32_t)atoi(argv[2]);
	}

	if(numThreads < 0) {
		// If passed a negative number of threads, use the total number of cores/SMT for the system
		numThreads = availableCores();
	} else if(numThreads == 0) {
		// If passed 0 for number of threads, use 1 thread
		numThreads = 1;
	}

	//fprintf(stdout, "Creating data structure with %d elements\n", numElements);

	radixStruct rs;

	if(initRadixStruct(&rs, numElements, 8) != 0) {
		fprintf(stderr, "Unable to initialize radix structure\n");
		exit(-1);
	}

	//fprintf(stdout, "\nPopulating data structure with random data\n\n");
	populateRadixStruct(&rs);

	sleep(2);
	execute(&rs, numThreads, STYLE_BASIC);
	sleep(2);

	freeRadixStruct(&rs);

	return 0;
}
