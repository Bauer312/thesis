#include "distance.h"

int main(int argc, char **argv) {
	if(argc < 4) {
		fprintf(stderr, "Usage: %s <numElements> <numClusters> <numDimensions> <optional numthreads> <optional numiterations>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	int32_t numElements = (int32_t)atoi(argv[1]);
	int32_t numClusters = (int32_t)atoi(argv[2]);
	int32_t numDimensions = (int32_t)atoi(argv[3]);
	int32_t numThreads = 1;
	int32_t numIterations = 1;
	if(argc >= 5) {
		numThreads = (int32_t)atoi(argv[4]);
	}
	if(argc >= 6) {
		numIterations = (int32_t)atoi(argv[5]);
	}

	if(numThreads < 0) {
		// If passed a negative number of threads, use the total number of cores/SMT for the system
		numThreads = availableCores();
	} else if(numThreads == 0) {
		// If passed 0 for number of threads, use 1 thread
		numThreads = 1;
	}
	if(numElements < 1 || numClusters < 1 || numDimensions < 1) {
		fprintf(stderr, "Number of elements (%d), clusters, (%d), and dimensions (%d) must be greater than 0\n",
			numElements, numClusters, numDimensions);
		exit(EXIT_FAILURE);
	}

	//fprintf(stdout, "Creating data structure\n%d elements\n%d clusters\n%d dimensions\n", numElements, numClusters, numDimensions);

	distanceStruct ds;
	if(initDistanceStruct(&ds, numElements, numClusters, numDimensions) != 0) {
		fprintf(stderr, "Unable to initialize the distance structure\n");
		exit(-1);
	}

	//fprintf(stdout, "\nPopulating data structure with random data\n\n");
	populateDistanceStruct(&ds);

	sleep(2);
	execute(&ds, numThreads, numIterations, STYLE_BASIC);
	sleep(2);
	execute(&ds, numThreads, numIterations, STYLE_SIMD);
	sleep(2);

	// For each cluster, calculate the distance to all elements
	/*
	for(int threadCount = 1; threadCount <= numThreads; threadCount++) {
		fprintf(stdout, "BASIC|%d threads|%d iterations", threadCount, numIterations);
		execute(&ds, threadCount, numIterations, STYLE_BASIC);
		sleep(1);
		fprintf(stdout, "SIMD|%d threads|%d iterations", threadCount, numIterations);
		execute(&ds, threadCount, numIterations, STYLE_SIMD);
		sleep(1);
	}
	*/

	freeDistanceStruct(&ds);

	return 0;
}
