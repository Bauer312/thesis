#include "distance.h"

/// Initialize the provided structure with space for the provided number of elements and clusters.  The number
///     of dimensions is forced to be a multiple of eight (8).
int32_t initDistanceStruct(distanceStruct *ds, int32_t numElements, int32_t numClusters, int32_t numDimensions) {
	if(ds) {
		ds->numDimensions = numDimensions;

		// Force the number of actual dimensions to be a multiple of 8, but at
		//  least the provided size
		if(numDimensions % 8 != 0) {
			numDimensions = numDimensions + (8 - numDimensions % 8);
		}
		
		ds->numElements = numElements;
		ds->numClusters = numClusters;
		ds->actualDimensions = numDimensions;
		
		size_t elementSize = sizeof(int32_t) * numElements * numDimensions;
		ds->elements = (int32_t*)malloc(elementSize);
		if(ds->elements) {
			memset(ds->elements, 0, elementSize);
		}

		size_t clusterSize = sizeof(int32_t) * numClusters * numDimensions;
		ds->clusters = (int32_t*)malloc(clusterSize);
		if(ds->clusters) {
			memset(ds->clusters, 0, clusterSize);
		}

		size_t distanceArraySize = sizeof(int64_t) * numElements * numClusters;
		ds->distances = (int64_t*)malloc(distanceArraySize);
		if(ds->distances) {
		memset(ds->distances, 0, distanceArraySize);
		}
		return 0;
	}
	return -1;
}

/// Free the memory associated with the distance structure
void freeDistanceStruct(distanceStruct *ds) {
	if(ds) {
		if(ds->elements) {
			free(ds->elements);
			ds->elements = NULL;
		}
		if(ds->clusters) {
			free(ds->clusters);
			ds->clusters = NULL;
		}
		if(ds->distances) {
			free(ds->distances);
			ds->distances = NULL;
		}
	}
}

/// Populates the provided distance structure with random values between -5000 and 5000
void populateDistanceStruct(distanceStruct *ds) {
	if(ds) {
		time_t t;
		srand((unsigned) time(&t));
		
		for(int i = 0; i < ds->numElements * ds->numDimensions; i++) {
			ds->elements[i] = rand() % (10001) - 5000;
		}

		for(int i = 0; i < ds->numClusters * ds->numDimensions; i++) {
			ds->clusters[i] = rand() % (10001) - 5000;
		}
	}
}

/// Calculate the distance between the requested cluster and element
int64_t distanceBetween(distanceStruct *ds, int32_t cluster, int32_t element) {
	// Make sure it is safe to proceed
	if(ds && cluster < ds->numClusters && element < ds->numElements) {
		int32_t offsetCluster = cluster * ds->actualDimensions;
		int32_t offsetElement = element * ds->actualDimensions;
		//printf("Cluster offset: %d Element offset: %d\n", offsetCluster, offsetElement);
		
		int64_t distance = 0;
		int64_t difference = 0;
		
		for(int32_t i = 0; i < ds->numDimensions; i++) {
			int64_t clusterValue = ds->clusters[offsetCluster + i];
			int64_t elementValue = ds->elements[offsetElement + i];
			difference = clusterValue - elementValue;
			
			distance += difference * difference;

			//printf("C: %lld E: %lld Diff: %lld Dist: %lld\n",
			//    clusterValue, elementValue, difference, distance);
		}
		return distance;
	}
	return 0;
}

/// Calculate the distance between the requested cluster and element using SIMD
int64_t distanceBetweenSIMD(distanceStruct *ds, int32_t cluster, int32_t element) {
	// Make sure it is safe to proceed
	if(ds && cluster < ds->numClusters && element < ds->numElements) {
		int64_t distance = 0;
		int32_t simdSize = 4;

		#if defined(__x86_64__) || defined(_M_X64)
			__m256i clusterValues;
			__m256i elementValues;
			__m256i differences;
			__m256i accumulator;
			simdSize = 8;
		#elif defined(__aarch64__) || defined(_M_ARM64)
			int32x4_t clusterValues;
			int32x4_t elementValues;
			int32x4_t differences;
			int32x4_t accumulator;
		#endif

		for(int32_t i = 0; i < ds->actualDimensions; i+= simdSize) {
			#if defined(__x86_64__) || defined(_M_X64)
				// Load values into SIMD registers
				clusterValues = _mm256_loadu_si256((__m256i*)&(ds->clusters[cluster * ds->actualDimensions + i]));
				elementValues = _mm256_loadu_si256((__m256i*)&(ds->elements[element * ds->actualDimensions + i]));

				// Subtract element values from cluster values
				differences = _mm256_sub_epi32(clusterValues, elementValues);

				// double each element
				accumulator = _mm256_mullo_epi32(differences, differences);

				// Sum all the elements together
				// Cast the first half to 128 bits
				__m128i firstHalf = _mm256_castsi256_si128(accumulator);
				// Extract the second half
				__m128i secondHalf = _mm256_extracti128_si256(accumulator, 1);
				// Add them together
				firstHalf = _mm_add_epi32(firstHalf, secondHalf);
				// Shuffle so the 3rd and 4th are located in the 1st and 2nd
				secondHalf = _mm_shuffle_epi32(firstHalf, _MM_SHUFFLE(0,1,2,3));
				// Add again
				firstHalf = _mm_add_epi32(firstHalf, secondHalf);
				// Shuffle one more time so 2nd is now in 1st
				secondHalf = _mm_shuffle_epi32(firstHalf, _MM_SHUFFLE(2,3,0,1));
				// Add them together one last time - the resulting 1st element is the sum of all elements
				firstHalf = _mm_add_epi32(firstHalf, secondHalf);
				distance += (int64_t)_mm_extract_epi32(firstHalf, 0);
			#elif defined(__aarch64__) || defined(_M_ARM64)
				// Load values into SIMD registers
				clusterValues = vld1q_s32(&(ds->clusters[cluster * ds->actualDimensions + i]));
				elementValues = vld1q_s32(&(ds->elements[element * ds->actualDimensions + i]));

				// Subtract element values from cluster values
				differences = vsubq_s32(clusterValues, elementValues);
				
				// double each element
				accumulator = vmulq_s32(differences, differences);
				
				// sum all elements and add to the distance variable
				distance += vaddvq_s32(accumulator);
			#endif
		}
		return distance;
	}
	return 0;
}

void setElementValue(distanceStruct *ds, int32_t element, int32_t dimension, int32_t value) {
	if(ds && element < ds->numElements && dimension < ds->numDimensions) {
		int32_t offset = element * ds->actualDimensions;
		ds->elements[offset + dimension] = value;
	}
}

void setClusterValue(distanceStruct *ds, int32_t cluster, int32_t dimension, int32_t value) {
	if(ds && cluster < ds->numClusters && dimension < ds->numDimensions) {
		int32_t offset = cluster * ds->actualDimensions;
		ds->clusters[offset + dimension] = value;
	}
}

void *invokeDistanceWithThread(void *arguments) {
	threadStruct *threadInfo = (threadStruct*)arguments;

	for(int32_t cIdx = 0; cIdx < threadInfo->ds->numClusters; cIdx++) {
		for(int32_t eIdx = threadInfo->start; eIdx < threadInfo->end; eIdx++) {
			int64_t distance = threadInfo->distanceFunc(threadInfo->ds, cIdx, eIdx);
			if(distance < 0) {
				fprintf(stderr, "Distance (%" PRId64 ") cannot be less than 0\n", distance);
			}
		}
	}
	return NULL;
}

int availableCores() {
	return sysconf(_SC_NPROCESSORS_CONF);
}

// Run distances for all elements between the start and end index (start, end]
void calculateDistance(distanceStruct *ds, int startIndex, int endIndex) {
	for(int32_t cIdx = 0; cIdx < ds->numClusters; cIdx++) {
	int clusterOffset = cIdx * ds->numElements;
		for(int32_t eIdx = startIndex; eIdx < endIndex; eIdx++) {
			ds->distances[clusterOffset + eIdx] = distanceBetween(ds, cIdx, eIdx);
			if(ds->distances[clusterOffset + eIdx] < 0) {
				fprintf(stderr, "Distance (%" PRId64 ") cannot be less than 0\n", ds->distances[clusterOffset + eIdx]);
			}
		}
	}
}

// Run distances for all elements between the start and end index (start, end]
void calculateDistanceSIMD(distanceStruct *ds, int startIndex, int endIndex) {
	for(int32_t cIdx = 0; cIdx < ds->numClusters; cIdx++) {
	int clusterOffset = cIdx * ds->numElements;
		for(int32_t eIdx = startIndex; eIdx < endIndex; eIdx++) {
			ds->distances[clusterOffset + eIdx] = distanceBetweenSIMD(ds, cIdx, eIdx);
			if(ds->distances[clusterOffset + eIdx] < 0) {
				fprintf(stderr, "Distance (%" PRId64 ") cannot be less than 0\n", ds->distances[clusterOffset + eIdx]);
			}
		}
	}
}

void execute(distanceStruct *ds, int numThreads, int numIterations, int style) {
	struct timespec startTime;
	struct timespec endTime;
	timespec_get(&startTime, TIME_UTC);

	threadStruct *threadArr = (threadStruct*)malloc(sizeof(threadStruct) * numThreads);
	if(threadArr == NULL) {
		fprintf(stderr, "Unable to allocate memory for %d thread structures\n", numThreads);
	}

	for(int iter = 0; iter < numIterations; iter++) {
		if(numThreads == 1) {
			if(style == STYLE_BASIC) {
				calculateDistance(ds, 0, ds->numElements);
			} else if(style == STYLE_SIMD) {
				calculateDistanceSIMD(ds, 0, ds->numElements);
			}

		} else {
			int elementsPerThread = ds->numElements / numThreads;
			int leftovers = ds->numElements % numThreads;



			threadArr[0].start = 0;
			threadArr[0].end = elementsPerThread + leftovers;
			threadArr[0].ds = ds;
			if(style == STYLE_BASIC) {
				threadArr[0].distanceFunc = &distanceBetween;
			} else if(style == STYLE_SIMD) {
				threadArr[0].distanceFunc = &distanceBetweenSIMD;
			}

			for(int i = 0; i < numThreads; i++) {
				if(i < numThreads - 1) {
					threadArr[i+1].start = threadArr[i].end;
					threadArr[i+1].end = threadArr[i].end + elementsPerThread;
					threadArr[i+1].ds = ds;
					if(style == STYLE_BASIC) {
						threadArr[i+1].distanceFunc = &distanceBetween;
					} else if(style == STYLE_SIMD) {
						threadArr[i+1].distanceFunc = &distanceBetweenSIMD;
					}
				}
				if(pthread_create(&(threadArr[i].thread), NULL, invokeDistanceWithThread, &threadArr[i]) != 0) {
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
		}
	}

	timespec_get(&endTime, TIME_UTC);

	int64_t startNanoseconds = (int64_t)startTime.tv_sec * 1000000000L + startTime.tv_nsec;
	int64_t endNanoseconds = (int64_t)endTime.tv_sec * 1000000000L + endTime.tv_nsec;
	//fprintf(stderr, "Duration (nanoseconds) %" PRId64 "\n", endNanoseconds - startNanoseconds);
	//fprintf(stderr, "Duration (milliseconds) %f\n", ((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	char startBuff[100];
	char endBuff[100];
	strftime(startBuff, sizeof startBuff, "%FT%T", gmtime(&startTime.tv_sec));
	strftime(endBuff, sizeof endBuff, "%FT%T", gmtime(&endTime.tv_sec));
	//printf("Start time: %s.%06ld UTC\n", startBuff, startTime.tv_nsec);
	//printf("End time: %s.%06ld UTC\n", endBuff, endTime.tv_nsec);

	fprintf(stdout, "%s|%d threads|%d iterations|%d elements|%d clusters|%d dimensions|%s.%09ld UTC|%s.%09ld UTC|%f (milliseconds)\n",
		style == STYLE_BASIC ? "BASIC" : "SIMD",
		numThreads, numIterations,
		ds->numElements, ds->numClusters, ds->numDimensions,
		startBuff, startTime.tv_nsec, endBuff, endTime.tv_nsec,
		((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	free(threadArr);
}
