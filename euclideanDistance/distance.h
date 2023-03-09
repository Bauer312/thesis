#ifndef distance_h
#define distance_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
	#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
	#include <arm_neon.h>
#endif

typedef struct distanceStruct {
	int32_t numElements;
	int32_t numClusters;
	int32_t numDimensions;
	int32_t actualDimensions;
	
	int32_t *elements;
	int32_t *clusters;
	int64_t *distances;
} distanceStruct;

typedef struct threadStruct {
	int start;
	int end;
	pthread_t thread;
	distanceStruct *ds;
	int64_t (*distanceFunc)(distanceStruct*, int32_t, int32_t);
} threadStruct;

#define STYLE_BASIC 1
#define STYLE_SIMD 2
#define STYLE_CUDA 3

/// Allocates and returns an initialized structure
int32_t initDistanceStruct(distanceStruct *ds, int32_t numElements, int32_t numClusters, int32_t numDimensions);
/// Free the memory associated with the distance structure
void freeDistanceStruct(distanceStruct *ds);
/// Populates the provided distance structure with random values
void populateDistanceStruct(distanceStruct *ds);
/// Calculate the distance between the requested cluster and element
int64_t distanceBetween(distanceStruct *ds, int32_t cluster, int32_t element);
/// Calculate the distance between the requested cluster and element using SIMD
int64_t distanceBetweenSIMD(distanceStruct *ds, int32_t cluster, int32_t element);
/// Set the value of a specific dimension in the provided element
void setElementValue(distanceStruct *ds, int32_t element, int32_t dimension, int32_t value);
/// Set the value of a specific dimention in the provided cluster
void setClusterValue(distanceStruct *ds, int32_t cluster, int32_t dimension, int32_t value);


/// Calculate the distance between all elements and clusters
void calculateDistance(distanceStruct *ds, int startIndex, int endIndex);
void calculateDistanceSIMD(distanceStruct *ds, int startIndex, int endIndex);
void execute(distanceStruct *ds, int numThreads, int numIterations, int style);
void *invokeDistanceWithThread(void *arguments);
int availableCores();
#endif /* distance_h */
