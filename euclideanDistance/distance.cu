#include "distance.h"
#include "distance.c"

// Compute all distances in one two-dimensional grid
__global__ void distanceCuda(int32_t numElements,
		int32_t numClusters,
		int32_t numDimensions,
		int32_t *elements,
		int32_t *clusters,
		int64_t *distances) {

	int elementNumber = threadIdx.x + blockIdx.x * blockDim.x;

	// Idiomatic CUDA - make sure that we don't go beyond the end of the array
	if(elementNumber < numElements) {
		int clusterNumber = blockIdx.y;
		int elementOffset = elementNumber * numDimensions;
		int clusterOffset = clusterNumber * numDimensions;

		int64_t difference = 0;
		int64_t distance = 0;

		// Calculate the distance iteratively
		for(int i = 0; i < numDimensions; i++) {
			difference = elements[elementOffset + i] - clusters[clusterOffset + i];
			distance += difference * difference;
		}

		// Store the distance
		distances[clusterNumber * numElements + elementNumber] = distance;
	}
}

// Use shared memory to calculate the distances.  Each thread in a block represents a single
//	dimension, while each block represents a single element and a single cluster.
__global__ void distanceShared(int32_t numDimensions,
		int32_t clusterNumber,
		int32_t *element,
		int32_t *cluster,
		int64_t *distance) {

	extern __shared__ int64_t sharedArr[];

	// gridDim.x is the number of elements (number of blocks in grid)
	// blockIdx.x is the element number (block within grid)
	// threadIdx.x is the dimension number (thread within block)

	int64_t difference = element[blockIdx.x * numDimensions + threadIdx.x] - cluster[threadIdx.x];
	sharedArr[threadIdx.x] = difference * difference;

	__syncthreads();

	// At this point, the sharedArr contains the values
	//	that need to be summed.  Do a parallel reduction
	//	to end up with the final sum.
	int dim = numDimensions / 2;
	while(dim >= 1) {
		if(threadIdx.x < dim) {
			//printf("sharedArr[%d] += sharedArr[%d]\n", threadIdx.x, threadIdx.x + dim);
			sharedArr[threadIdx.x] += sharedArr[threadIdx.x + dim];
		}
		dim = dim / 2;
		__syncthreads();
	}

	// At this point, sharedArr[0] contains the full distance
	if(threadIdx.x == 0) {
		distance[clusterNumber * gridDim.x + blockIdx.x] = sharedArr[0];
	}
}


void calculateDistanceCuda(distanceStruct *ds, int numIterations) {
	struct timespec startTime;
	struct timespec endTime;

	cudaError_t err;

	int deviceId;
	err = cudaGetDevice(&deviceId);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Get Device Id Error: %s\n", cudaGetErrorString(err)); }


	// Allocate GPU memory
	size_t elementArrSize = sizeof(int32_t) * ds->numElements * ds->numDimensions;
	int32_t *elements;
	err = cudaMalloc(&elements, elementArrSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Malloc Element Error: %s\n", cudaGetErrorString(err)); }

	size_t clusterArrSize = sizeof(int32_t) * ds->numClusters * ds->numDimensions;
	int32_t *clusters;
	err = cudaMalloc(&clusters, clusterArrSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Malloc Cluster Error: %s\n", cudaGetErrorString(err)); }

	size_t distanceArrSize = sizeof(int64_t) * ds->numClusters * ds->numElements;
	int64_t *distances;
	err = cudaMalloc(&distances, distanceArrSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Malloc Distances Error: %s\n", cudaGetErrorString(err)); }

	timespec_get(&startTime, TIME_UTC);

	// Copy data to GPU
	err = cudaMemcpy(elements, ds->elements, elementArrSize, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy Element Error: %s\n", cudaGetErrorString(err)); }
	err = cudaMemcpy(clusters, ds->clusters, clusterArrSize, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy Cluster Error: %s\n", cudaGetErrorString(err)); }

	// Execute
	size_t threadsPerBlock = 256;
	size_t numBlocksX = (ds->numElements + threadsPerBlock - 1) / threadsPerBlock;
	size_t numBlocksY = ds->numClusters;

	for(int iter = 0; iter < numIterations; iter++) {
		distanceCuda<<<dim3(numBlocksX,numBlocksY,1), dim3(threadsPerBlock,1,1)>>>(ds->numElements, ds->numClusters, ds->numDimensions, elements, clusters, distances);
		err = cudaGetLastError();
		if(err != cudaSuccess) { fprintf(stderr, "CUDA Execution Error: %s\n", cudaGetErrorString(err)); }

		err = cudaDeviceSynchronize();
		if(err != cudaSuccess) { fprintf(stderr, "CUDA Synchronize Error: %s\n", cudaGetErrorString(err)); }
	}

	// Copy results back to CPU
	err = cudaMemcpy(ds->distances, distances, distanceArrSize, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy Distances Error: %s\n", cudaGetErrorString(err)); }

	timespec_get(&endTime, TIME_UTC);

	// Free memory
	err = cudaFree(elements);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Free Element Error: %s\n", cudaGetErrorString(err)); }
	err = cudaFree(clusters);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Free Cluster Error: %s\n", cudaGetErrorString(err)); }
	err = cudaFree(distances);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Free Distances Error: %s\n", cudaGetErrorString(err)); }


	int64_t startNanoseconds = (int64_t)startTime.tv_sec * 1000000000L + startTime.tv_nsec;
	int64_t endNanoseconds = (int64_t)endTime.tv_sec * 1000000000L + endTime.tv_nsec;
	//fprintf(stderr, "Duration (nanoseconds) %" PRId64 "\n", endNanoseconds - startNanoseconds);
	//fprintf(stderr, "Duration (milliseconds) %f\n", ((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	char startBuff[100];
	char endBuff[100];
	strftime(startBuff, sizeof startBuff, "%FT%T", gmtime(&startTime.tv_sec));
	strftime(endBuff, sizeof endBuff, "%FT%T", gmtime(&endTime.tv_sec));
	fprintf(stdout, "CudaGPU|1 threads|%d iterations|%s.%09ld UTC|%s.%09ld UTC|%f (milliseconds)\n",
			numIterations, startBuff, startTime.tv_nsec, endBuff, endTime.tv_nsec,
			((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	// Now that we have calcuated everything, let's test the data
	/*
	int64_t distance;
	for(int32_t cIdx = 0; cIdx < ds->numClusters; cIdx++) {
		int clusterOffset = cIdx * ds->numElements;
		for(int32_t eIdx = 0; eIdx < ds->numElements; eIdx++) {
			distance = distanceBetween(ds, cIdx, eIdx);
			if(ds->distances[clusterOffset + eIdx] != distance) {
				fprintf(stderr, "Cluster %d Element %d -> %" PRId64 " vs %" PRId64 "\n", cIdx, eIdx, ds->distances[clusterOffset + eIdx], distance);
			}
		}
	}
	*/
	//fprintf(stdout, "Done checking data\n");
}

void calculateDistanceCudaMultipass(distanceStruct *ds, int numIterations) {
	struct timespec startTime;
	struct timespec endTime;

	cudaError_t err;

	int deviceId;
	err = cudaGetDevice(&deviceId);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Get Device Id Error: %s\n", cudaGetErrorString(err)); }


	cudaStream_t *streams;
	streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * ds->numClusters);

	// Allocate GPU memory
	size_t elementArrSize = sizeof(int32_t) * ds->numElements * ds->numDimensions;
	int32_t *elements;
	err = cudaMallocManaged(&elements, elementArrSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Malloc Element Error: %s\n", cudaGetErrorString(err)); }

	size_t clusterArrSize = sizeof(int32_t) * ds->numClusters * ds->numDimensions;
	int32_t *clusters;
	err = cudaMallocManaged(&clusters, clusterArrSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Malloc Cluster Error: %s\n", cudaGetErrorString(err)); } 

	size_t distanceArrSize = sizeof(int64_t) * ds->numClusters * ds->numElements;
	int64_t *distances;
	err = cudaMallocManaged(&distances, distanceArrSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Malloc Distance Error: %s\n", cudaGetErrorString(err)); }


	for(int i = 0; i < ds->numClusters; i++) {
	err = cudaStreamCreate(&streams[i]);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Stream Create %d Error: %s\n", i, cudaGetErrorString(err)); }
	}

	memcpy(elements, ds->elements, elementArrSize);
	memcpy(clusters, ds->clusters, clusterArrSize);

	timespec_get(&startTime, TIME_UTC);

	cudaMemPrefetchAsync(elements, elementArrSize, deviceId);
	cudaMemPrefetchAsync(clusters, clusterArrSize, deviceId);

	// Execute
	size_t threadsPerBlock = ds->numDimensions;
	size_t numBlocks = ds->numElements;

	for(int iter = 0; iter < numIterations; iter++) {
		for(int i = 0; i < ds->numClusters; i++) {
		distanceShared<<<numBlocks, threadsPerBlock, sizeof(int64_t) * ds->numDimensions, streams[i]>>>(ds->numDimensions,
			i, elements, &clusters[i * ds->numDimensions], distances);
		err = cudaGetLastError();
		if(err != cudaSuccess) { fprintf(stderr, "CUDA Execution %d Error: %s\n", i, cudaGetErrorString(err)); }
		}
	}

	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Synchronize Error: %s\n", cudaGetErrorString(err)); }

	memcpy(ds->distances, distances, distanceArrSize);

	timespec_get(&endTime, TIME_UTC);

	// Check data
	int64_t testDistance;
	for(int32_t cIdx = 0; cIdx < ds->numClusters; cIdx++) {
		int clusterOffset = cIdx * ds->numElements;
		for(int32_t eIdx = 0; eIdx < ds->numElements; eIdx++) {
			testDistance = distanceBetween(ds, cIdx, eIdx);
			if(ds->distances[clusterOffset + eIdx] != testDistance) {
				fprintf(stderr, "Cluster %d Element %d -> %" PRId64 " vs %" PRId64 "\n", cIdx, eIdx, ds->distances[clusterOffset + eIdx], testDistance);
			}
		}
	}

	// Free memory and destroy the streams
	err = cudaFree(elements);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Free Element Error: %s\n", cudaGetErrorString(err)); }
	err = cudaFree(clusters);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Free Cluster Error: %s\n", cudaGetErrorString(err)); }
	err = cudaFree(distances);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Free Distance Error: %s\n", cudaGetErrorString(err)); }

	for(int i = 0; i < ds->numClusters; i++) {
	err = cudaStreamDestroy(streams[i]);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Stream Destroy %d Error: %s\n", i, cudaGetErrorString(err)); }
	}
	free(streams);

	int64_t startNanoseconds = (int64_t)startTime.tv_sec * 1000000000L + startTime.tv_nsec;
	int64_t endNanoseconds = (int64_t)endTime.tv_sec * 1000000000L + endTime.tv_nsec;
	//fprintf(stderr, "Duration (nanoseconds) %" PRId64 "\n", endNanoseconds - startNanoseconds);
	//fprintf(stderr, "Duration (milliseconds) %f\n", ((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	char startBuff[100];
	char endBuff[100];
	strftime(startBuff, sizeof startBuff, "%FT%T", gmtime(&startTime.tv_sec));
	strftime(endBuff, sizeof endBuff, "%FT%T", gmtime(&endTime.tv_sec));
	fprintf(stdout, "CudaGPU|%d threads|%d iterations|%d elements|%d clusters|%d dimensions%s.%06ld UTC|%s.%06ld UTC|%f (milliseconds)\n",
					ds->numClusters, numIterations,
					ds->numElements, ds->numClusters, ds->numDimensions,
					startBuff, startTime.tv_nsec, endBuff, endTime.tv_nsec,   
					((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);
}

int main(int argc, char **argv) {
	if(argc < 4) {
		fprintf(stderr, "Usage: %s <numElements> <numClusters> <numDimensions> <optional numthreads>\n", argv[0]);
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

	//fprintf(stdout, "Creating data structure|%d elements|%d clusters|%d dimensions\n", numElements, numClusters, numDimensions);

	distanceStruct ds;
	if(initDistanceStruct(&ds, numElements, numClusters, numDimensions) != 0) {
		fprintf(stderr, "Unable to initialize the distance structure\n");
		exit(-1);
	}

	//fprintf(stdout, "Populating data structure with random data\n");
	populateDistanceStruct(&ds);

	

	// For each cluster, calculate the distance to all elements
	//execute(&ds, 1, numIterations, STYLE_BASIC);
	/*
	for(int threadCount = 1; threadCount <= numThreads; threadCount++) {
		execute(&ds, threadCount, numIterations, STYLE_BASIC);
		sleep(1);
		execute(&ds, threadCount, numIterations, STYLE_SIMD);
		sleep(1);
	}
	*/

	sleep(2);
	//fprintf(stdout, "Calculating CUDA distances with %d iterations\n", numIterations);
	calculateDistanceCuda(&ds, numIterations);
	sleep(2);

	//fprintf(stdout, "Calculating streaming CUDA distances with %d iterations\n", numIterations);
	//calculateDistanceCudaMultipass(&ds, numIterations);
	//sleep(1);
	
	freeDistanceStruct(&ds);

	return 0;
}
