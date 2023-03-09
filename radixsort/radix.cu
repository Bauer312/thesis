#include "radix.h"
#include "radix.c"

#define RADIX_COUNT 256
#define RADIX_MASK 255
#define RADIX_SHIFT 8
#define RADIX_PASSES 4

__global__ void msb(int numElements, unsigned int *firstArr, unsigned int *secondArr, unsigned int *histogramArr) {
	unsigned int shift = RADIX_SHIFT * (RADIX_PASSES - 1);
	unsigned int mask = RADIX_MASK << shift;
	unsigned int atomicInc = 1;
	int numRadix = RADIX_COUNT;

	unsigned int *srcArr = firstArr;
	unsigned int *tgtArr = secondArr;

	int elementNumber = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for(int i = elementNumber; i < numElements; i += stride) {
		unsigned int maskedValue = (srcArr[i] & mask) >> shift;
		atomicAdd(&histogramArr[maskedValue], atomicInc);
	}

	__syncthreads();

	if(elementNumber == 0) {
		unsigned int temp = 0;
		unsigned int count = 0;
		for(int i = 0; i < numRadix; i++) {
			temp = histogramArr[i];
			if(temp > 0) { histogramArr[i] = count; }
			count += temp;
		}
	}

	__syncthreads();

	for(int i = elementNumber; i < numElements; i += stride) {
		unsigned int maskedValue = (srcArr[i] & mask) >> shift;
		unsigned int newIndex = atomicAdd(&histogramArr[maskedValue], atomicInc);
		tgtArr[newIndex] = srcArr[i];
	}

}

__global__ void msb2(int numElements, unsigned int *firstArr, unsigned int *secondArr, unsigned int *histogramArr) {
	unsigned int radixValue = blockIdx.x;

	// Do not bother to do anything unless there are elements in this radix value
	if(histogramArr[radixValue] > 0) {
		unsigned int elementNumber = threadIdx.x;
		unsigned int stride = blockDim.x;

		__shared__ unsigned int localHist[RADIX_COUNT];
		if(elementNumber == 0) {
			memset(&localHist, 0, sizeof(unsigned int) * RADIX_COUNT);
		}
		__syncthreads();

		int startIndex = 0;
		int endIndex = histogramArr[radixValue];

		if(radixValue > 0) {
			int temp = radixValue - 1;
			while(temp > 0 && histogramArr[temp] == 0) { temp--; }
			startIndex = histogramArr[temp];
		}

		// At this point, run MSB on the elements from startIndex to endIndex (not including endIndex)
		unsigned int *srcArr = secondArr;
		unsigned int *tgtArr = firstArr;
		unsigned int shift = RADIX_SHIFT * (RADIX_PASSES - 2);
		unsigned int mask = RADIX_MASK << shift;
		unsigned int atomicInc = 1;
		unsigned int groupElements = endIndex - startIndex;
		unsigned int maskedValue;
		unsigned int newIndex;

		for(int idx = elementNumber; idx < groupElements; idx += stride) {
			maskedValue = (srcArr[idx + startIndex] & mask) >> shift;
			atomicAdd(&localHist[maskedValue], atomicInc);
		}
		__syncthreads();

		if(elementNumber == 0) {
			unsigned int temp = 0;
			unsigned int count = startIndex;
			for(int idx = 0; idx < RADIX_COUNT; idx++) {
				temp = localHist[idx];
				if(temp > 0) { localHist[idx] = count; }
				count += temp;
			}
		}

		__syncthreads();

		// Save for later...
		unsigned int lsbStart = localHist[threadIdx.x];

		for(int idx = elementNumber; idx < groupElements; idx += stride) {
			maskedValue = (srcArr[idx + startIndex] & mask) >> shift;
			newIndex = atomicAdd(&localHist[maskedValue], atomicInc);
			tgtArr[newIndex] = srcArr[idx + startIndex];
		}
		__syncthreads();

		// Do LSB for pass 3 and 4
		// Each thread handles the entire radix range of a radix subset
		unsigned int lsbEnd = localHist[threadIdx.x];
		unsigned int subsetSize = lsbEnd - lsbStart;

		// Only do work if there is more than 1 element to sort
		if(subsetSize > 1) {
			unsigned int lsbHist[RADIX_COUNT];
			shift = 0;
			mask = RADIX_MASK;

			for(unsigned int pIdx = 0; pIdx < 2; pIdx++) {
				unsigned int *tempArr = srcArr;
				srcArr = tgtArr;
				tgtArr = tempArr;

				memset(&lsbHist, 0, sizeof(unsigned int) * RADIX_COUNT);

				for(unsigned int idx = lsbStart; idx < lsbEnd; idx++) {
					maskedValue = (srcArr[idx] & mask) >> shift;
					lsbHist[maskedValue]++;
				}

				unsigned int temp = 0;
				unsigned int count = lsbStart;
				for(int idx = 0; idx < RADIX_COUNT; idx++) {
					temp = lsbHist[idx];
					if(temp > 0) { lsbHist[idx] = count; }
					count += temp;
				}

				for(unsigned int idx = lsbStart; idx < lsbEnd; idx++) {
					maskedValue = (srcArr[idx] & mask) >> shift;
					tgtArr[lsbHist[maskedValue]] = srcArr[idx];
					lsbHist[maskedValue]++;
				}
				shift = RADIX_SHIFT;
				mask = mask << shift;
			}
		}
	}
}

__global__ void histogram(int numElements, int slotNum, unsigned int *srcArr, unsigned int *histogramArr) {
	unsigned int shift = slotNum * 8;
	unsigned int mask = 255 << shift;
	unsigned int atomicInc = 1;

	int elementNumber = threadIdx.x + blockIdx.x * blockDim.x;

	if(elementNumber < numElements) {
		unsigned int maskedValue = (srcArr[elementNumber] & mask) >> shift;
		atomicAdd(&histogramArr[maskedValue], atomicInc);
	}
}

__global__ void sort(int numElements, int slotNum, unsigned int *srcArr, unsigned int *tgtArr, unsigned int *histogramArr) {
	unsigned int shift = slotNum * 8;
	unsigned int mask = 255 << shift;
	unsigned int numRadixValues = 256;

	unsigned int temp = 0;
	unsigned int count = 0;
	for(int i = 0; i < numRadixValues; i++) {
		temp = histogramArr[i];
		histogramArr[i] = count;
		count += temp;
	}

	for(int i = 0; i < numElements; i++) {
		unsigned int maskedValue = (srcArr[i] & mask) >> shift;
		tgtArr[histogramArr[maskedValue]] = srcArr[i];
		histogramArr[maskedValue]++;
	}
	memset(histogramArr, 0, sizeof(unsigned int) * numRadixValues);
}

void printArr(unsigned int *arr, int numElements) {
	for(int i = 0; i < numElements; i++) {
		printf("%.8x\n", arr[i]);
	}
}

void executeRadixSortCuda(radixStruct *rs) {
	struct timespec startTime;
	struct timespec endTime;

	cudaError_t err;

	int deviceId;
	err = cudaGetDevice(&deviceId);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Get Device Id Error: %s\n", cudaGetErrorString(err)); }

	// Allocate GPU memory
	size_t arraySize = sizeof(unsigned int) * rs->numElements;
	size_t radixSize = sizeof(unsigned int) * rs->numRadixValues;

	unsigned int *primaryArr;
	unsigned int *secondaryArr;
	unsigned int *histogramArr;

	err = cudaMalloc(&primaryArr, arraySize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Primary Array Allocation Error: %s\n", cudaGetErrorString(err)); }
	err = cudaMalloc(&secondaryArr, arraySize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Secondary Array Allocation Error: %s\n", cudaGetErrorString(err)); }
	err = cudaMalloc(&histogramArr, radixSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Histogram Array Allocation Error: %s\n", cudaGetErrorString(err)); }

	err = cudaMemset(histogramArr, 0, radixSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Histogram Array Memset Error: %s\n", cudaGetErrorString(err)); }

	timespec_get(&startTime, TIME_UTC);

	// Copy data to the GPU
	err = cudaMemcpy(primaryArr, (unsigned int*)rs->primaryArr, arraySize, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy primaryArr Error: %s\n", cudaGetErrorString(err)); }

	// Execute...
	//size_t threadsPerBlock = 256;
	//size_t numBlocks = (rs->numElements + threadsPerBlock - 1) / threadsPerBlock;

	msb<<<1, 1024>>>(rs->numElements, primaryArr, secondaryArr, histogramArr);
	msb2<<<RADIX_COUNT, RADIX_COUNT>>>(rs->numElements, primaryArr, secondaryArr, histogramArr);
	//histogram<<<numBlocks, threadsPerBlock>>>(rs->numElements, 0, primaryArr, histogramArr);
	//sort<<<1,1>>>(rs->numElements, 0, primaryArr, secondaryArr, histogramArr);
	//histogram<<<numBlocks, threadsPerBlock>>>(rs->numElements, 1, secondaryArr, histogramArr);
	//sort<<<1,1>>>(rs->numElements, 1, secondaryArr, primaryArr, histogramArr);
	//histogram<<<numBlocks, threadsPerBlock>>>(rs->numElements, 2, primaryArr, histogramArr);
	//sort<<<1,1>>>(rs->numElements, 2, primaryArr, secondaryArr, histogramArr);
	//histogram<<<numBlocks, threadsPerBlock>>>(rs->numElements, 3, secondaryArr, histogramArr);
	//sort<<<1,1>>>(rs->numElements, 3, secondaryArr, primaryArr, histogramArr);
	err = cudaGetLastError();
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Execution Error: %s\n", cudaGetErrorString(err)); }

	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Synchronize Error: %s\n", cudaGetErrorString(err)); }

	// Copy data back to CPU
	err = cudaMemcpy((unsigned int*)rs->primaryArr, primaryArr, arraySize, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy primaryArr Error: %s\n", cudaGetErrorString(err)); }
	//err = cudaMemcpy((unsigned int*)rs->secondaryArr, secondaryArr, arraySize, cudaMemcpyDeviceToHost);
	//if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy secondaryArr Error: %s\n", cudaGetErrorString(err)); }

	timespec_get(&endTime, TIME_UTC);

	// Free memory
	err = cudaFree(primaryArr);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Primary Array Free Error: %s\n", cudaGetErrorString(err)); }
	err = cudaFree(secondaryArr);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Secondary Array Free Error: %s\n", cudaGetErrorString(err)); }
	err = cudaFree(histogramArr);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Histogram Array Free Error: %s\n", cudaGetErrorString(err)); }

	int64_t startNanoseconds = (int64_t)startTime.tv_sec * 1000000000L + startTime.tv_nsec;
	int64_t endNanoseconds = (int64_t)endTime.tv_sec * 1000000000L + endTime.tv_nsec;
	char startBuff[100];
	char endBuff[100];
	strftime(startBuff, sizeof startBuff, "%FT%T", gmtime(&startTime.tv_sec));
	strftime(endBuff, sizeof endBuff, "%FT%T", gmtime(&endTime.tv_sec));
	fprintf(stdout, "CudaGPU|1 threads|%d elements|%s.%09ld UTC|%s.%09ld UTC|%f (milliseconds)\n",
			rs->numElements, startBuff, startTime.tv_nsec, endBuff, endTime.tv_nsec,
			((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	// Verify that the array is now sorted
	if(testSorted(rs) != 0) {
		fprintf(stderr, "The array is not sorted\n");
	} else {
		fprintf(stdout, "The array is sorted!\n");
	}
}

int main(int argc, char **argv) {
	if(argc < 2) {
		fprintf(stderr, "Usage: %s <numElements>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	int32_t numElements = (int32_t)atoi(argv[1]);

	//fprintf(stdout, "Creating data structure with %d elements\n", numElements);

	radixStruct rs;

	if(initRadixStruct(&rs, numElements, 8) != 0) {
		fprintf(stderr, "Unable to initialize radix structure\n");
		exit(EXIT_FAILURE);
	}

	//fprintf(stdout, "Populating data structure with random data\n");
	populateRadixStruct(&rs);

	sleep(2);
	executeRadixSortCuda(&rs);
	sleep(2);

	freeRadixStruct(&rs);

	return 0;
}
