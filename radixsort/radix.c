#include "radix.h"

int32_t initRadixStruct(radixStruct *rs, int32_t numElements, int32_t radixSize) {
	if(rs) {
		rs->numElements = numElements;
		rs->radixSize = radixSize;
		if(radixSize == 8) {
			rs->mask = 255;
			rs->numRadixValues = 256;
		} else if(radixSize == 4) {
			rs->mask = 15;
			rs->numRadixValues = 16;
		} else {
			fprintf(stderr, "Invalid radix size (must be 4 or 8)\n");
			return -1;
		}
		rs->numSlots = (sizeof(uint32_t) * 8) / rs->radixSize;

		rs->primaryArr = (uint32_t*)malloc(sizeof(uint32_t) * numElements);
		if(rs->primaryArr) {
			memset(rs->primaryArr, 0, sizeof(uint32_t) * numElements);
		} else {
			fprintf(stderr, "Unable to allocate memory for the primary array\n");
			return -1;
		}

		rs->secondaryArr = (uint32_t*)malloc(sizeof(uint32_t) * numElements);
		if(rs->secondaryArr) {
			memset(rs->secondaryArr, 0, sizeof(uint32_t) * numElements);
		} else {
			fprintf(stderr, "Unable to allocate memory for the secondary array\n");
			return -1;
		}

		return 0;
	}
	return -1;
}

void freeRadixStruct(radixStruct *rs) {
	if(rs) {
		if(rs->primaryArr) {
			free(rs->primaryArr);
		} else {
			fprintf(stderr, "primaryArr is a NULL pointer\n");
		}
		if(rs->secondaryArr) {
			free(rs->secondaryArr);
		} else {
			fprintf(stderr, "secondaryArr is a NULL pointer\n");
		}
		memset(rs, 0, sizeof(radixStruct));
	} else {
		fprintf(stderr, "Passed a NULL pointer to freeRadixStruct\n");
	}
}

void populateRadixStruct(radixStruct *rs) {
	if(rs) {
		time_t t;
		srand((unsigned) time(&t));

		for(int i = 0; i < rs->numElements; i++) {
			rs->primaryArr[i] = rand();
		}
	}
}

void printRadix(radixStruct *rs, int32_t whichSlot, int32_t swapped) {
	uint32_t mask;

	uint32_t *usedArr = rs->primaryArr;
	if(swapped == 1) {
		usedArr = rs->secondaryArr;
	}

	int32_t numSlots = (sizeof(uint32_t) * 8) / rs->radixSize;
	if(whichSlot >= numSlots) {
		fprintf(stdout, "Not a valid radix slot %d (total # slots = %d)\n", whichSlot, numSlots);
		return;
	}

	int32_t shiftAmount = (numSlots - (whichSlot + 1)) * rs->radixSize;
	mask = rs->mask << shiftAmount;

	fprintf(stdout, "\n-----\n Radix Size: %d Slot %d Mask %.8x\n", rs->radixSize, whichSlot, mask);
	for(int32_t i = 0; i < rs->numElements; i++) {
		fprintf(stdout, "%.8x == %.8x == %.8x\n",
			usedArr[i],
			usedArr[i] & mask,
			(usedArr[i] & mask) >> shiftAmount);
	}
}

void populateHistogram(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end) {
	uint32_t mask;

	// Alternate between the primary and secondary arrays
	uint32_t *usedArr = rs->primaryArr;
	if(whichSlot % 2 == 1) {
		usedArr = rs->secondaryArr;
	}

	int32_t shiftAmount = (rs->numSlots - (whichSlot + 1)) * rs->radixSize;
	mask = rs->mask << shiftAmount;

	int32_t idx = start;
	uint32_t maskedValue;
	while(idx < end) {
		maskedValue = (usedArr[idx] & mask) >> shiftAmount;
		histogram[maskedValue].numElements++;
		idx++;
	}

	uint32_t currentIndex = start;
	for(uint32_t idx = 0; idx < rs->numRadixValues; idx++) {
		histogram[idx].radixValue = idx;
		histogram[idx].startIndex = currentIndex;
		histogram[idx].currentIndex = currentIndex;
		//fprintf(stdout, "Hist Slot %d Radix %.2x Index: %d\n", whichSlot, idx, currentIndex);
		currentIndex += histogram[idx].numElements;
	}
}

#if defined(__x86_64__) || defined(_M_X64)
void populateHistogramSIMD(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end) {
	uint32_t mask;

	// Alternate between the primary and secondary arrays
	uint32_t *usedArr = rs->primaryArr;
	if(whichSlot % 2 == 1) {
		usedArr = rs->secondaryArr;
	}

	int32_t shiftAmount = (rs->numSlots - (whichSlot + 1)) * rs->radixSize;
	mask = rs->mask << shiftAmount;

	uint32_t idx = (uint32_t)start;
	uint32_t maskedValue;
	while(idx < (uint32_t)end) {
		maskedValue = (usedArr[idx] & mask) >> shiftAmount;
		histogram[maskedValue].numElements++;
		idx++;
	}

	uint32_t currentIndex = (uint32_t)start;
	for(idx = 0; idx < rs->numRadixValues; idx++) {
		histogram[idx].radixValue = idx;
		histogram[idx].startIndex = currentIndex;
		histogram[idx].currentIndex = currentIndex;
		//fprintf(stdout, "Hist Slot %d Radix %.2x Index: %d\n", whichSlot, idx, currentIndex);
		currentIndex += histogram[idx].numElements;
	}
}
#elif defined(__aarch64__) || defined(_M_ARM64)
void populateHistogramSIMD(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end) {
	uint32_t mask;

	// Alternate between the primary and secondary arrays
	uint32_t *usedArr = rs->primaryArr;
	if(whichSlot % 2 == 1) {
		usedArr = rs->secondaryArr;
	}

	int32_t simdSize = 4;
	int32_t shiftAmount = (rs->numSlots - (whichSlot + 1)) * rs->radixSize;
	mask = rs->mask << shiftAmount;

	uint32_t maskArr[] = {mask, mask, mask, mask};
	int32_t shiftArr[] = {-shiftAmount, -shiftAmount, -shiftAmount, -shiftAmount};
	uint32x4_t simdMask = vld1q_u32(&(maskArr[0]));
	int32x4_t simdShift = vld1q_s32(&(shiftArr[0]));
	uint32x4_t elements;

	// To extract an unsigned 32-bit integer from lane 0 of a NEON vector
	// result = vget_lane_u32(vec64a, 0)
	int32_t elementsToProcess = end - start;
	int32_t idx = start;
	while(elementsToProcess >= simdSize) {
		elements = vld1q_u32(&(usedArr[idx]));

		// Mask each element
		elements = vandq_u32(elements, simdMask);

		// Shift each element
		//elements = vshrq_n_u32(elements, shiftAmount);
		elements = vshlq_u32(elements, simdShift);

		// Update the histogram
		histogram[vgetq_lane_u32(elements, 0)].numElements++;
		histogram[vgetq_lane_u32(elements, 1)].numElements++;
		histogram[vgetq_lane_u32(elements, 2)].numElements++;
		histogram[vgetq_lane_u32(elements, 3)].numElements++;

		elementsToProcess -= simdSize;
		idx += simdSize;
	}
	if(elementsToProcess > 0) {
		// Do the rest like normal
		uint32_t maskedValue;
		while(idx < end) {
			maskedValue = (usedArr[idx] & mask) >> shiftAmount;
			histogram[maskedValue].numElements++;
			idx++;
		}
	}

	uint32_t currentIndex = start;
	for(idx = 0; idx < rs->numRadixValues; idx++) {
		histogram[idx].radixValue = idx;
		histogram[idx].startIndex = currentIndex;
		histogram[idx].currentIndex = currentIndex;
		//fprintf(stdout, "Hist Slot %d Radix %.2x Index: %d\n", whichSlot, idx, currentIndex);
		currentIndex += histogram[idx].numElements;
	}
}
#endif

void sortSlot(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end) {
	uint32_t mask;

	// Alternate between the primary and secondary arrays
	uint32_t *srcArr = rs->primaryArr;
	uint32_t *tgtArr = rs->secondaryArr;
	if(whichSlot % 2 == 1) {
		srcArr = rs->secondaryArr;
		tgtArr = rs->primaryArr;
	}

	int32_t numSlots = (sizeof(int32_t) * 8) / rs->radixSize;
	int32_t shiftAmount = (numSlots - (whichSlot + 1)) * rs->radixSize;
	mask = rs->mask << shiftAmount;

	int32_t idx = start;
	int32_t maskedValue;
	while(idx < end) {
		maskedValue = (srcArr[idx] & mask) >> shiftAmount;
		tgtArr[histogram[maskedValue].currentIndex] = srcArr[idx];
		// Update the histogram for any future element with the same masked value
		histogram[maskedValue].currentIndex++;
		idx++;
	}
}

#if defined(__x86_64__) || defined(_M_X64)
void sortSlotSIMD(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end) {
	uint32_t mask;

	// Alternate between the primary and secondary arrays
	uint32_t *srcArr = rs->primaryArr;
	uint32_t *tgtArr = rs->secondaryArr;
	if(whichSlot % 2 == 1) {
		srcArr = rs->secondaryArr;
		tgtArr = rs->primaryArr;
	}

	int32_t numSlots = (sizeof(int32_t) * 8) / rs->radixSize;
	int32_t shiftAmount = (numSlots - (whichSlot + 1)) * rs->radixSize;
	mask = rs->mask << shiftAmount;

	int32_t idx = start;
	int32_t maskedValue;
	while(idx < end) {
		maskedValue = (srcArr[idx] & mask) >> shiftAmount;
		tgtArr[histogram[maskedValue].currentIndex] = srcArr[idx];
		// Update the histogram for any future element with the same masked value
		histogram[maskedValue].currentIndex++;
		idx++;
	}
}
#elif defined(__aarch64__) || defined(_M_ARM64)
void sortSlotSIMD(radixStruct *rs, histogramRadix *histogram, int32_t whichSlot, int32_t start, int32_t end) {
	uint32_t mask;

	// Alternate between the primary and secondary arrays
	uint32_t *srcArr = rs->primaryArr;
	uint32_t *tgtArr = rs->secondaryArr;
	if(whichSlot % 2 == 1) {
		srcArr = rs->secondaryArr;
		tgtArr = rs->primaryArr;
	}

	int32_t numSlots = (sizeof(int32_t) * 8) / rs->radixSize;
	int32_t shiftAmount = (numSlots - (whichSlot + 1)) * rs->radixSize;
	mask = rs->mask << shiftAmount;

	int32_t idx = start;
	int32_t maskedValue;
	while(idx < end) {
		maskedValue = (srcArr[idx] & mask) >> shiftAmount;
		//fprintf(stdout, "Value: %.8x Masked Value: %.8x New Index: %d\n", srcArr[idx], maskedValue, histogram[maskedValue].currentIndex);
		tgtArr[histogram[maskedValue].currentIndex] = srcArr[idx];
		// Update the histogram for any future element with the same masked value
		histogram[maskedValue].currentIndex++;
		idx++;
	}
}
#endif

void *invokeRadixWithThread(void *arguments) {
	threadStruct *threadInfo = (threadStruct*)arguments;

	/*
		Recursively sort each radix bucket.  There is no need to spawn more threads
			because each thread is contains approximately the same number of elements
			to be sorted.
	*/
	for(uint32_t eIdx = 0; eIdx < threadInfo->radixCount; eIdx++) {
		// Allocate histogram
		histogramRadix *histogram = (histogramRadix*)malloc(sizeof(histogramRadix) * threadInfo->rs->numRadixValues);
		if(histogram) {
			memset(histogram, 0, sizeof(histogramRadix) * threadInfo->rs->numRadixValues);
		}

		// Populate histogram
		threadInfo->histFunc(threadInfo->rs, histogram, threadInfo->slotNumber + 1,
			threadInfo->radiiToProcess[eIdx]->startIndex,
			threadInfo->radiiToProcess[eIdx]->startIndex + threadInfo->radiiToProcess[eIdx]->numElements);

		// Sort based on histogram
		threadInfo->sortFunc(threadInfo->rs, histogram, threadInfo->slotNumber + 1,
			threadInfo->radiiToProcess[eIdx]->startIndex,
			threadInfo->radiiToProcess[eIdx]->startIndex + threadInfo->radiiToProcess[eIdx]->numElements);

		if(threadInfo->slotNumber < threadInfo->rs->numSlots - 2) {
			// For each populated radix in histogram, recurse until all slots handled
			for(uint32_t hIdx = 0; hIdx < threadInfo->rs->numRadixValues; hIdx++) {
				if(histogram[hIdx].numElements > 0) {
					threadStruct ts;
					memset(&ts, 0, sizeof(threadStruct));
					ts.histFunc = threadInfo->histFunc;
					ts.sortFunc = threadInfo->sortFunc;
					ts.rs = threadInfo->rs;
					ts.slotNumber = threadInfo->slotNumber + 1;
					ts.threadNumber = threadInfo->threadNumber;
					ts.totalElements = histogram[hIdx].numElements;
					ts.radixCount = 1;
					ts.radiiToProcess = (histogramRadix**)malloc(sizeof(histogramRadix*) * 1);
					if(ts.radiiToProcess) {
						ts.radiiToProcess[0] = &histogram[hIdx];
						(void)invokeRadixWithThread((void*)&ts);
						free(ts.radiiToProcess);
						ts.radiiToProcess = NULL;
					}
				}
			}
		}

		// Deallocate histogram
		free(histogram);
		histogram = NULL;
	}

	return NULL;
}

void executeQS(radixStruct *rs) {
	struct timespec startTime;
	struct timespec endTime;
	timespec_get(&startTime, TIME_UTC);

	qs(rs->primaryArr, 0, rs->numElements - 1);

	timespec_get(&endTime, TIME_UTC);

	int64_t startNanoseconds = (int64_t)startTime.tv_sec * 1000000000L + startTime.tv_nsec;
	int64_t endNanoseconds = (int64_t)endTime.tv_sec * 1000000000L + endTime.tv_nsec;

	char startBuff[100];
	char endBuff[100];
	strftime(startBuff, sizeof startBuff, "%FT%T", gmtime(&startTime.tv_sec));
	strftime(endBuff, sizeof endBuff, "%FT%T", gmtime(&endTime.tv_sec));

	fprintf(stdout, "QUICKSORT|%s.%06ld UTC|%s.%06ld UTC|%f (milliseconds)\n",
			startBuff, startTime.tv_nsec, endBuff, endTime.tv_nsec,
			((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);
}

void execute(radixStruct *rs, int32_t numThreads, int32_t style) {
	struct timespec startTime;
	struct timespec endTime;
	timespec_get(&startTime, TIME_UTC);

	threadStruct *threadArr = (threadStruct*)malloc(sizeof(threadStruct) * numThreads);
	if(threadArr) {
		memset(threadArr, 0, sizeof(threadStruct) * numThreads);
	} else {
		fprintf(stderr, "Unable to allocate memory for %d thread structures\n", numThreads);
		return;
	}

	for(int32_t idx = 0; idx < numThreads; idx++) {
		threadArr[idx].radiiToProcess = (histogramRadix**)malloc(sizeof(histogramRadix*) * rs->numRadixValues);
		if(threadArr[idx].radiiToProcess) {
			memset(threadArr[idx].radiiToProcess, 0, sizeof(histogramRadix*) * rs->numRadixValues);
		} else {
			fprintf(stderr, "Unable to allocate memory for thread elements\n");
			return;
		}
		
		threadArr[idx].rs = rs;
		if(style == STYLE_BASIC) {
			threadArr[idx].histFunc = &populateHistogram;
			threadArr[idx].sortFunc = &sortSlot;
		} else {
			threadArr[idx].histFunc = &populateHistogramSIMD;
			threadArr[idx].sortFunc = &sortSlot;
		}
		
		threadArr[idx].threadNumber = idx;
	}


	// Allocate the histogram array
	histogramRadix *histogram = (histogramRadix*)malloc(sizeof(histogramRadix) * rs->numRadixValues);
	if(histogram) {
		memset(histogram, 0, sizeof(histogramRadix) * rs->numRadixValues);
	} else {
		fprintf(stderr, "Unable to allocate memory for the histogram\n");
		return;
	}

	// Populate the histogram for the first MSB slot
	populateHistogram(rs, histogram, 0, 0, rs->numElements);

	// Do the sort for this slot
	sortSlot(rs, histogram, 0, 0, rs->numElements);

	// Allocate radix values to threads    
	int32_t elementsPerThread = rs->numElements / numThreads;
	int32_t leftovers = rs->numElements % numThreads;

	int32_t elementCount = elementsPerThread + leftovers;

	int32_t currentThread = 0;
	int32_t processingIdx = 0;
	for(uint32_t idx = 0; idx < rs->mask; idx++) {
		if(histogram[idx].numElements > 0) {
			threadArr[currentThread].radiiToProcess[processingIdx] = &histogram[idx];
			threadArr[currentThread].radixCount++;
			processingIdx++;
			elementCount -= histogram[idx].numElements;

			if(elementCount <= 0) {
				elementCount = elementsPerThread;
				currentThread++;
				if(currentThread == numThreads) { currentThread = numThreads - 1; }
				processingIdx = 0;
			}
		}
	}

	// Start the threads working
	for(int32_t idx = 0; idx < numThreads; idx++) {
		if(pthread_create(&(threadArr[idx].thread), NULL, invokeRadixWithThread, &threadArr[idx]) != 0) {
			fprintf(stderr, "Unable to start thread %d\n", idx);
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

	fprintf(stdout, "%s|%d threads|%d elements|%s.%09ld UTC|%s.%09ld UTC|%f (milliseconds)\n",
		style == STYLE_BASIC ? "BASIC" : "SIMD",
		numThreads, rs->numElements,
		startBuff, startTime.tv_nsec, endBuff, endTime.tv_nsec,
		((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	for(int32_t idx = 0; idx < numThreads; idx++) {
		if(threadArr[idx].radiiToProcess) {
			free(threadArr[idx].radiiToProcess);
		}
	}
	free(histogram);
	free(threadArr);
}

int testSorted(radixStruct *rs) {
	if(rs) {
		for(int32_t i = 1; i < rs->numElements; i++) {
			if(rs->primaryArr[i] < rs->primaryArr[i - 1]) {
				fprintf(stderr, "Current: %.8x Previous: %.8x\n", rs->primaryArr[i], rs->primaryArr[i - 1]);
				return -1;
			}
		}
		return 0;
	}
	return -1;
}

int availableCores() {
	return sysconf(_SC_NPROCESSORS_CONF);
}

void qs(uint32_t *arr, int32_t low, int32_t high) {
	if(arr) {
		if(low >= high || low < 0) return;

		int32_t pIdx = qsp(arr, low, high);

		qs(arr, low, pIdx - 1);
		qs(arr, pIdx + 1, high);
	}
}

int32_t qsp(uint32_t *arr, int32_t low, int32_t high) {
	uint32_t pivot = arr[high];

	int32_t idx = low - 1;

	for(int32_t j = low; j < high; j++) {
		if(arr[j] <= pivot) {
			idx++;
			uint32_t temp = arr[idx];
			arr[idx] = arr[j];
			arr[j] = temp;
		}
	}
	idx++;
	uint32_t temp = arr[idx];
	arr[idx] = arr[high];
	arr[high] = temp;
	return idx;
}
