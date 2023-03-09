#include "distance.h"

int testBasic() {
	distanceStruct ds;
	if(initDistanceStruct(&ds, 1, 1, 8) != 0) {
		fprintf(stderr, "Unable to initialize the distance structure\n");
		return -1;
	}

	setElementValue(&ds, 0, 0, 1);
	setElementValue(&ds, 0, 1, 1);
	setElementValue(&ds, 0, 2, 1);
	setElementValue(&ds, 0, 3, 1);
	setElementValue(&ds, 0, 4, 1);
	setElementValue(&ds, 0, 5, 1);
	setElementValue(&ds, 0, 6, 1);
	setElementValue(&ds, 0, 7, 1);

	setClusterValue(&ds, 0, 0, 2);
	setClusterValue(&ds, 0, 1, 2);
	setClusterValue(&ds, 0, 2, 2);
	setClusterValue(&ds, 0, 3, 2);
	setClusterValue(&ds, 0, 4, 2);
	setClusterValue(&ds, 0, 5, 2);
	setClusterValue(&ds, 0, 6, 2);
	setClusterValue(&ds, 0, 7, 2);

	int distance = distanceBetween(&ds, 0, 0);
	if(distance != 8) {
		fprintf(stderr, "Invalid distance: %d vs 8\n", distance);
		return -1;
	}

	freeDistanceStruct(&ds);

	return 0;
}

int testBasicSIMD() {
	distanceStruct ds;
	if(initDistanceStruct(&ds, 1, 1, 24) != 0) {
		fprintf(stderr, "Unable to initialize the distance structure\n");
		return -1;
	}

	populateDistanceStruct(&ds);

	int distance = distanceBetween(&ds, 0, 0);
	int distanceSIMD = distanceBetweenSIMD(&ds, 0, 0);

	if(distanceSIMD != distance) {
		fprintf(stderr, "Non-matching distance: %d (Basic) vs %d (SIMD)\n", distance, distanceSIMD);
		return -1;
	}

	freeDistanceStruct(&ds);

	return 0;
}

int testComparingBasicToSIMD() {
	distanceStruct ds;
	int retVal = 0;
	int32_t numElements = 1000;
	int32_t numClusters = 10;
	int32_t numDimensions = 64;

	if(initDistanceStruct(&ds, numElements, numClusters, numDimensions) != 0) {
		fprintf(stderr, "Unable to initialize the distance structure\n");
		return -1;
	}

	populateDistanceStruct(&ds);

	// Calculate using the basic style
	for(int cIdx = 0; cIdx < numClusters; cIdx++) {
		for(int eIdx = 0; eIdx < numElements; eIdx++) {
			if(distanceBetween(&ds, cIdx, eIdx) != distanceBetweenSIMD(&ds, cIdx, eIdx)) {
				retVal++;
			}
		}
	}

	freeDistanceStruct(&ds);

	return retVal;
}

int testOddDimensions() {
	distanceStruct ds;
	if(initDistanceStruct(&ds, 1, 1, 6) != 0) {
		fprintf(stderr, "Unable to initialize the distance structure\n");
		return -1;
	}

	setElementValue(&ds, 0, 0, 1);
	setElementValue(&ds, 0, 1, 1);
	setElementValue(&ds, 0, 2, 1);
	setElementValue(&ds, 0, 3, 1);
	setElementValue(&ds, 0, 4, 1);
	setElementValue(&ds, 0, 5, 1);

	setClusterValue(&ds, 0, 0, 2);
	setClusterValue(&ds, 0, 1, 2);
	setClusterValue(&ds, 0, 2, 2);
	setClusterValue(&ds, 0, 3, 2);
	setClusterValue(&ds, 0, 4, 2);
	setClusterValue(&ds, 0, 5, 2);

	int distance = distanceBetween(&ds, 0, 0);
	if(distance != 6) {
		fprintf(stderr, "Invalid distance: %d vs 6\n", distance);
		return -1;
	}

	freeDistanceStruct(&ds);

	return 0;
}

int testThreeElements() {
	distanceStruct ds;
	if(initDistanceStruct(&ds, 3, 1, 8) != 0) {
		fprintf(stderr, "Unable to initialize the distance structure\n");
		return -1;
	}

	setClusterValue(&ds, 0, 0, 2);
	setClusterValue(&ds, 0, 1, 2);
	setClusterValue(&ds, 0, 2, 2);
	setClusterValue(&ds, 0, 3, 2);
	setClusterValue(&ds, 0, 4, 2);
	setClusterValue(&ds, 0, 5, 2);
	setClusterValue(&ds, 0, 6, 2);
	setClusterValue(&ds, 0, 7, 2);

	for(int i = 0; i < 3; i++) {
		setElementValue(&ds, i, 0, 1);
		setElementValue(&ds, i, 1, 1);
		setElementValue(&ds, i, 2, 1);
		setElementValue(&ds, i, 3, 1);
		setElementValue(&ds, i, 4, 1);
		setElementValue(&ds, i, 5, 1);
		setElementValue(&ds, i, 6, 1);
		setElementValue(&ds, i, 7, 1);

		int distance = distanceBetween(&ds, 0, i);
		if(distance != 8) {
			fprintf(stderr, "Invalid distance: %d vs 8\n", distance);
			return -1;
		}
	}

	freeDistanceStruct(&ds);

	return 0;
}

int main(int argc, char **argv) {
	int testStatus = 0;

	if(testBasic() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "testBasic");
	}
	if(testBasicSIMD() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "testBasicSIMD");
	}
	if(testOddDimensions() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "testOddDimensions");
	}
	if(testThreeElements() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "testThreeElements");
	}
	if(testComparingBasicToSIMD() != 0) {
		testStatus++;
		fprintf(stderr, "FAIL: %s\n", "testComparingBasicToSIMD");
	}

	if(testStatus == 0) {
		fprintf(stdout, "All tests successful\n");
	} else {
		fprintf(stdout, "Some tests failed\n");
	}
}
