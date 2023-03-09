#include "radix.h"

int main(int argc, char **argv) {
	if(argc < 2) {
		fprintf(stderr, "Usage: %s <numElements>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	int32_t numElements = (int32_t)atoi(argv[1]);

	fprintf(stdout, "Creating data structure with %d elements\n", numElements);

	radixStruct rs;

	if(initRadixStruct(&rs, numElements, 8) != 0) {
		fprintf(stderr, "Unable to initialize radix structure\n");
		exit(-1);
	}

	fprintf(stdout, "\nPopulating data structure with random data\n\n");
	populateRadixStruct(&rs);

	sleep(1);

	executeQS(&rs);

	freeRadixStruct(&rs);
	
	return 0;
}
