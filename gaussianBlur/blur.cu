#include "blur.h"
#include "blur.c"

__global__ void blurKernel(int imgSize, int convSize, int *imageArr, int *outputArr, float *convoArr) {
	int rowIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int colIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int rowStride = gridDim.x * blockDim.x;
	int colStride = gridDim.y * blockDim.y;

	for(int row = rowIdx; row < imgSize; row += rowStride) {
		if(row >= convSize && row < imgSize - convSize) {
			for(int col = colIdx; col < imgSize; col += colStride) {
				if(col >= convSize && col < imgSize - convSize) {
					// Blur this pixel
					int cRow = 0;
					int elementCount = 0;
					float accumulator = 0.0;

					for(int rowIter = row - convSize; rowIter <= row + convSize; rowIter++) {
						int rowOffset = rowIter * imgSize;
						int cRowOffset = cRow * (convSize * 2 + 1);
						int cCol = 0;
						for(int colIter = col - convSize; colIter <= col + convSize; colIter++) {
							accumulator += (float)imageArr[rowOffset + colIter] * convoArr[cRowOffset + cCol];
							elementCount++;
							cCol++;
						}
						cRow++;
					}
					outputArr[row * imgSize + col] = ceil(accumulator / elementCount);
				}
			}
		}
	}
}

void executeBlurCuda(blurStruct *bs) {
	struct timespec startTime;
	struct timespec endTime;

	cudaError_t err;

	int deviceId;
	err = cudaGetDevice(&deviceId);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Get Device Id Error: %s\n", cudaGetErrorString(err)); }

	// Allocate GPU memory
	size_t imageSize = sizeof(int32_t) * bs->imageSize * bs->imageSize;
	int convolutionElementSide = bs->convolutionDistance * 2 + 1;
	size_t convSize = sizeof(float) * convolutionElementSide * convolutionElementSide;

	int *imageArr;
	int *outputArr;
	float *convoArr;

	err = cudaMalloc(&imageArr, imageSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Image Array Allocation Error: %s\n", cudaGetErrorString(err)); }
	err = cudaMalloc(&outputArr, imageSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Output Array Allocation Error: %s\n", cudaGetErrorString(err)); }
	err = cudaMalloc(&convoArr, convSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Convolution Array Allocation Error: %s\n", cudaGetErrorString(err)); }

	err = cudaMemset(outputArr, 0, imageSize);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Output Array Memset Error: %s\n", cudaGetErrorString(err)); }

	timespec_get(&startTime, TIME_UTC);

	// Copy data to the GPU
	err = cudaMemcpy(imageArr, (int*)bs->image, imageSize, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy imageArr Error: %s\n", cudaGetErrorString(err)); }
	err = cudaMemcpy(convoArr, (float*)bs->convolutionMatrix, convSize, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy convoArr Error: %s\n", cudaGetErrorString(err)); }

	// Execute...
	size_t threadsPerBlock = 32;
	size_t numBlocks = (bs->imageSize + threadsPerBlock - 1) / threadsPerBlock;
	if(numBlocks > 64) { numBlocks = 64; }

	blurKernel<<<dim3(numBlocks,numBlocks,1), dim3(threadsPerBlock,threadsPerBlock,1)>>>(bs->imageSize, bs->convolutionDistance, imageArr, outputArr, convoArr);
	err = cudaGetLastError();
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Execution Error: %s\n", cudaGetErrorString(err)); }

	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Synchronize Error: %s\n", cudaGetErrorString(err)); }

	// Copy data back to CPU
	err = cudaMemcpy((unsigned int*)bs->simdImage, outputArr, imageSize, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy outputArr Error: %s\n", cudaGetErrorString(err)); }

	timespec_get(&endTime, TIME_UTC);

	// Free memory
	err = cudaFree(imageArr);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Image Array Free Error: %s\n", cudaGetErrorString(err)); }
	err = cudaFree(outputArr);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Output Array Free Error: %s\n", cudaGetErrorString(err)); }
	err = cudaFree(convoArr);
	if(err != cudaSuccess) { fprintf(stderr, "CUDA Convolution Array Free Error: %s\n", cudaGetErrorString(err)); }

	int64_t startNanoseconds = (int64_t)startTime.tv_sec * 1000000000L + startTime.tv_nsec;
	int64_t endNanoseconds = (int64_t)endTime.tv_sec * 1000000000L + endTime.tv_nsec;
	char startBuff[100];
	char endBuff[100];
	strftime(startBuff, sizeof startBuff, "%FT%T", gmtime(&startTime.tv_sec));
	strftime(endBuff, sizeof endBuff, "%FT%T", gmtime(&endTime.tv_sec));
	fprintf(stdout, "CudaGPU|1 threads|%f stdDev|%d imageSize|%s.%09ld UTC|%s.%09ld UTC|%f (milliseconds)\n",
		bs->standardDeviation, bs->imageSize,
		startBuff, startTime.tv_nsec, endBuff, endTime.tv_nsec,
		((double)endNanoseconds - (double)startNanoseconds) / 1000000.0);

	// Verify that the blur matches the non-cuda version
	/*
	for(int row = 0; row < bs->imageSize; row++) {
		size_t rowOffset = row * bs->imageSize;
		for(int col = 0; col < bs->imageSize; col++) {
			if(abs(bs->newImage[rowOffset + col] - bs->simdImage[rowOffset + col]) > 1) {
				fprintf(stderr, "[%d,%d] %d vs %d\n", col, row, bs->newImage[rowOffset + col], bs->simdImage[rowOffset + col]);
			}
		}
	}
	*/
}

int main(int argc, char **argv) {
	if(argc != 3) {
		fprintf(stderr, "Usage: %s <standardDeviation> <imageSize>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	double stdDev = atof(argv[1]);
	int32_t imgSize = (int32_t)atoi(argv[2]);

	//fprintf(stdout, "Creating data structure\n%f standard deviation\n%d image Size\n", stdDev, imgSize);

	blurStruct bs;

	if(initBlurStruct(&bs, stdDev, imgSize) != 0) {
        	fprintf(stderr, "Unable to initialize blur structure\n");
		exit(-1);
	}

	//fprintf(stdout, "\nPopulating data structure with random image data\n\n");
	populateBlurStruct(&bs);
	//execute(&bs, 1, STYLE_BASIC);

	sleep(2);
	executeBlurCuda(&bs);
	sleep(2);

	freeBlurStruct(&bs);

	return 0;
}
