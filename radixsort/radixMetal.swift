import Foundation
import Metal

if CommandLine.argc != 2 {
	print("Usage: <\(CommandLine.arguments[0])> [NumElements]")
} else {
	let numElements = Int32(CommandLine.arguments[1]) ?? 30

	var rs: radixStruct = radixStruct()

	//print("Creating data structure with \(numElements) elements")
	if initRadixStruct(&rs, numElements, 8) != 0 {
		print("Error when initializing the radix struct")
	}

	//print("Populating data structure with random data");
	populateRadixStruct(&rs)

	sleep(2)
	executeMetal(&rs)
	sleep(2)

	freeRadixStruct(&rs)

}

func executeMetal(_ rs: inout radixStruct) {
	guard let device = MTLCreateSystemDefaultDevice() else {
		fatalError( "Failed to get the system's default Metal device." )
	}
	guard let commandQueue = device.makeCommandQueue() else {
		fatalError( "Failed to make command queue." )
	}
	print("Metal Device: \(device.name)")
	//print("Recommended Working Set Size: \(device.recommendedMaxWorkingSetSize)")
	//print("Max Buffer Length: \(device.maxBufferLength)")

	let kernelSrc = """
		#include <metal_stdlib>
		#include <metal_atomic>
		//#include <string.h>

		using namespace metal;

		#define RADIX_COUNT 256
		#define RADIX_MASK 255
		#define RADIX_SHIFT 8
		#define RADIX_PASSES 4

		kernel void msb(device const int32_t* shapeArr,
						device uint32_t *firstArr,
						device uint32_t *secondArr,
						device atomic_uint *histogramArr,
						uint elementNumber [[thread_index_in_threadgroup]],
						uint stride [[threads_per_threadgroup]]) {
			
			uint32_t shift = RADIX_SHIFT * (RADIX_PASSES - 1);
			uint32_t mask = RADIX_MASK << shift;
			uint32_t atomicInc = 1;
			int numRadix = RADIX_COUNT;
			int numElements = shapeArr[0];

			for(int i = elementNumber; i < numElements; i += stride) {
				uint32_t maskedValue = (firstArr[i] & mask) >> shift;
				atomic_fetch_add_explicit(&histogramArr[maskedValue], atomicInc, memory_order_relaxed);
			}

			threadgroup_barrier(mem_flags::mem_none);

			if(elementNumber == 0) {
				uint32_t temp = 0;
				uint32_t count = 0;
				for(int i = 0; i < numRadix; i++) {
					temp = atomic_load_explicit(&histogramArr[i], memory_order_relaxed);
					if(temp > 0) { atomic_store_explicit(&histogramArr[i], count, memory_order_relaxed); }
					count += temp;
				}
			}

			threadgroup_barrier(mem_flags::mem_none);

			for(int i = elementNumber; i < numElements; i += stride) {
				uint32_t maskedValue = (firstArr[i] & mask) >> shift;
				uint32_t newIndex = atomic_fetch_add_explicit(&histogramArr[maskedValue], atomicInc, memory_order_relaxed);
				secondArr[newIndex] = firstArr[i];
			}
		}

		kernel void msb2(device const int32_t* shapeArr,
						device uint32_t *firstArr,
						device uint32_t *secondArr,
						device uint32_t *histogramArr,
						threadgroup atomic_uint *localHist [[threadgroup(0)]],
						uint elementNumber [[thread_index_in_threadgroup]],
						uint stride [[threads_per_threadgroup]],
						uint radixValue [[threadgroup_position_in_grid]]) {
	
			// Do not bother to do anything unless there are elements in this radix value
			if(histogramArr[radixValue] > 0) {
				atomic_store_explicit(&localHist[elementNumber], 0, memory_order_relaxed);
				threadgroup_barrier(mem_flags::mem_none);
				
				int startIndex = 0;
				int endIndex = histogramArr[radixValue];
				
				if(radixValue > 0) {
					int temp = radixValue - 1;
					while(temp > 0 && histogramArr[temp] == 0) { temp--; }
					startIndex = histogramArr[temp];
				}
				
				// At this point, run MSB on the elements from startIndex to endIndex (not including endIndex)
				device uint32_t *srcArr = secondArr;
				device uint32_t *tgtArr = firstArr;
				uint32_t shift = RADIX_SHIFT * (RADIX_PASSES - 2);
				uint32_t mask = RADIX_MASK << shift;
				uint32_t atomicInc = 1;
				uint32_t groupElements = endIndex - startIndex;
				uint32_t maskedValue;
				uint32_t newIndex;
				
				for(uint idx = elementNumber; idx < groupElements; idx += stride) {
					maskedValue = (srcArr[idx + startIndex] & mask) >> shift;
					atomic_fetch_add_explicit(&localHist[maskedValue], atomicInc, memory_order_relaxed);
				}
				threadgroup_barrier(mem_flags::mem_none);
				
				if(elementNumber == 0) {
					uint32_t temp = 0;
					uint32_t count = startIndex;
					for(int idx = 0; idx < RADIX_COUNT; idx++) {
						temp = atomic_load_explicit(&localHist[idx], memory_order_relaxed);
						if(temp > 0) { atomic_store_explicit(&localHist[idx], count, memory_order_relaxed);}
						count += temp;
					}
				}
				
				threadgroup_barrier(mem_flags::mem_none);
				
				// Save for later...
				unsigned int lsbStart = atomic_load_explicit(&localHist[elementNumber], memory_order_relaxed);
				
				for(uint idx = elementNumber; idx < groupElements; idx += stride) {
					maskedValue = (srcArr[idx + startIndex] & mask) >> shift;
					newIndex = atomic_fetch_add_explicit(&localHist[maskedValue], atomicInc, memory_order_relaxed);
					tgtArr[newIndex] = srcArr[idx + startIndex];
				}
				threadgroup_barrier(mem_flags::mem_none);
				
				// Do LSB for pass 3 and 4
				// Each thread handles the entire radix range of a radix subset
				uint32_t lsbEnd = atomic_load_explicit(&localHist[elementNumber], memory_order_relaxed);
				uint32_t subsetSize = lsbEnd - lsbStart;
				
				// Only do work if there is more than 1 element to sort
				if(subsetSize > 1) {
					//unsigned int lsbHist[RADIX_COUNT];
					shift = 0;
					mask = RADIX_MASK;
					
					for(unsigned int pIdx = 0; pIdx < 2; pIdx++) {
						uint32_t lsbHist[RADIX_COUNT] = {0};
						device uint32_t *tempArr = srcArr;
						srcArr = tgtArr;
						tgtArr = tempArr;
						
						//memset(&lsbHist, 0, sizeof(unsigned int) * RADIX_COUNT);
						
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
	"""
	
	let srcLibrary: MTLLibrary
	let msbFunction: MTLFunction?
	let msb2Function: MTLFunction?
	let computePipelineMSB: MTLComputePipelineState
	let computePipelineMSB2: MTLComputePipelineState
	do {
		srcLibrary = try device.makeLibrary(source: kernelSrc, options: nil)
	} catch {
		fatalError( "Failed to create srcLibrary." )
	}

	msbFunction = srcLibrary.makeFunction(name: "msb")
	msb2Function = srcLibrary.makeFunction(name: "msb2")

	do {
		computePipelineMSB = try device.makeComputePipelineState(function: msbFunction!)
		computePipelineMSB2 = try device.makeComputePipelineState(function: msb2Function!)
	} catch {
		fatalError( "Failed to create computePipelineMSB." )
	}

	// Start counting the time
	//let startTime = Date()

	var shapeData: [Int32] = []
	shapeData.reserveCapacity(1)
	shapeData.append(rs.numElements)

	let shapeBufferSize = MemoryLayout<Int32>.stride * 1
	guard let shapeBuffer = device.makeBuffer(bytes: shapeData, length: shapeBufferSize, options: .storageModeShared) else {
	fatalError( "Failed to make shapeBuffer." )
	}

	let bufferSize: Int = MemoryLayout<UInt32>.stride * Int(rs.numElements)
	// Copy the elements
	guard let primaryBuffer = device.makeBuffer(bytes: rs.primaryArr, length: bufferSize, options: .storageModeShared) else {
		fatalError( "Failed to make primaryBuffer" )
	}
	// No need to copy elements
	guard let secondaryBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
		fatalError( "Failed to make primaryBuffer" )
	}
	let histSize: Int = MemoryLayout<UInt32>.stride * 256
	// No need to copy elements
	guard let histogramBuffer = device.makeBuffer(length: histSize, options: .storageModeShared) else {
		fatalError( "Failed to make histogramBuffer" )
	}

	// Start counting the time
	let startTime = Date()

	// Start a compute pass
	guard let commandBuffer = commandQueue.makeCommandBuffer() else {
		fatalError( "Failed to make commandBuffer." )
	}
	guard let msbEncoder = commandBuffer.makeComputeCommandEncoder() else {
		fatalError( "Failed to make msbEncoder." )
	}

	msbEncoder.setComputePipelineState(computePipelineMSB)
	msbEncoder.setBuffer(shapeBuffer, offset: 0, index: 0)
	msbEncoder.setBuffer(primaryBuffer, offset: 0, index: 1)
	msbEncoder.setBuffer(secondaryBuffer, offset: 0, index: 2)
	msbEncoder.setBuffer(histogramBuffer, offset: 0, index: 3)

	let msbGridSize = MTLSizeMake(1024, 1, 1)
	let msbThreadGroupSize = MTLSizeMake(1024, 1, 1)

	msbEncoder.dispatchThreads(msbGridSize, threadsPerThreadgroup: msbThreadGroupSize)
	msbEncoder.endEncoding()

	guard let msb2Encoder = commandBuffer.makeComputeCommandEncoder() else {
		fatalError( "Failed to make msb2Encoder." )
	}

	msb2Encoder.setComputePipelineState(computePipelineMSB2)
	msb2Encoder.setThreadgroupMemoryLength(MemoryLayout<UInt32>.stride * 256, index: 0)
	msb2Encoder.setBuffer(shapeBuffer, offset: 0, index: 0)
	msb2Encoder.setBuffer(primaryBuffer, offset: 0, index: 1)
	msb2Encoder.setBuffer(secondaryBuffer, offset: 0, index: 2)
	msb2Encoder.setBuffer(histogramBuffer, offset: 0, index: 3)

	let msb2GridSize = MTLSizeMake(65536, 1, 1)
	let msb2ThreadGroupSize = MTLSizeMake(256, 1, 1)

	msb2Encoder.dispatchThreads(msb2GridSize, threadsPerThreadgroup: msb2ThreadGroupSize)
	msb2Encoder.endEncoding()

	commandBuffer.commit()
	commandBuffer.waitUntilCompleted()
	

	// Stop counting the time
	let stopTime = Date()
	let iterationTime = stopTime.timeIntervalSince(startTime)
	let formatter = DateFormatter()
	formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSS"
	//print("Duration (milliseconds) \(iterationTime * 1000)")
	print("MetalGPU|1 threads|\(rs.numElements) elements|\(formatter.string(from: startTime))|\(formatter.string(from: stopTime))|\(iterationTime * 1000) (milliseconds)")

	// Verify that the data is the same as the basic output
	/*
	var outputPtr = primaryBuffer.contents()
	var elementValue: UInt32
	var prevValue: UInt32
	var notSorted: Bool
	prevValue = 0
	notSorted = false
	for _ in 0..<rs.numElements {
		elementValue = outputPtr.load(as: UInt32.self)
		if elementValue < prevValue { notSorted = true }
		outputPtr = outputPtr + MemoryLayout<UInt32>.stride
		prevValue = elementValue
	}
	if notSorted { print("Not Sorted") } else { print("Sorted!") }
	*/
}