import Foundation
import Metal

if CommandLine.argc != 5 {
	print("Usage: <\(CommandLine.arguments[0])> [NumPoints] [NumClusters] [NumDimensions] [NumIterations]")
} else {
	let numElements = Int32(CommandLine.arguments[1]) ?? 0
	let numClusters = Int32(CommandLine.arguments[2]) ?? 0
	let numDimensions = Int32(CommandLine.arguments[3]) ?? 0
	let numIterations = Int32(CommandLine.arguments[4]) ?? 1

	var ds: distanceStruct = distanceStruct()

	print("Creating data structure with \(numElements) elements, \(numClusters) clusters and \(numDimensions) dimensions")
	if initDistanceStruct(&ds, numElements, numClusters, numDimensions) != 0 {
		print("Error when initializing the distance struct")
	}

	print("Populating data structure with random data");
	populateDistanceStruct(&ds)

	//execute(&ds, 1, numIterations, STYLE_BASIC)

	sleep(2)
	executeMetal(&ds, numIterations: numIterations)
	sleep(2)

	/*
	var simdDistance: Int64
	var basicDistance: Int64
	var anyError = false
	for cIdx in 0..<ds.numClusters {
		for eIdx in 0..<ds.numElements {
			basicDistance = distanceBetween(&ds, cIdx, eIdx)
			simdDistance = distanceBetweenSIMD(&ds, cIdx, eIdx)
			if simdDistance != basicDistance {
				print("Basic: \(basicDistance) SIMD: \(simdDistance)")
				anyError = true
			}
		}
	}
	if anyError == true {
		print("SIMD and Basic did not resolve to same values")
	} else {
		print("No validation errors found")
	}
	*/

	freeDistanceStruct(&ds)

}

func executeMetal(_ ds: inout distanceStruct, numIterations: Int32) {
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
		using namespace metal;

		kernel void distanceMetal(device const int32_t* shapeArr,
									device const int32_t* elementArr,
									device const int32_t* clusterArr,
									device int64_t* distanceArr,
									threadgroup int64_t *sharedArr [[threadgroup(0)]],
									uint3 gridIndex [[threadgroup_position_in_grid]],
									uint threadIndex [[thread_index_in_threadgroup]]) {
			
			int elementNumber = (int)gridIndex[0];
			int clusterNumber = (int)gridIndex[1];
			int dimensionNumber = (int)threadIndex;
			int numElements = (int)shapeArr[0];
			int numDimensions = (int)shapeArr[2];

			int elementOffset = elementNumber * numDimensions;
			int clusterOffset = clusterNumber * numDimensions;
			int64_t difference = 0;
			int64_t distance = 0;

			sharedArr[dimensionNumber] = elementArr[elementOffset + dimensionNumber] - clusterArr[clusterOffset + dimensionNumber];
			sharedArr[dimensionNumber] = sharedArr[dimensionNumber] * sharedArr[dimensionNumber];

			threadgroup_barrier(mem_flags::mem_none);
			
			int dim = numDimensions / 2;
			while(dim >= 1) {
				if(dimensionNumber < dim) {
					sharedArr[dimensionNumber] += sharedArr[dimensionNumber + dim];
				}
				dim = dim / 2;
				threadgroup_barrier(mem_flags::mem_none);
			}

			if(dimensionNumber == 0) {
				distanceArr[clusterNumber * numElements + elementNumber] = sharedArr[dimensionNumber];
			}
		}
	"""
	
	let srcLibrary: MTLLibrary
	let srcDiffFunction: MTLFunction?
	let computePipeline: MTLComputePipelineState
	do {
		srcLibrary = try device.makeLibrary(source: kernelSrc, options: nil)
	} catch {
		fatalError( "Failed to create srcLibrary." )
	}

	srcDiffFunction = srcLibrary.makeFunction(name: "distanceMetal")

	do {
		computePipeline = try device.makeComputePipelineState(function: srcDiffFunction!)
	} catch {
		fatalError( "Failed to create computePipeline." )
	}

	// Start counting the time
	//let startTime = Date()

	var shapeData: [Int32] = []
	shapeData.reserveCapacity(3)
	shapeData.append(ds.numElements)
	shapeData.append(ds.numClusters)
	shapeData.append(ds.numDimensions)

	let shapeBufferSize = MemoryLayout<Int32>.stride * 3
	guard let shapeBuffer = device.makeBuffer(bytes: shapeData, length: shapeBufferSize, options: .storageModeShared) else {
	fatalError( "Failed to make shapeBuffer." )
	}

	let elementBufferSize: Int = MemoryLayout<Int32>.stride * Int(ds.numElements) * Int(ds.numDimensions)
	guard let elementBuffer = device.makeBuffer(bytes: ds.elements, length: elementBufferSize, options: .storageModeShared) else {
		fatalError( "Failed to make elementBuffer" )
	}

	let clusterBufferSize: Int = MemoryLayout<Int32>.stride * Int(ds.numClusters) * Int(ds.numDimensions)
	guard let clusterBuffer = device.makeBuffer(bytes: ds.clusters, length: clusterBufferSize, options: .storageModeShared) else {
		fatalError( "Failed to make clusterBuffer" )
	}

	let distanceBufferSize: Int = MemoryLayout<Int64>.stride * Int(ds.numClusters) * Int(ds.numElements)
	guard let distanceBuffer = device.makeBuffer(length: distanceBufferSize, options: .storageModeShared) else {
		fatalError("Failed to make distanceBuffer.")
	}

	// Start counting the time
	let startTime = Date()

	for _ in 0..<numIterations {
		// Start a compute pass
		guard let commandBuffer = commandQueue.makeCommandBuffer() else {
			fatalError( "Failed to make commandBuffer." )
		}
		guard let distanceEncoder = commandBuffer.makeComputeCommandEncoder() else {
			fatalError( "Failed to make distanceEncoder." )
		}

		distanceEncoder.setComputePipelineState(computePipeline)
		distanceEncoder.setThreadgroupMemoryLength(MemoryLayout<Int64>.stride * Int(ds.numDimensions), index: 0)
		distanceEncoder.setBuffer(shapeBuffer, offset: 0, index: 0)
		distanceEncoder.setBuffer(elementBuffer, offset: 0, index: 1)
		distanceEncoder.setBuffer(clusterBuffer, offset: 0, index: 2)
		distanceEncoder.setBuffer(distanceBuffer, offset: 0, index: 3)

		let totalThreadsX = Int(ds.numElements) * Int(ds.numDimensions)
		let totalThreadsY = Int(ds.numClusters)
		let gridSize = MTLSizeMake(totalThreadsX, totalThreadsY, 1)
		
		let threadsPerGroup = Int(ds.numDimensions)
		let threadGroupSize = MTLSizeMake(threadsPerGroup, 1, 1)

		distanceEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
		distanceEncoder.endEncoding()

		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()
	}
	

	// Stop counting the time
	let stopTime = Date()
	let iterationTime = stopTime.timeIntervalSince(startTime)
	let formatter = DateFormatter()
	formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSS"
	//print("Duration (milliseconds) \(iterationTime * 1000)")
	print("MetalGPU|1 threads|\(numIterations) iterations|\(ds.numElements) elements|\(ds.numClusters) clusters|\(ds.numDimensions) dimensions|\(formatter.string(from: startTime))|\(formatter.string(from: stopTime))|\(iterationTime * 1000) (milliseconds)")

	// Verify that the data is the same as the basic output
	/*
	var outputPtr = distanceBuffer.contents()

	var metalDistance: Int64
	var basicDistance: Int64
	for cIdx in 0..<ds.numClusters {
		for eIdx in 0..<ds.numElements {
			metalDistance = outputPtr.load(as: Int64.self)
			outputPtr = outputPtr + MemoryLayout<Int64>.stride
			basicDistance = distanceBetween(&ds, cIdx, eIdx)
			if metalDistance != basicDistance {
				print("Basic: \(basicDistance) Metal: \(metalDistance)")
			}
		}
	}
	*/

}