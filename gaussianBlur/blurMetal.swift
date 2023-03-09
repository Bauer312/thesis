import Foundation
import Metal

if CommandLine.argc != 3 {
	print("Usage: <\(CommandLine.arguments[0])> [Std Deviation] [ImgSize]")
} else {
	let stdDev = Float64(CommandLine.arguments[1]) ?? 0.84089642
	let imgSize = Int32(CommandLine.arguments[2]) ?? 1024

	var bs: blurStruct = blurStruct()

	if initBlurStruct(&bs, stdDev, imgSize) != 0 {
		print("Error when initializing the blur struct")
	}

	//print("Populating data structure with random data");
	populateBlurStruct(&bs)

	//execute(&bs, 1, STYLE_BASIC);

	sleep(2)
	executeMetal(&bs)
	sleep(2)

	freeBlurStruct(&bs)

}

func executeMetal(_ bs: inout blurStruct) {
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

		kernel void blur(device const int32_t* shapeArr,
						device const int32_t *imageArr,
						device int32_t *outputArr,
						device const float *convArr,
						uint3 index [[thread_position_in_grid]]) {
			
			int x = index[0];
			int y = index[1];
			int imgSize = shapeArr[0];
			int convSize = shapeArr[1];
			
			if(x >= convSize && x < imgSize - convSize) {
				if(y >= convSize && y < imgSize - convSize) {
					// Only process data within the buffer
					int cRow = 0;
					int elementCount = 0;
					float accumulator = 0.0;
					for(int row = y - convSize; row <= y + convSize; row++) {
						int rowOffset = row * imgSize;
						int cRowOffset = cRow * (convSize * 2 + 1);
						int cCol = 0;
						for(int col = x - convSize; col <= x + convSize; col++) {
							accumulator += (float)imageArr[rowOffset + col] * convArr[cRowOffset + cCol];
							elementCount++;
							cCol++;
						}
						cRow++;
					}
					//outputArr[y * imgSize + x] = (int)accumulator;
					outputArr[y * imgSize + x] = ceil(accumulator / elementCount);
				}
			}
		}
	"""
	
	let srcLibrary: MTLLibrary
	let blurFunction: MTLFunction?
	let computePipeline: MTLComputePipelineState
	do {
		srcLibrary = try device.makeLibrary(source: kernelSrc, options: nil)
	} catch {
		fatalError( "Failed to create srcLibrary." )
	}

	blurFunction = srcLibrary.makeFunction(name: "blur")

	do {
		computePipeline = try device.makeComputePipelineState(function: blurFunction!)
	} catch {
		fatalError( "Failed to create computePipeline." )
	}

	// Start counting the time
	//let startTime = Date()

	var shapeData: [Int32] = []
	shapeData.reserveCapacity(2)
	shapeData.append(bs.imageSize)
	shapeData.append(bs.convolutionDistance)

	let shapeBufferSize = MemoryLayout<Int32>.stride * 2
	guard let shapeBuffer = device.makeBuffer(bytes: shapeData, length: shapeBufferSize, options: .storageModeShared) else {
	fatalError( "Failed to make shapeBuffer." )
	}

	let bufferSize: Int = MemoryLayout<Int32>.stride * Int(bs.imageSize) * Int(bs.imageSize)
	// Copy the elements
	guard let imageBuffer = device.makeBuffer(bytes: bs.image, length: bufferSize, options: .storageModeShared) else {
		fatalError( "Failed to make imageBuffer" )
	}
	// No need to copy elements
	guard let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
		fatalError( "Failed to make outputBuffer" )
	}

	let convolutionElementSide: Int = Int(bs.convolutionDistance) * 2 + 1;
	let convSize: Int = MemoryLayout<Float32>.stride * convolutionElementSide * convolutionElementSide
	// No need to copy elements
	guard let convBuffer = device.makeBuffer(bytes: bs.convolutionMatrix, length: convSize, options: .storageModeShared) else {
		fatalError( "Failed to make convBuffer" )
	}

	// Start counting the time
	let startTime = Date()

	// Start a compute pass
	guard let commandBuffer = commandQueue.makeCommandBuffer() else {
		fatalError( "Failed to make commandBuffer." )
	}
	guard let blurEncoder = commandBuffer.makeComputeCommandEncoder() else {
		fatalError( "Failed to make blurEncoder." )
	}

	blurEncoder.setComputePipelineState(computePipeline)
	blurEncoder.setBuffer(shapeBuffer, offset: 0, index: 0)
	blurEncoder.setBuffer(imageBuffer, offset: 0, index: 1)
	blurEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
	blurEncoder.setBuffer(convBuffer, offset: 0, index: 3)

	let msbGridSize = MTLSizeMake(Int(bs.imageSize), Int(bs.imageSize), 1)
	let msbThreadGroupSize = MTLSizeMake(16, 16, 1)

	blurEncoder.dispatchThreads(msbGridSize, threadsPerThreadgroup: msbThreadGroupSize)
	blurEncoder.endEncoding()

	commandBuffer.commit()
	commandBuffer.waitUntilCompleted()
	

	// Stop counting the time
	let stopTime = Date()
	let iterationTime = stopTime.timeIntervalSince(startTime)
	let formatter = DateFormatter()
	formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSS"
	//print("Duration (milliseconds) \(iterationTime * 1000)")
	print("MetalGPU|1 threads|\(bs.standardDeviation) stdDev|\(bs.imageSize) imageSize|\(formatter.string(from: startTime))|\(formatter.string(from: stopTime))|\(iterationTime * 1000) (milliseconds)")

	// Verify that the data is the same as the basic output
	/*
	var outputPtr = outputBuffer.contents()
	var elementValue: Int32
	var matching: Bool
	matching = true

	for row in 0..<Int(bs.imageSize) {
		let rowOffset: Int = row * Int(bs.imageSize)
		for col in 0..<Int(bs.imageSize) {
			elementValue = outputPtr.load(as: Int32.self)
			outputPtr = outputPtr + MemoryLayout<Int32>.stride
			if abs(elementValue - bs.newImage[rowOffset + col]) > 1 {
				matching = false
				print("[\(row),\(col)] GPU \(elementValue) Basic \(bs.newImage[rowOffset + col])")
			}
		}
	}
	if matching { print("Match!") } else { print("Not matching") }
	*/
}
