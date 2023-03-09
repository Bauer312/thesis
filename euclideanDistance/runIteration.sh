#!/usr/bin/env bash

# Example Usage
# sudo ./runIteration.sh m1mini.dat >> m1mini-executiontime.dat

exe=./distance

if [ $(uname -s) = 'Linux' ]; then
	if [ $(uname -m) = 'x86_64' ]; then
		gpu=./distanceCuda
		pwr=../run_turbostat.sh
	fi
else
	gpu=./distanceMetal
	pwr=../run_power.sh
fi

touch $1

for numPoints in 1000 10000; do
	for numClusters in 16 32 64; do
		for numDimensions in 16 32 64; do
			$pwr rundata "$gpu $numPoints $numClusters $numDimensions 1 40"
			cat rundata.dat >> $1
			sleep 1

			for numThreads in 1 4 -1; do
				$pwr rundata "$exe $numPoints $numClusters $numDimensions $numThreads 40"
				cat rundata.dat >> $1
				sleep 1
			done
		done
	done
done
