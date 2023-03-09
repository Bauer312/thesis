#!/usr/bin/env bash

# Example Usage
# sudo ./runIteration.sh m1mini.dat > m1mini-executiontime.dat

exe=./radixsort

if [ $(uname -s) = 'Linux' ]; then
  if [ $(uname -m) = 'x86_64' ]; then
    gpu=./radixCuda
	pwr=../run_turbostat.sh
  fi
else
  gpu=./radixMetal
  pwr=../run_power.sh
fi

touch $1

for numIter in 1000000 5000000 10000000; do
  $pwr rundata "$gpu $numIter"
  cat rundata.dat >> $1
  sleep 1
  for numThreads in 1 2 4 6 8 10; do
    $pwr rundata "$exe $numIter $numThreads"
    cat rundata.dat >> $1
    sleep 1
  done
done
