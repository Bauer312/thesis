#!/usr/bin/env bash

# Example Usage
# sudo ./runIteration.sh m1mini.dat > m1mini-executiontime.dat

exe=./gaussian

if [ $(uname -s) = 'Linux' ]; then
  if [ $(uname -m) = 'x86_64' ]; then
    gpu=./blurCuda
	pwr=../run_turbostat.sh
  fi
else
  gpu=./blurMetal
  pwr=../run_power.sh
fi

touch $1

for stdDev in 0.25 0.5 0.85; do
  for imgSize in 5000 7500 10000; do
    $pwr rundata "$gpu $stdDev $imgSize"
    cat rundata.dat >> $1
    sleep 1
    for numThreads in 1 2 4 6 8 10; do
      $pwr rundata "$exe $stdDev $imgSize $numThreads"
      cat rundata.dat >> $1
      sleep 1
    done
  done
done
