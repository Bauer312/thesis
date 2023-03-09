UNAME_S := $(shell uname -s)
$(info Operating System: $(UNAME_S))
UNAME_M := $(shell uname -m)
$(info Architecture: $(UNAME_M))

ifeq ($(UNAME_S),Linux)
	CC = gcc
ifeq ($(UNAME_M),x86_64)
	CCFLAGS = -std=c11 -O2 -mavx2 -Wall
endif
ifeq ($(UNAME_M),aarch64)
	CCFLAGS = -std=c11 -O2 -Wall
endif
	LDFLAGS = -lm -lpthread
endif
ifeq ($(UNAME_S),Darwin)
	CC = clang
	SCC = swiftc
	CCFLAGS = -std=c11 -O2 -Wall
	LDFLAGS = -lm
endif

distance: main.o distance.o
	$(CC) $(CCFLAGS) main.o distance.o -o distance $(LDFLAGS)

testdistance: testdistance.o distance.o
	$(CC) $(CCFLAGS) testdistance.o distance.o -o testdistance $(LDFLAGS)

distance.o: distance.c distance.h
	$(CC) $(CCFLAGS) distance.c -c

main.o: main.c
	$(CC) $(CCFLAGS) main.c -c

distanceCuda: distance.cu distance.c distance.h
	nvcc -o distanceCuda distance.cu -Xcompiler -mavx2 -Xcompiler -Wall -Xcompiler -O2 -Xcompiler -DENABLE_CUDA

distanceMetal.o: distanceMetal.swift distance.c distance.h
	$(SCC) distanceMetal.swift -c -import-objc-header distance.h

distanceMetal: distanceMetal.o distance.o
	$(SCC) distanceMetal.o distance.o -o distanceMetal -framework Metal -framework CoreGraphics

testdistance.o: testdistance.c
	$(CC) $(CCFLAGS) testdistance.c -c

test: testdistance
	./testdistance

cuda: distanceCuda

metal: distanceMetal

clean:
	rm -f *.o *~ distance testdistance distanceCuda distanceMetal
