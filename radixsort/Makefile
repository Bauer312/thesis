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

radixsort: main.o radix.o
	$(CC) $(CCFLAGS) main.o radix.o -o radixsort $(LDFLAGS)

qs: qs.o radix.o
	$(CC) $(CCFLAGS) qs.o radix.o -o qs $(LDFLAGS)

testradix: testradix.o radix.o
	$(CC) $(CCFLAGS) testradix.o radix.o -o testradix $(LDFLAGS)

radix.o: radix.c radix.h
	$(CC) $(CCFLAGS) radix.c -c

main.o: main.c radix.h
	$(CC) $(CCFLAGS) main.c -c

qs.o: qs.c radix.h
	$(CC) $(CCFLAGS) qs.c -c

radixCuda: radix.cu radix.c radix.h
	nvcc -g -G -o radixCuda radix.cu -Xcompiler -mavx2 -Xcompiler -Wall -Xcompiler -O2 -Xcompiler -DENABLE_CUDA

radixMetal.o: radixMetal.swift radix.c radix.h
	$(SCC) radixMetal.swift -c -import-objc-header radix.h

radixMetal: radixMetal.o radix.o
	$(SCC) radixMetal.o radix.o -o radixMetal -framework Metal -framework CoreGraphics

testradix.o: testradix.c
	$(CC) $(CCFLAGS) testradix.c -c

test: testradix
	./testradix

cuda: radixCuda

metal: radixMetal

clean:
	rm -f *.o *~ radixsort testradix radixCuda radixMetal qs
