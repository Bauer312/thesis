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

gaussian: main.o blur.o
	$(CC) $(CCFLAGS) main.o blur.o -o gaussian $(LDFLAGS)

main.o: main.c
	$(CC) $(CCFLAGS) main.c -c

blur.o: blur.c blur.h
	$(CC) $(CCFLAGS) blur.c -c

blurCuda: blur.cu blur.c blur.h
	nvcc -g -G -o blurCuda blur.cu -Xcompiler -mavx2 -Xcompiler -Wall -Xcompiler -O2 -Xcompiler -DENABLE_CUDA

blurMetal.o: blurMetal.swift blur.c blur.h
	$(SCC) blurMetal.swift -c -import-objc-header blur.h

blurMetal: blurMetal.o blur.o
	$(SCC) blurMetal.o blur.o -o blurMetal -framework Metal -framework CoreGraphics

testblur: testblur.o blur.o
	$(CC) $(CCFLAGS) testblur.o blur.o -o testblur $(LDFLAGS)

testblur.o: testblur.c
	$(CC) $(CCFLAGS) testblur.c -c

test: testblur
	./testblur

cuda: blurCuda

metal: blurMetal

clean:
	rm -f *.o *~ testblur gaussian blurMetal blurCuda
