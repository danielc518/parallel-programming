After running make, move to the 'bin' directory to run the programs.

Except for the CUDA program, the input dimension for parallel programs MUST be a multiple of 5.

Sample Java serial version execution (10 = # of dimensions):
	java Serial 10

Sample Java parallel version execution (5 = # of dimensions, 2 = # of threads to use):
	java Parallel 5 2

Sample C++ serial version execution (10 = # of dimensions):
	./serial.o 10

Sample C++ parallel version (OpenMP) execution (5 = # of dimensions, 2 = # of threads to use):
	./parallel.o 5 2

Sample CUDA version execution (32 = # of blocks to use):
	sudo ./parallel_cuda.o 32
For CUDA version, the input vector dimension should be modified in the source file.