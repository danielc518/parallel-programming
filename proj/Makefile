BIN = bin/

all: $(BIN)
	nvcc parallel.cu -o $(BIN)parallel_cuda.o -std=c++11
	g++ serial.cpp -o $(BIN)serial.o -std=c++11
	g++ parallel.cpp -o $(BIN)parallel.o -std=c++11 -fopenmp
	javac Serial.java -d $(BIN)
	javac Parallel.java -d $(BIN)

clean:
	rm -rf $(BIN) 

$(BIN):
	mkdir -p $(BIN)