SERIAL_TARGET=serial
PARALLEL_TARGET=parallel
OPTIMIZED_TARGET=optimized

SERIAL_SOURCE=serial.c
PARALLEL_SOURCE=parallel.c
OPTIMIZED_SOURCE=optimized.c

CFLAGS += -std=c99 -fopenmp 
LFLAGS += -lgomp

CFLAGS_DEBUG = -O0 -g3

CFLAGS_RELEASE = -O3

SRC = ./
BIN = bin/
BIN_O = ./
INCLUDE = /usr/include/ -I. -IInclude -ISource

CC=gcc
CXX=g++
MD=mkdir

SERIAL_OBJECTS=$(addprefix $(BIN_O), $(addsuffix .o, $(basename $(SERIAL_SOURCE))))
PARALLEL_OBJECTS=$(addprefix $(BIN_O), $(addsuffix .o, $(basename $(PARALLEL_SOURCE))))
OPTIMIZED_OBJECTS=$(addprefix $(BIN_O), $(addsuffix .o, $(basename $(OPTIMIZED_SOURCE))))


all: CFLAGS += $(CFLAGS_DEBUG)
all: $(BIN)$(SERIAL_TARGET)
all: $(BIN)$(PARALLEL_TARGET)
all: $(BIN)$(OPTIMIZED_TARGET)

release: CFLAGS += $(CFLAGS_RELEASE)
release: $(BIN)$(ALIGN_S2_TARGET)
release: $(BIN)$(SYMMETRY_S2_TARGET)
release: $(BIN)$(SERIAL_TARGET)
release: $(BIN)$(PARALLEL_TARGET)
release: $(BIN)$(OPTIMIZED_TARGET)

clean:
	rm -f $(BIN)$(SERIAL_TARGET)
	rm -f $(BIN)$(PARALLEL_TARGET)
	rm -f $(BIN)$(OPTIMIZED_TARGET)
	rm -f $(SERIAL_OBJECTS)
	rm -f $(PARALLEL_OBJECTS)
	rm -f $(OPTIMIZED_OBJECTS)
	rm -rf $(BIN) 
	
$(BIN):
	mkdir -p $(BIN)

$(BIN)$(SERIAL_TARGET): $(SERIAL_OBJECTS)
	mkdir -p $(BIN)
	$(CC) -o $@ $(SERIAL_OBJECTS) $(LFLAGS) 

$(BIN)$(PARALLEL_TARGET): $(PARALLEL_OBJECTS)
	mkdir -p $(BIN)
	$(CC) -o $@ $(PARALLEL_OBJECTS) $(LFLAGS)

$(BIN)$(OPTIMIZED_TARGET): $(OPTIMIZED_OBJECTS)
	mkdir -p $(BIN)
	$(CC) -o $@ $(OPTIMIZED_OBJECTS) $(LFLAGS) 

$(BIN_O)%.o: $(SRC)%.c
	$(CC) -c -o $@ $(CFLAGS) $<

$(BIN_O)%.o: $(SRC)%.cpp
	$(CXX) -c -o $@ $(CFLAGS) $<

