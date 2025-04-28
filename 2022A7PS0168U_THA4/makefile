# Makefile for MPI-OpenMP Hybrid Program on macOS

# Compiler settings
CC = mpicc
CFLAGS = -Wall -Xpreprocessor -fopenmp -O2

# Get LLVM path from Homebrew (assuming it's installed via Homebrew)
LLVM_PATH = $(shell brew --prefix llvm 2>/dev/null || echo "/usr/local/opt/llvm")
LDFLAGS = -L$(LLVM_PATH)/lib -lomp

# Program settings
TARGET = log_analyzer
INPUT_FILE = BGL_2k.log
SOURCES = mpi.c

# Main target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
build:
	mpicc -Wall -Xpreprocessor -fopenmp -O2 -I/usr/local/opt/libomp/include -o log_analyzer mpi.c -L/usr/local/opt/libomp/lib -lomp
# Run the program
run:
	mpirun -np 4 ./$(TARGET) $(INPUT_FILE)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Show environment information (useful for debugging)
info:
	@echo "Compiler: $(CC)"
	@echo "LLVM path: $(LLVM_PATH)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "To run with a specific input file, use:"
	@echo "make run INPUT_FILE=your_log_file.log"

.PHONY: all run clean info