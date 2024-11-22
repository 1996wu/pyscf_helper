CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 -fPIC -fopenmp

SRC = $(wildcard *.cpp)
OBJ = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRC))
TARGET = libs$(shell python3-config --extension-suffix)

# Paths to Pybind11 and Python include directories
PYBIND11_INCLUDE = $(shell python3 -m pybind11 --includes)

# Output directory for build
BUILD_DIR = ./build

# Compiler and linker flags
LDFLAGS = -shared -fPIC

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDE) -c $< -o $@

# Link object files into a shared library
$(BUILD_DIR)/$(TARGET): $(OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(BUILD_DIR)/$(TARGET) $(LDFLAGS)
	cp $(BUILD_DIR)/$(TARGET) ../$(TARGET)

# Clean up build files
clean:
	rm -rf $(BUILD_DIR)
