PLATFORM = mac  # mac

LDFLAGS = -shared -fPIC
ifeq ($(strip $(PLATFORM)), linux)
    CXX = g++
    CXXFLAGS = -O3 -Wall -std=c++17 -fPIC -fopenmp
    LDFLAGS += -fopenmp
else ifeq ($(strip $(PLATFORM)), mac)
    CXX = clang++
    OMP_PATH = $(shell brew --prefix libomp)
    CXXFLAGS = -O3 -Wall -std=c++17 -fPIC -Xpreprocessor -fopenmp -I$(OMP_PATH)/include
    LDFLAGS += -L$(OMP_PATH)/lib -lomp -undefined dynamic_lookup
endif

SRC = $(wildcard *.cpp)
OBJ = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRC))
TARGET = libs$(shell python3-config --extension-suffix)

# Paths to Pybind11 and Python include directories
PYBIND11_INCLUDE = $(shell python -m pybind11 --includes)

BUILD_DIR = ./build

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDE) -c $< -o $@

# Link object files into a shared library
$(BUILD_DIR)/$(TARGET): $(OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(BUILD_DIR)/$(TARGET) $(LDFLAGS)
	cp $(BUILD_DIR)/$(TARGET) ../$(TARGET)

# Clean up build files
clean:
	rm -rf $(BUILD_DIR)
