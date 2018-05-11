CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
NVCC := nvcc
LINK := g++ -fPIC

INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += --std=c++11
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += --std=c++11

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcudart
OBJS = Intersect.cu.o test.cpp.o
TARGET = Test
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES: .cpp .cu .o

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(LINKLINE)
