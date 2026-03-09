NVCC = nvcc
ARCH = -arch=sm_50
FLAGS = -rdc=true -pg
OUTPUT = neura.out

SOURCES = code_initial/main.cu code_initial/matrix.cu code_initial/mnist.cu code_initial/ann.cu
OBJECTS = $(SOURCES:.cu=.o)

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $^

clean:
	rm $(OUTPUT)

.PHONY: all clean
