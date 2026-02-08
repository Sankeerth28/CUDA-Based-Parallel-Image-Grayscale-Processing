NVCC=nvcc
CFLAGS=`pkg-config --cflags opencv4`
LFLAGS=`pkg-config --libs opencv4`

all:
	$(NVCC) src/main.cu -o image_filter $(CFLAGS) $(LFLAGS)

clean:
	rm -f image_filter
