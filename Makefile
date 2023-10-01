CC = g++ 
SRCS = ./src/*.cpp
INC = ./src/
OPTS = -std=c++17 -Wall -Werror -lpthread -O2

EXEC = bin/kmeans

all: 
	clean cleancuda compile cudaseq

compile:
	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

compile2:
	$(CC) $(SRCS) -std=c++17 -Wall -Werror -lpthread -O -g3 -I$(INC) -o $(EXEC) 
clean:
	rm -f $(EXEC)

cleancuda:
	rm -f ./cuda/kmeans
	rm -f ./cuda/*.o

cudaseq:
	nvcc -o ./cuda/kmeans.o -c ./cuda/kmeans.cpp
	nvcc -o ./cuda/kmeans_kernel.o -c ./cuda/kmeans_kernel.cu
	nvcc -o ./cuda/kmeans ./cuda/kmeans.o ./cuda/kmeans_kernel.o

debug: clean compile2

sequential: clean compile