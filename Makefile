CC = g++ 
SRCS = ./src/*.cpp
INC = ./src/
OPTS = -std=c++17 -Wall -Werror -lpthread -O2

EXEC = bin/kmeans

all: clean compile

compile:
	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

compile2:
	$(CC) $(SRCS) -std=c++17 -Wall -Werror -lpthread -O -g3 -I$(INC) -o $(EXEC) 
clean:
	rm -f $(EXEC)

debug: clean compile2