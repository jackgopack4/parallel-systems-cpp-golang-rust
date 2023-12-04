CC = mpicxx
SRCS = ./src/*.cpp
INC = ./src/
OPTS = -std=c++17 -Wall -Werror -Wextra -O3 -lmpi


EXEC = bin/nbody

all: clean compile

compile:
	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

compile2:
	$(CC) $(SRCS) -std=c++17 -Wall -Werror -O -g3 -lmpi -I$(INC) -o $(EXEC) 
clean:
	rm -f $(EXEC)

debug: clean compile2