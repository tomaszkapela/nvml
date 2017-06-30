CC=gcc
OUT=perflush

CFLAGS=-O3 -Wall -mavx
LDFLAGS=

SRC=main.c

all:
	$(CC) $(SRC) $(CFLAGS) $(LDFLAGS) -o $(OUT)

clean:
	rm -rf $(OUT)
