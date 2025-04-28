CC = gcc
CFLAGS = -fopenmp -Wall -lm
LDFLAGS =

TARGET = modified_jacobi

SRCS = modified_jacobi.c

OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean