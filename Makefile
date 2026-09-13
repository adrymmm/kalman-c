CFLAGS = -Wall -Wextra -fPIC -Isrc

all: libkalman.so

matrix.o: src/matrix.c src/matrix.h
	gcc -c $(CFLAGS) src/matrix.c -o matrix.o

kalman.o: src/kalman.c src/kalman.h src/matrix.h
	gcc -c $(CFLAGS) src/kalman.c -o kalman.o

libkalman.so: matrix.o kalman.o
	gcc -shared -o libkalman.so matrix.o kalman.o

test_llt: tests/kalman_test.c matrix.o kalman.o
	gcc $(CFLAGS) tests/kalman_test.c matrix.o kalman.o -o test_llt -lm

clean:
	rm -f *.o *.so test_llt