all: libkalman.so

matrix.o: src/matrix.c src/matrix.h
	gcc -c -fPIC src/matrix.c -o matrix.o

kalman.o: src/kalman.c src/kalman.h src/matrix.h
	gcc -c -fPIC src/kalman.c -o kalman.o

libkalman.so: matrix.o kalman.o
	gcc -shared -o libkalman.so matrix.o kalman.o

clean:
	rm -f *.o *.so