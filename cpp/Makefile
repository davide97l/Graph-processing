all: clean program1 program2

program1: page_rank.cpp
	g++ -std=c++11 -fopenmp -g -O3 -o pr page_rank.cpp

program2: trust_rank.cpp
	g++ -std=c++11 -fopenmp -g -O3 -o tr trust_rank.cpp
clean:
	rm -rf pr *~ *.*~
	rm -rf tr *~ *.*~
