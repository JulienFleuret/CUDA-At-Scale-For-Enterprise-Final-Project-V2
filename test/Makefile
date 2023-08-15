CXX = g++

# Specify the flags for the compiler (-I includes the header files, -L directory of the library, -l links the library)
OCV = $(echo pkg-config --list-all | grep -i opencv | cut -d' ' -f1)

CXXFLAGS := $(shell pkg-config --cflags --libs `pkg-config --list-all | grep -i opencv | cut -d' ' -f1`)

all: test
	
test: 
	$(CXX) -o test test.cpp $(CXXFLAGS)

# Rule for cleaning
.PHONY: clean
clean:
	rm -f test
	
.PHONY: test
	chmod +x test
	./test
