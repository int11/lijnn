all: ctypes_test.so cuda_test.so extension_test.so

ctypes_test.so: ctypes_test.cpp
	g++ -g -shared ctypes_test.cpp -o ctypes_test.so

cuda_test.so: cuda_test.cu
	nvcc -Xcompiler -fPIC -shared -g cuda_test.cu -o cuda_test.so

extension_test.so: extension_test.cpp
	g++ -g -shared extension_test.cpp -o extension_test.so
