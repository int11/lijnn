INC=-I/usr/include/python3.10 -I/home/injea/lijnn/.venv/lib/python3.10/site-packages/numpy/core/include 
all: ctypes_test.so cuda_test.so extension_test.so numpy_extension.so

ctypes_test.so: ctypes_test.cpp
	g++ -g -shared ctypes_test.cpp -o ctypes_test.so

cuda_test.so: cuda_test.cu
	nvcc -Xcompiler -fPIC -shared -g cuda_test.cu -o cuda_test.so

extension_test.so: extension_test.cpp
	g++ -g -shared extension_test.cpp -o extension_test.so $(INC)

numpy_extension.so: numpy_extension.cpp
	g++ -fPIC -g -shared numpy_extension.cpp -o numpy_extension.so $(INC)

