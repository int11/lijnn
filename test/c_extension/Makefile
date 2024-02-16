INC=-I/usr/include/python3.10 -I/home/injea/lijnn/.venv/lib/python3.10/site-packages/numpy/core/include 
all: ctypes.so cuda.so cpythonExtension.so numpyExtension.so

ctypes.so: ctypes.cpp
	g++ -g -shared ctypes.cpp -o ctypes.so

cuda.so: cuda.cu
	nvcc -Xcompiler -fPIC -shared -g cuda.cu -o cuda.so

cpythonExtension.so: cpythonExtension.cpp
	g++ -g -shared cpythonExtension.cpp -o cpythonExtension.so $(INC)

numpyExtension.so: numpyExtension.cpp
	g++ -fPIC -g -shared numpyExtension.cpp -o numpyExtension.so $(INC)

