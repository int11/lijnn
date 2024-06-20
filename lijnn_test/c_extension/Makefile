INC=-I/usr/include/python3.10 -I ./.venv/lib/python3.10/site-packages/numpy/core/include/ -I/usr/local/cuda/include/
all: ctypes.so cuda.so cpythonExtension.so numpyExtension.so

%.so: %.cpp
	g++ -fPIC -g -shared $< -o $@ $(INC)

cuda.so: cuda.cu
	nvcc -Xcompiler -fPIC -shared -g cuda.cu -o cuda.so