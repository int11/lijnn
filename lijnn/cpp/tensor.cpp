#include <list>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <string>
#include <cuda_fp16.h>

using namespace std;

enum class DataType {
    Int,
    Half,
    Float,
    Double
};

template <typename T>
class tensor_base {
public:
    tensor_base(int size = 0) : size_(size),  data(new T[size]) {}

    ~tensor_base() {
        delete[] data;
    }

    T& operator[](int index) {
        return data[index];
    }

    int size() const {
        return size_;
    }

private:
    int size_;
    T* data;
};

class tensor{
public:
    tensor(int size, DataType data_type_) : data_type(data_type_) {
        switch (data_type) {
            case DataType::Int:
                tensor_base_ptr = new tensor_base<int>(size);
                break;
            case DataType::Half:
                tensor_base_ptr = new tensor_base<__half>(size);
                break;
            case DataType::Float:
                tensor_base_ptr = new tensor_base<float>(size);
                break;
            case DataType::Double:
                tensor_base_ptr = new tensor_base<double>(size);
                break;
        }
    }

    ~tensor() {
        switch (data_type) {
            case DataType::Int:
                delete static_cast<tensor_base<int>*>(tensor_base_ptr);
                break;
            case DataType::Half:
                delete static_cast<tensor_base<__half>*>(tensor_base_ptr);
                break;
            case DataType::Float:
                delete static_cast<tensor_base<float>*>(tensor_base_ptr);
                break;
            case DataType::Double:
                delete static_cast<tensor_base<double>*>(tensor_base_ptr);
                break;
        }
    }
private:
    DataType data_type;
    void* tensor_base_ptr;
};



int main() {
    tensor t(5, DataType::Int);
    return 0;
}