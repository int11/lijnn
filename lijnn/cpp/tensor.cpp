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



// 공통 인터페이스
class ITensor {
public:
    virtual ~ITensor() = default;
    virtual void printType() const = 0;
};

// 구체적 타입 구현
template<typename T>
class TensorImpl : public ITensor {
public:
    void printType() const override {
        std::cout << typeid(T).name() << std::endl;
    }
};

// 타입 소거를 사용하는 Tensor 클래스
class Tensor {
private:
    std::shared_ptr<ITensor> tensor_ptr;

public:
    template<typename T>
    Tensor(T) : tensor_ptr(std::make_shared<TensorImpl<T>>()) {}

    void printType() const {
        tensor_ptr->printType();
    }
};

// 사용 예
int main() {
    Tensor intTensor(int{});
    Tensor floatTensor(float{});

    intTensor.printType(); // 출력: i
    floatTensor.printType(); // 출력: f

    return 0;
}


int main() {
    tensor t(5, DataType::Int);
    return 0;
}