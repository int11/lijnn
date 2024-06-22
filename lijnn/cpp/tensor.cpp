#include <list>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <string>
#include <cuda_fp16.h>

using namespace std;
enum class DataType {
    int,
    __half,
    float,
    double
};

template <typename T>
class tensor_base {
public:
    tensor_base(int size = 0) : size_(size),  data(new T[capacity]) {}

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
            case DataType::int:
                values = new tensor_base<int>();
                break;
            case DataType::__half:
                values = new tensor_base<__half>();
                break;
            case DataType::float:
                values = new tensor_base<float>();
                break;
            case DataType::double:
                values = new tensor_base<double>();
                break;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor& t) {
        os << "[ ";
        for (int value : t.values) {
            os << value << " ";
        }
        os << "]";
        return os;
    }
private:
    DataType data_type;
};



int main() {
    tensor t(5, 'int');
    cout << t << endl;
    return 0;
}