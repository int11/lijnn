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
class data_ptr {
public:
    data_ptr() : size_(0), capacity(1), data_(new T[capacity]) {}

    ~data_ptr() {
        delete[] data_;
    }

    void push_back(const T& value) {
        if (size_ == capacity) {
            capacity *= 2;
            T* newData = new T[capacity];
            for (int i = 0; i < size_; ++i) {
                newData[i] = data_[i];
            }
            delete[] data_;
            data_ = newData;
        }
        data_[size_++] = value;
    }

    T& operator[](int index) {
        return data_[index];
    }

    int size() const {
        return size_;
    }

private:
    int size_;
    int capacity;
    T* data_;
};

class tensor{
public:
    tensor(std::initializer_list<int> list, const DataType data_type_) : data_type(data_type_) {
        

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
    tensor t({1, 2, 3});
    cout << t << endl;
    return 0;
}