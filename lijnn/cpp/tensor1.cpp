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