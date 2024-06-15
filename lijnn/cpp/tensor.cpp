#include <list>
#include <cuda.h>
#include <vector>
#include <iostream>
using namespace std;

class tensor{
public:
    tensor(const std::vector<int>& values_) : values(values_) {}

    friend std::ostream& operator<<(std::ostream& os, const tensor& t) {
        os << "[ ";
        for (int value : t.values) {
            os << value << " ";
        }
        os << "]";
        return os;
    }

private:
    std::vector<int> values;
};

int main() {
    tensor t({1, 2, 3});
    cout << t << endl;
    return 0;
}