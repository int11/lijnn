#include <list>

class Variable {
public:
    Variable(const std::list<int>& values) : values_(values) {}

private:
    std::list<int> values_;
};
