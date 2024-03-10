#include <half.hpp>
#include <iostream>
using namespace std;
int main() {
    cout << fixed;
    cout.precision(16);
    const double e = 2.7182818284590452353602874713527;
    float f = pow(e, 2);
    cout << f << " " << sizeof(f) << endl;
    return 0; 
}
