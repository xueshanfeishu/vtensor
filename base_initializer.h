#ifndef __BASE_INITIALIZER_H__
#define __BASE_INITIALIZER_H__

#include <string>
#include "ndarray.h"

using namespace std;

class Ndarray;

class Base_initializer{
    public:
        Base_initializer(){};
        ~Base_initializer(){};
        string method;
        virtual vim_int32 initialize(Ndarray * var) {return 0;};
};

#endif
