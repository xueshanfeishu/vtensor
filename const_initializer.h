#ifndef __CONST_INITIALIZER_H__
#define __CONST_INITIALIZER_H__

#include "base_initializer.h"
#include "ndarray.h"


using namespace std;

class Constant_initializer:public Base_initializer{
    public:
        Constant_initializer(Real const_value_)
        {
            method = "constant";
            const_value  = const_value_;
        };
        ~Constant_initializer(){};
        vim_int32 initialize(Ndarray * var);
        inline Real get_const(){return const_value;};
        inline void set_const(Real val_){ const_value = val_;};
    private:
        Real const_value;
};

#endif
