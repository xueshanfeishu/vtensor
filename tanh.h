
#ifndef __TANH_H__
#define __TANH_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"


class Tanh: public Node{
    public:
        Tanh();
        ~Tanh();
        
        vim_int32 forward();                // forward
        vim_int32 backward(vim_uint64 k);
    protected:
    private:
        
};


#endif
