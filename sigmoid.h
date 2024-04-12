
#ifndef __SIGMOID_H__
#define __SIGMOID_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"


class Sigmoid: public Node{
    public:
        Sigmoid();
        ~Sigmoid();
        
        vim_int32 forward();                // forward
        vim_int32 backward(vim_uint64 k);
    protected:
    private:
        
};


#endif
