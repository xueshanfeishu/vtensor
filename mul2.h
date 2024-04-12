
#ifndef __MUL2_H__
#define __MUL2_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"


class Mul2: public Node{
    public:
        Mul2();
        ~Mul2();
        
        vim_int32 forward();                // forward
        vim_int32 backward(vim_uint64 k);
    protected:
        
    private:
        
};


#endif
