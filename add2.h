
#ifndef __ADD2_H__
#define __ADD2_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"


class Add2: public Node{
    public:
        Add2();
        ~Add2();
        
        vim_int32 forward();                // forward
        vim_int32 backward(vim_uint64 k);
    protected:
    private:
        
};


#endif
