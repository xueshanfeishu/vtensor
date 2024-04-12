
#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"


class Transpose: public Node{
    public:
        Transpose();
        Transpose(vector<vim_uint32>perm_);
        ~Transpose();
        
        vector<VTensor * > build(VTensor* input, vector<vim_uint32>perm);
        vim_int32 forward();                // forward
        vim_int32 backward(vim_uint64 k);
    protected:
    private:
        vim_int32 cal_back_perm();
        
        vector<vim_uint32>perm;
        vector<vim_uint32>back_perm;
        vim_int32 transpose(VTensor *tensor_in, MatrixRm *m_out, vector<vim_uint32>perm_);

};


#endif
