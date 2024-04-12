
#ifndef __CTC_LOSS_H__
#define __CTC_LOSS_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"

class Ctc_loss: public Node{
    public:
        Ctc_loss(vim_uint32 logits_time_major_ = 1, vim_uint32 blank_index_ = 0);
        ~Ctc_loss();
        
        vector<VTensor * > build(vector<vector <vim_uint32>> *labels_, 
                                  VTensor * logits_ ,
                                  vector <vim_uint32> *label_length_, 
                                  vector <vim_uint32> *logit_legth);
        vim_int32 forward();                // forward
        vim_int32 backward(vim_uint64 k);
    protected:
        
    private:
        vim_int32 backword_ctc();
        inline Real log_sum_exp2(Real x1, Real x2);
        inline MatrixRm log_sum_exp2m(MatrixRm m1, MatrixRm m2);
        inline Real log_sum_exp1v(MatrixRm m);
        inline vim_uint32 tn2row(vim_uint32 t, vim_uint32 n)
        {
            if(logits_time_major)
                return t*N+n;
            else
                return n*T+t;            
        };

        vector<vector <vim_uint32>> *labels;
        vector <vim_uint32> *label_length; 
        vector <vim_uint32> *logit_length;
        
        vim_uint32 blank_index;
        vim_uint32 logits_time_major;

        MatrixRm log_softmax_logits;
        vector<MatrixRm> for_blanks;
        vector<MatrixRm> back_blanks;
        vector<MatrixRm> alphas;
        vector<MatrixRm> betas;

        vim_uint32 T;
        vim_uint32 N;
        vim_uint32 C;
};


#endif
