
#include "transpose.h"

Transpose::Transpose():Node()
{
    this->op_name = "Transpose";
}

Transpose::Transpose(vector<vim_uint32>perm_):Node()
{
    this->op_name = "Transpose";
    for(vim_int32 i=perm.size()-1; i>=0; i--)
        assert( perm_[i] < perm.size());
    perm = perm_;
    cal_back_perm();
}

Transpose::~Transpose()
{
}

vim_int32 Transpose::forward()
{
    transpose(inputs[0], output->get_matrix(), perm);
    return 0;
}

vim_int32 Transpose::backward(vim_uint64 k)
{   
    MatrixRm m_tmp(output->get_matrix()->rows(),output->get_matrix()->cols());
    transpose(output, &m_tmp, back_perm);
    inputs[1]->set_back_data(this, &m_tmp);
    return 0;
}


vim_int32 Transpose::cal_back_perm()
{
    back_perm.resize(perm.size());
    for(vim_int32 i=0;i<perm.size();i++)
        back_perm[perm[i]] = i;

    return 0;
}

vim_int32 Transpose::transpose(VTensor *tensor_in, MatrixRm *m_out, vector<vim_uint32>perm_)
{
    vim_uint32 cols = 1, rows = 1;
    vim_int32 idx;

    for(idx=perm_.size()-1; idx>=0; idx--)
        if(perm_[idx] == idx)
            cols *= perm_[idx];
        else
            break;

    if(idx<0)
    {//no need to transpose
        *m_out = *tensor_in->get_matrix();
    }
    else
    {
        rows = inputs[0]->get_matrix()->size()/cols;
        Map <MatrixRm> mm_in(tensor_in->get_matrix()->data(), rows, cols);
        Map <MatrixRm> mm_out(m_out->data(), rows, cols);

#define MAX_SHAPE_SIZE (8)
        vim_uint32 in_shape_prod[MAX_SHAPE_SIZE];
        in_shape_prod[idx] = tensor_in->shape[idx];
        for(vim_int32 i=idx-1; i>0; i--)
            in_shape_prod[i] = tensor_in->shape[i]*in_shape_prod[i+1];

        vim_uint32 out_shape_prod[MAX_SHAPE_SIZE];
        for(vim_int32 i=idx; i>0; i--)
            out_shape_prod[i] = tensor_in->shape[perm_[idx]];

        for(vim_uint32 i=0;i<rows;i++)
        {
            vim_uint32 in_idx[MAX_SHAPE_SIZE];
            vim_uint32 out_idx[MAX_SHAPE_SIZE];
            vim_uint32 k=i;
            for(vim_uint32 j=0;j<idx;j++)
            {
                in_idx[j] = k/in_shape_prod[j+1];
                k = k%in_shape_prod[j+1];
            }
            in_idx[idx] = k;

            for(vim_uint32 j=0;j<=idx;j++)
                out_idx[i] = in_idx[perm_[i]];

            k = out_idx[0];
            for(vim_uint32 j=1;j<=idx;j++)
                k = k*out_shape_prod[j] + out_idx[j];

            mm_out.row(k) = mm_in.row(i);
        }
    }
    return 0;
}


vector<VTensor * > Transpose::build(VTensor* input, vector<vim_uint32>perm_)
{
    perm = perm_;
    cal_back_perm();
    register_input_tensor(input);
    output->name = name + ":0";
    output->owner = this;
    return {output};
}

