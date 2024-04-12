
#include "sigmoid.h"

Sigmoid::Sigmoid():Node()
{
    this->op_name = "Sigmoid";
}

Sigmoid::~Sigmoid()
{
}


vim_int32 Sigmoid::forward()
{
    *output->get_matrix() = 1.0 / (1.0 + (- inputs[0]->get_matrix()->array()).exp()) ;
    output->shape = inputs[0]->shape;
    return 0;
}



vim_int32 Sigmoid::backward(vim_uint64 k)
{    
    MatrixRm m_input_back_tmp = (output->get_matrix()->array()) * (1 - (output->get_matrix()->array())) * (output->get_back_matrix()->array())  ;
    inputs[0]->set_back_data(this, &m_input_back_tmp);
    return 0;
}


