
#include "tanh.h"

Tanh::Tanh():Node()
{
    this->op_name = "Tanh";
}

Tanh::~Tanh()
{
}


vim_int32 Tanh::forward()
{
    *output->get_matrix() = inputs[0]->get_matrix()->array().tanh();
    output->shape = inputs[0]->shape;
    return 0;
}

vim_int32 Tanh::backward(vim_uint64 k)
{    
    MatrixRm m_input_back_tmp = (1 - (output->get_matrix()->array()) * (output->get_matrix()->array())) * (output->get_back_matrix()->array())  ;
    inputs[0]->set_back_data(this, &m_input_back_tmp);
    return 0;
}
