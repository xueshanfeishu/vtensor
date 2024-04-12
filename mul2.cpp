
#include "mul2.h"

Mul2::Mul2():Node()
{
    this->op_name = "Mul2";
}

Mul2::~Mul2()
{
}

vim_int32 Mul2::forward()
{
    *output->get_matrix() = inputs[0]->get_matrix()->array() * inputs[1]->get_matrix()->array();
    output->shape = inputs[0]->shape;
    return 0;
}


vim_int32 Mul2::backward(vim_uint64 k)
{   
    MatrixRm m_tmp = output->get_back_matrix()->array() * inputs[1]->get_matrix()->array();
    inputs[0]->set_back_data(this, &m_tmp);
    m_tmp = output->get_back_matrix()->array() * inputs[0]->get_matrix()->array();
    inputs[1]->set_back_data(this, &m_tmp);
    return 0;
}


