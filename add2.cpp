
#include "add2.h"

Add2::Add2():Node()
{
    this->op_name = "Add2";
}

Add2::~Add2()
{
}

vim_int32 Add2::forward()
{
    *output->get_matrix() = *(inputs[0]->get_matrix()) + *(inputs[1]->get_matrix());
    output->shape = inputs[0]->shape;
    // if(name=="add3"){
    //     cout<<name<<"input0:"<<(inputs[0]->get_matrix())->row(0)<<endl;
    //     cout<<name<<"input1:"<<(inputs[1]->get_matrix())->row(0)<<endl;
    //     cout<<name<<"output:"<<output->get_matrix()->row(0)<<endl;
    // }
    return 0;
}

vim_int32 Add2::backward(vim_uint64 k)
{    
    inputs[0]->set_back_data(this, output->get_back_matrix());
    inputs[1]->set_back_data(this, output->get_back_matrix());
    return 0;
}

