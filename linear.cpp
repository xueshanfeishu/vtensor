#include "train_configuration.h"
#include "linear.h"

Linear::Linear(vim_uint32 in_elem_num, vim_uint32 out_elem_num):Node()
{
    weight.preset_shape({in_elem_num, out_elem_num});
    bias.preset_shape({out_elem_num});
    this->op_name = "Linear";

    weight_updater = NULL;
    bias_updater = NULL;
}

Linear::~Linear()
{
    if(weight_updater)
        callback_trainable_variable_updater(weight_updater);
    if(bias_updater)
        callback_trainable_variable_updater(bias_updater);
}


vim_int32 Linear::forward()
{
    //  根据input_tensor和weight bias来计算output_tensor
    assert (inputs[0]->shape[1] == weight.shape[0]);
    (*output->get_matrix()) = (* (inputs[0]->get_matrix()) ) * (*weight.get_matrix()) + (*bias.get_matrix());
    return 0;
}

vim_int32 Linear::backward(vim_uint64 k)
{
    //
    MatrixRm bias_nabla = output->get_back_matrix()->colwise().sum();
    bias_updater->update( &bias_nabla, k, output->shape[0] );

    MatrixRm weight_nabla = MatrixRm::Zero(weight.shape[0],weight.shape[1]);

    for (vim_int32 i=0; i<output->shape[0]; i++)
        weight_nabla += output->get_back_matrix()->row(i).transpose() * inputs[0]->get_matrix()->row(i);
    weight_updater->update( &weight_nabla , k, output->shape[0]);
    
    MatrixRm m_input_back_tmp = output->get_back_matrix()->transpose() * (* (weight.get_matrix()));
    inputs[0]->set_back_data(this, &m_input_back_tmp);
    return 0;
}

vim_int32 Linear::alloc_trainable_variable_updaters()
{
    weight_updater = alloc_trainable_variable_updater(&weight);
    if(weight_updater == NULL)
    {
        assert(weight_updater != NULL);
        return -1;
    }
    bias_updater = alloc_trainable_variable_updater(&bias);
    if(bias_updater == NULL)
    {
        assert(bias_updater != NULL);
        callback_trainable_variable_updater(weight_updater);
        return -1;
    }
    return 0;
}

vim_int32 Linear::initialize_trainable_variables()
{
    vim_int32 res = initialize_variable(&weight);
    assert(res == 0 );
    res = initialize_variable(&bias);
    assert( res == 0 );
    return 0;
}

vim_int32 Linear::load(const char * path)
{
    string str = path;
    str += name; 
    weight.load((str+".weight.bin").c_str());
    bias.load((str+".bias.bin").c_str());
    return 0;
}

vim_int32 Linear::save(const char * path)
{
    string str = path;
    str += name; 
    weight.save((str+".weight.bin").c_str());
    bias.save((str+".bias.bin").c_str());
    return 0;
}   
