#include "node.h"
#include "train_configuration.h"

using namespace std;

Node::Node()
{
    output = new VTensor();
    assert( output );
    // output->name = name + ":0";   这个时候还没有name
    inputs.clear();
    parent = 0;
}

Node::~Node(){};

void Node::register_input_tensor(VTensor * tensor)
{
    inputs.push_back(tensor);
    tensor->to_nodes.push_back(this);
}

Base_updater * Node::alloc_trainable_variable_updater( Ndarray * target_)
{
    Train_configuration & tc = Train_configuration::get_instance();
    assert (tc.get_updater_factory() != NULL);
    Base_updater * updater = tc.get_updater_factory()->alloc_trainable_variable_updater(target_);
    assert (updater != NULL);
    return updater;
}

void Node::callback_trainable_variable_updater(Base_updater * updater_)
{
    Train_configuration & tc = Train_configuration::get_instance();
    assert (tc.get_updater_factory() != NULL);
    tc.get_updater_factory()->callback(updater_);
}

vim_int32 Node::initialize_variable(Ndarray *target_ )
{
    Train_configuration &tc = Train_configuration::get_instance();
    Base_initializer * init = tc.get_initializer();
    assert(init != NULL);
    return init->initialize(target_);
}

vector<VTensor * > Node::build(VTensor* input)
{
    register_input_tensor(input);
    output->name = name + ":0";
    output->owner = this;
    return {output};
}

vector<VTensor * > Node::build(VTensor* input1, VTensor* input2)
{
    register_input_tensor(input1);
    register_input_tensor(input2);
    output->owner = this;
    return {output};
}
