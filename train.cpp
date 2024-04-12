#include "train.h"
#include "wavenet.h"
#include "data_generator.h"

Train::Train(Network* network, Base_data_generator *dg_):Base_train(network,dg_)
{

}

Train::~Train()
{

}

vector <VTensor*> Train::build()
{
    Data_generator * dg = dynamic_cast <Data_generator *> (dg_train);
    Wavenet * net = dynamic_cast <Wavenet *> (network);
    return net->build(&(dg->labels), (dg->output), &(dg->label_length), &(dg->logit_length));
}

// int main()
// {
//     Wavenet net;
//     Data_generator dg;
//     Train train(&net, &dg);
//     train.set_param_dir("../parameters");
//     train.run(32,50,1000,10000,"randn", "adam");
// }


