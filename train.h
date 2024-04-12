#ifndef __TRAIN_H__
#define __TRAIN_H__

#include "base_train.h"
#include "network.h"

class Train:public Base_train{
    public:
        Train(Network * network, Base_data_generator *dg_);
        ~Train();

        vector <VTensor*> build();

};


#endif
