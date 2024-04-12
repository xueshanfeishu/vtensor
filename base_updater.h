#ifndef __BASE_UPDATER_H__
#define __BASE_UPDATER_H__

#include <string>
#include "ndarray.h"

using namespace std;

class Base_updater{
    public:
        Base_updater(Ndarray * target_);
        ~Base_updater();
        virtual void update(MatrixRm * nabla, vim_uint64 k, vim_uint32 batch_size){};
        string method;
    protected:
        Ndarray * target;
};

class Base_updater_factory{
    public:
        Base_updater_factory(){};
        ~Base_updater_factory(){};
        virtual Base_updater * alloc_trainable_variable_updater(Ndarray * target_){return NULL;};
        virtual void callback(Base_updater * ) {};
    protected:
};


#endif