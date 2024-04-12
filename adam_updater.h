#ifndef __ADAM_UPDATER_H__
#define __ADAM_UPDATER_H__

#include "base_updater.h"
#include "types.h"
#include "ndarray.h"

class Adam_updater:public Base_updater{
    public:
        Adam_updater(Ndarray * target_);
        ~Adam_updater();  
        void update(MatrixRm * nabla, vim_uint64 k, vim_uint32 batch_size);
        inline void set_param(Real epsilon_=0.001, Real p1_=0.9, Real p2_=0.999, Real delta_=1.0e-8)
            {
                epsilon = epsilon_;
                p1 = p1_;
                p2 = p2_;
                delta = delta_;
            };
    private:
        MatrixRm s;
        MatrixRm r;
        static Real epsilon;
        static Real p1;
        static Real p2;
        static Real delta;
};

class Adam_updater_factory:public Base_updater_factory{
    public:
        Adam_updater_factory();
        ~Adam_updater_factory();
        Base_updater * alloc_trainable_variable_updater(Ndarray * target_);
        void callback(Base_updater * );
};
#endif
