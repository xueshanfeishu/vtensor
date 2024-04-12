#ifndef __RANDN_INITIALIZER_H__
#define __RANDN_INITIALIZER_H__
#include <random>
#include "base_initializer.h"
#include "ndarray.h"
#include "types.h"

using namespace std;


class Randn_initializer:public Base_initializer{
    public:
        Randn_initializer(Real mean_, Real sigma_)
        {
            method = "constant";
            set_mean_sigma(mean_,sigma_);
        };
        ~Randn_initializer(){};

        vim_int32 initialize(Ndarray * var);

        inline void set_mean_sigma(Real mean_, Real sigma_)
        {
            mean = mean_;
            sigma = sigma_;
            using Params = std::normal_distribution<Real>::param_type;
            randn.param(Params {mean,sigma});
        }

    private:
        static normal_distribution<Real> randn;
        static Real generate_random(Real dummy);
        static default_random_engine random_engine;
        Real mean;
        Real sigma;
};

#endif
