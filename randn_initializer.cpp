
#include <ctime>
#include <random>
#include "randn_initializer.h"

using namespace std;

default_random_engine Randn_initializer::random_engine(time(0));
normal_distribution<Real> Randn_initializer::randn(0,1);

Real Randn_initializer::generate_random(Real dummy){
    return randn(random_engine);
}

vim_int32 Randn_initializer::initialize(Ndarray * var)
{
    assert(var->shape.size() > 0);
    MatrixRm * m = var->get_matrix();
    *m = MatrixRm::Zero(var->rows(),var->cols()).unaryExpr(ptr_fun(generate_random));
    return 0;
}
