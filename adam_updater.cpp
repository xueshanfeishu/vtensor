#include "adam_updater.h"
#include <math.h>

Real Adam_updater::epsilon = 0.001;
Real Adam_updater::p1 = 0.9;
Real Adam_updater::p2 = 0.999;
Real Adam_updater::delta = 1.0e-8;

Adam_updater::Adam_updater(Ndarray * target_):Base_updater(target_)
{
    s = MatrixRm::Zero(target_->rows(), target_->cols());
    r = MatrixRm::Zero(target_->rows(), target_->cols());
}

Adam_updater::~Adam_updater()
{
}

void Adam_updater::update(MatrixRm * nabla,vim_uint64 k, vim_uint32 batch_size)
{
    if(k>0)
    {
        MatrixRm nabla1 = (1.0f/batch_size) * (*nabla);
        s = p1 * s + nabla1 * (1.0f - p1);
        r = p2 * r.array() + (1.0f - p2) * nabla1.array() * nabla1.array() ;
        Real epslion_t = epsilon * sqrt(1.0 - pow(p2, k)) / (1.0 - pow(p1, k)) ;
        MatrixRm * m_target = target->get_matrix();
        m_target->array() -= epslion_t * s.array() / (r.array().sqrt() + delta);
    }
}

Adam_updater_factory::Adam_updater_factory()
{

}

Adam_updater_factory::~Adam_updater_factory()
{

}

Base_updater * Adam_updater_factory::alloc_trainable_variable_updater(Ndarray * target_)
{
    return dynamic_cast<Base_updater *> (new Adam_updater(target_));
}

void Adam_updater_factory::callback(Base_updater * updater)
{
    delete updater;
}


