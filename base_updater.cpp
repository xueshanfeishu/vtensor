#include "base_updater.h"

Base_updater::Base_updater( Ndarray * target_)
{
    target = target_;
    method = "none";
}

Base_updater::~Base_updater()
{
    
}
