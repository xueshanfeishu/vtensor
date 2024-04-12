#include "const_initializer.h"

vim_int32 Constant_initializer::initialize(Ndarray * var){
    assert(var->shape.size() > 0);
    MatrixRm * m = var->get_matrix();
    *m = MatrixRm::Constant(m->rows(),m->cols(), const_value);
    return 0;
}
