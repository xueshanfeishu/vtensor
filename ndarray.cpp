#include <fstream>
#include "ndarray.h"
#include "types.h"
#include "base_initializer.h"


Ndarray::Ndarray()
{
}

Ndarray::Ndarray(vector<vim_uint32> shape_)
{
    preset_shape(shape_);
}

Ndarray::Ndarray(vector<vim_uint32> shape_, Base_initializer * initializer_)
{
    shape = shape_;
    initialize(initializer_);
}

vim_int32 Ndarray::reshape(vector<vim_uint32> shape_)
{
    if(size() != size(shape_))
        assert (size() == size(shape_));
    shape = shape_;
    return 0;
};

Ndarray::~Ndarray()
{
}


vim_uint32 Ndarray::size(vector<vim_uint32> shape_)
{
    vim_uint32 dims = shape_.size();
    if (shape_.empty())
        return 0;

    vim_uint32 num = 1;
    for (vim_uint32 i = 0; i < dims; i++)
        num *= shape_[i];
    return num;
}

void Ndarray::print_shape()
{
    vim_uint32 dims = shape.size();
    cout<< "the Ndarray\'s shape size " << dims << "\n";
    cout<< "the Ndarray\'s shape is: ";
    if (dims > 1)
        for(vim_uint32 i=0;i<dims-1;i++)
            cout << shape[i] << " x ";
    if (dims > 0)
        cout << shape[dims-1] << "\n";        
}

void Ndarray::print_data()
{
       
}

void Ndarray::copy_from(void * data_, vector<vim_uint32> shape_)
{
    assert( data_ != NULL );
    assert( shape_.size() != 0 );
    shape = shape_;
    Map <MatrixRm> mm((Real*)data_, rows(), cols());
    m = mm;
}

vim_int32 Ndarray::initialize( Base_initializer * initializer)
{
    assert ( shape.size() != 0);
    initializer->initialize(this);
    return 0;
}

vim_int32 Ndarray::load(const char * path)
{
    // cout << "ndarray.cpp :80 loading "<< path <<endl;
    ifstream fin(path, ifstream::binary);
    assert(fin.is_open());
    Real * data = new Real [size()];
    assert( data != NULL);
    fin.read(reinterpret_cast<char*>(data), sizeof(Real)*size());
    copy_from(data, shape);
    assert(fin.gcount() == size()*sizeof(Real));    
    delete [] data;
    fin.close();
    return 0;
}

vim_int32 Ndarray::save(const char * path)
{
    ofstream fout(path,  ofstream::binary);
    assert(fout.is_open());
    Real * data = m.data();
    fout.write(reinterpret_cast<char*>(data), sizeof(Real)*size());
    fout.close();
    return 0;
}

// int main()
// {
//     vector<vim_uint32> shape = {3,4,5};
//     void * data = (void *)new Real [80];
//     Ndarray nd(EMUM_REAL);
//     nd.set_shape(shape);
//     nd.print_shape();
//     nd.set_data(data);
    
//     nd.reshape({5,4,3});
//     nd.print_shape();
//     nd.print_data();
// }

// g++ ndarray.cpp -o test -I eigen/




