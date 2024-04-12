#ifndef __NDARRAY_H__
#define __NDARRAY_H__


#include <iostream>
#include <assert.h> 
#include <vector>
#include <Eigen/Dense>
#include "types.h"
#include "base_initializer.h"

class Base_initializer;

using Eigen::Matrix;
using Eigen::Map;
using Eigen::RowMajor;
using Eigen::Dynamic;
using Eigen::VectorXf;

using namespace std;

typedef Matrix<Real, Dynamic, Dynamic, RowMajor> MatrixRm;
typedef Matrix<Real, Dynamic, 1> VectorRm;
typedef Matrix<Real, 1, Dynamic> RowVectorRm;


class Ndarray{
    public:
        Ndarray();
        Ndarray(vector<vim_uint32> shape_);
        Ndarray(vector<vim_uint32> shape_,Base_initializer * initializer);
        ~Ndarray();
        inline vim_uint32 size()   {   return size(shape);   };
        vim_uint32 size(vector<vim_uint32> shape_);
        inline void preset_shape(vector<vim_uint32> shape_) {shape = shape_;};
        vim_int32 reshape(vector<vim_uint32> shape_);
        inline MatrixRm * get_matrix()   {   return &m;   };
        void copy_from(void * data_, vector<vim_uint32> shape_);
        vim_int32 initialize(Base_initializer * initializer);
        inline void setZero(vector<vim_uint32> shape_){
            shape = shape_;
            m = MatrixRm::Zero(rows(),cols());
        };

        inline void setZero(){
            m = MatrixRm::Zero(rows(),cols());
        };

        inline vim_uint32 cols(){
            if(shape.size() == 0)
                return 0;
            else if(shape.size() == 1)
                return 1;
            else
                return shape.back();
        };

        inline vim_uint32 rows(){
            if(shape.size() == 0)
                return 0;
            else if(shape.size() == 1)
                return shape[0];
            else
            {
                vim_uint32 row = 1;
                for(vim_int32 i=shape.size()-2; i>=0; i--)
                {
                    row *= shape[i];
                }
                return row;
            }
        };

        vim_int32 load(const char * path);
        vim_int32 save(const char * path);

        void print_shape();
        void print_data();
        vector<vim_uint32> shape;                       // max shape size
        DTYPE dtype;
    protected:
        MatrixRm m;        
        
    private:
};

#endif