
#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"
#include "base_updater.h"
#include "base_initializer.h"

class Conv2d: public Node{
    public:
        Conv2d(vim_uint32 in_chan_, vim_uint32 out_chan_, HW ksize_, HW stride_, string padding_, HW delation_);
        Conv2d(vim_uint32 in_chan_, vim_uint32 out_chan_, HW ksize_, HW stride_, HW padding_, HW delation_);
        ~Conv2d();
        
        virtual vim_int32 prerun();                 // make graph
        vim_int32 forward();                // forward
        vim_int32 backward(vim_uint64 k);
        vim_int32 postrun();
        vim_int32 alloc_trainable_variable_updaters();          // 为训练准备相应的数据结构
        vim_int32 initialize_trainable_variables(); // 为所有训练变量初始化
        vim_int32 load(const char * path);                    // 恢复参数
        vim_int32 save(const char * path);                    // 保存参数
    protected:
    private:
        inline HW get_eff_ksize()
        {
            return  HW {delation.h*(ksize.h-1)+1, delation.w*(ksize.w-1)+1};
        };

        void cal_padding();
        vim_int32 im2col(Real* in_data, Real* tmp_data);
        vim_int32 back_col2im(Real* in_data, Real* tmp_data);

        Ndarray  weight;
        Base_updater * weight_updater;
        Ndarray  bias;
        Base_updater * bias_updater;
        
        HW ksize;
        HW stride;
        HW padding;
        HW delation;
        vector<vim_uint32>in_shape;
        vector<vim_uint32>out_shape;

        string padding_str;
};


#endif
