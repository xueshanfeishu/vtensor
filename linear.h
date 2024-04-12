
#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"
#include "base_updater.h"
#include "base_initializer.h"


class Linear: public Node{
    public:
        Linear(vim_uint32 in_elem_num, vim_uint32 out_elm_num);
        ~Linear();
        
        vim_int32 forward();                // forward
        vim_int32 backward(vim_uint64 k);
        vim_int32 alloc_trainable_variable_updaters();          // 为训练准备相应的数据结构
        vim_int32 initialize_trainable_variables(); // 为所有训练变量初始化
        vim_int32 load(const char * path);                    // 恢复参数
        vim_int32 save(const char * path);                    // 保存参数
    protected:
    private:
        Ndarray  weight;
        Base_updater * weight_updater;
        Ndarray  bias;
        Base_updater * bias_updater;
        
};


#endif
