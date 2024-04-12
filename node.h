#ifndef __NODE_H__
#define __NODE_H__

#include <vector>
#include <string>
#include "types.h"
#include "tensor.h"
#include "base_updater.h"
#include "base_initializer.h"

enum NODE_TYPE { TRAIN_TYPE = 1, INFER_TYPE = 2, TRAIN_INFER_TYPE = 3 };

class VTensor;

class Node{
    public:
        Node();
        ~Node();
        virtual vim_int32 prerun() {return 0;};          // 准备资源
        virtual vim_int32 forward(){return 0;};         // 前向计算
        virtual vim_int32 postrun() {return 0;};         // 销毁资源
        virtual vim_int32 backward(vim_uint64 k){return 0;};        // 反向传播
        virtual vim_int32 alloc_trainable_variable_updaters() {return 0;};        // 设置训练环境：为每个训练变量都分配一个训练器
        virtual vim_int32 load(const char * path){return 0;};                    // 恢复参数
        virtual vim_int32 save(const char * path) {return 0;};                    // 保存参数
        virtual vim_int32 initialize_trainable_variables() {return 0;};

        inline virtual void set_name(const char* name_){ name = name_;};
        inline virtual void set_type(NODE_TYPE ntype_){ntype = ntype_;};
        void register_input_tensor(VTensor * tensor);
        virtual vector<VTensor * > build(VTensor* input);
        virtual vector<VTensor * > build(VTensor* input1 ,VTensor* input2);

        string name;                        // 当前节点的名字
        string op_name;                     // 当前节点的操作符名字
        vector <VTensor * > inputs;         // 记录当前node的所有input tensor，这样就可以自动找到input了
        VTensor *output;
        Node * parent;
    protected:
        vim_int32 initialize_variable(Ndarray *target_);   // 为所有训练变量初始化。
        Base_updater * alloc_trainable_variable_updater( Ndarray * target_);
        void callback_trainable_variable_updater( Base_updater * updater_);
        NODE_TYPE ntype;
    private:
};

#endif
