#ifndef __BASE_TRAIN_H__
#define __BASE_TRAIN_H__

#include <string>
#include "base_data_generator.h"
#include "network.h"
#include "types.h"

class Base_train{
    public:
        Base_train(Network * network, Base_data_generator *dg_);
        ~Base_train();

        vim_int32 set_param_dir(const char * param_dir_);
        vim_int32 run(  vim_uint32 batch_size, 
                        vim_uint32 epoch, 
                        vim_uint32 test_period, 
                        vim_uint32 save_period, 
                        const char * init_name, 
                        const char* updater_name,
                        const char * dataset_dir);
        vim_int32 postrun();
        vim_int32 backward(vim_uint64 k);
        vim_uint64 configure_train_env(const char * init_name, const char* updater_name);
        vim_uint64 load();                    // 恢复参数
        vim_int32 load(const char * path );                    // 恢复参数
        vim_int32 save(vim_uint64 k);                    // 保存参数
        vim_int32 save(const char * path);                    // 保存参数
        virtual vector <VTensor*> build() 
        { 
            vector <VTensor*> no_use;
            return no_use;
        };
        vim_int32 forward();
    protected:
        vim_int32 prerun();
        vim_int32 initialize_trainable_variables();
        vim_int32 alloc_trainable_variable_updaters();
        Network * network;
        Base_data_generator * dg_train;
        Base_data_generator * dg_test;
        string param_dir;
    private:
};


#endif
