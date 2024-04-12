#include<sys/types.h>       //linux
#include<sys/stat.h>       //linux
#include<dirent.h>          //linux
#include "base_train.h"
#include "train_configuration.h"

Base_train::Base_train(Network * network_, Base_data_generator *dg_)
{
    network = network_;
    dg_train = dg_;
}

Base_train::~Base_train()
{
    
}

vim_int32 Base_train::prerun()
{
    cout << "prerun ......" << endl;
    network->prerun();
    cout << "prerun successfully" << endl;
    return 0;
    
}

vim_int32 Base_train::forward()
{
    // cout << "forward ......" << endl;
    network->forward();
    // cout << "forward successfully" << endl;
    return 0;
}

vim_int32 Base_train::postrun()
{
    cout << "postrun ......" << endl;
    network->postrun();
    cout << "postrun successfully" << endl;
    return 0;
    
}

vim_int32 Base_train::backward(vim_uint64 k)
{
    // cout << "backward ......" << endl;
    network->backward(k);
    // cout << "backward successfully" << endl;
    return 0;
}

vim_uint64 Base_train::load()
{
    vim_int64 k,kmax = 0;
    struct dirent *entry;
    DIR * dp = opendir(param_dir.c_str());
    assert(dp != NULL);
    entry=readdir(dp);
    while(entry)
    {
        if(DIR * dp1 = opendir((param_dir + entry->d_name).c_str()))
        {//be sure it is a dir
            k = atoll(entry->d_name);
            if(k>kmax) 
                kmax = k;
            closedir(dp1);
        }   
        entry=readdir(dp);
    }
    closedir(dp);

    if(kmax == 0)
    {
        cout << "Cannot find parameter directory from " << param_dir << endl;
        return 0;
    }
    string str = param_dir + to_string(kmax) + "/";
    cout << "Loding parameters from " << str << "......" << endl;
    network->load(str.c_str());
    cout << "Loding parameters successfully" << endl;
    return kmax;
}

vim_int32 Base_train::load(const char * path)
{
    string str = path;
    str = str + "/";
    cout << "Loding parameters from " << str << "......" << endl;
    network->load(str.c_str());
    cout << "Loding parameters successfully" << endl;
    return 0;
}

vim_int32 Base_train::save(vim_uint64 k)
{
    string str = param_dir + to_string(k);
    DIR *dp = opendir(str.c_str());
    if(dp)
        closedir(dp);
    else
    {
        int res = mkdir(str.c_str(),0755);
        assert(res == 0);
    }
        
    
    cout << "Saving parameters to " <<str << endl;
    str += "/";
    network->save(str.c_str());
    cout << "Save parameters to " <<str << " successflully !" << endl;
    return 0;
}

vim_int32 Base_train::save(const char * path)
{
    string str = path;
    DIR *dp = opendir(str.c_str());
    if(dp)
    {
        closedir(dp);
    }
    else
    {
        int res = mkdir(str.c_str(),0755);
        assert(res == 0);
    }
    
    cout << "Saving parameters to " <<str << endl;
    str += "/";
    network->save(str.c_str());
    cout << "Save parameters to " <<str << " successflully !" << endl;
    return 0;
}

vim_int32 Base_train::initialize_trainable_variables()
{
    cout << "initialize trainable variables ......" << endl;
    network->initialize_trainable_variables();
    cout << "initialize trainable variables successfully" << endl;
    return 0;
}

vim_int32 Base_train::alloc_trainable_variable_updaters()
{
    cout << "alloc trainable variable updaters ......" << endl;
    network->alloc_trainable_variable_updaters();
    cout << "alloc trainable variable updaters successfully" << endl;
    return 0;
}

vim_int32 Base_train::run(  vim_uint32 batch_size, 
                            vim_uint32 epoch, 
                            vim_uint32 test_period, 
                            vim_uint32 save_period, 
                            const char * init_name, 
                            const char* updater_name,
                            const char * dataset_dir)
{
    build();
    vim_uint64 k = configure_train_env(init_name, updater_name);
    
    prerun();
    for(vim_uint32 epoch_=0;epoch_<epoch; epoch_++)
    {
        vim_uint64 count = 0;
        dg_train->reset();
        vim_uint32 real_num = dg_train->generate(dataset_dir,batch_size);
        while(real_num != 0)
        {
            count += real_num;
            k++;
            cout << "epoch:" << epoch_ << "  batch_size:" << batch_size << "  count:" << count << "  ";
            forward();
            backward(k);
            if( (k%save_period) == 0 )
                save(k);
            // if(k%test_period == 0)
            //     test();
            real_num = dg_train->generate(dataset_dir,batch_size);
        }
    }
    postrun();
    return 0;
}

vim_int32 Base_train::set_param_dir(const char * param_dir_)
{
    DIR * dp = opendir(param_dir_);
    assert( dp != NULL);
    if(mkdir(param_dir_, 0755) != 0)
        closedir(dp);
    
    param_dir = param_dir_;

    if(param_dir.at(param_dir.size()-1) != '/')
        param_dir.push_back('/');
    return 0;
}

vim_uint64 Base_train::configure_train_env(const char * init_name, const char* updater_name)
{
    vim_uint64 res;
    Train_configuration & conf = Train_configuration::get_instance();
    conf.set_initializer(init_name);
    conf.set_updater_factory(updater_name);
    assert(conf.get_initializer() != NULL);
    assert(conf.get_updater_factory() != NULL);
    res = load();
    if( res == 0)
        initialize_trainable_variables();
    alloc_trainable_variable_updaters();

    return res;
}

