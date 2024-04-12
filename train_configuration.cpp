#include <string.h>
#include "train_configuration.h"
#include "randn_initializer.h"
#include "adam_updater.h"


Train_configuration::Train_configuration()
{
    initiailizer = NULL;
    updater_factory = NULL;
}

Train_configuration::~Train_configuration()
{
    if(initiailizer != NULL)
    {
        delete initiailizer;
        initiailizer = NULL;
    }

    if(updater_factory != NULL)
    {
        delete updater_factory;
        updater_factory = NULL;
    }
}

vim_int32 Train_configuration::set_initializer(const char* init_name)
{
    if(strcmp(init_name,"randn") == 0)
    {
        if(initiailizer != NULL)
            delete initiailizer;
        initiailizer = new Randn_initializer(0.0, 0.05);
        assert(initiailizer != NULL);
    }
    else
    {
        cout << init_name << " initializer is not supported now, EXIT!" << endl;
        assert(1==0);
    }

    return 0;
}


vim_int32 Train_configuration::set_updater_factory(const char* updater_name)
{
    if(strcmp(updater_name,"adam")==0)
    {
        if(updater_factory != NULL)
            delete updater_factory;
        updater_factory = new Adam_updater_factory();
        assert(updater_factory != NULL);
    }
    else
    {
        cout << updater_name << " updater is not supported now, EXIT!" << endl;
        assert(1==0);
    }
    return 0;
}

        