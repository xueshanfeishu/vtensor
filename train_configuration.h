#ifndef __TRAIN_CONFIGURATION_H__
#define __TRAIN_CONFIGURATION_H__

#include "base_initializer.h"
#include "base_updater.h"

class Train_configuration{
    public:
        static Train_configuration& get_instance()
        {
            static Train_configuration config;      //Guaranteed to be destroyed.
            return config;
        };

        inline Base_initializer * get_initializer(){   return initiailizer;    };
        vim_int32 set_initializer(const char* init_name);
        
        inline Base_updater_factory * get_updater_factory(){    return updater_factory; };
        vim_int32 set_updater_factory(const char* updater_name);
        ~Train_configuration();
    private:
        Train_configuration();
        Train_configuration(Train_configuration const&) = delete;
        void operator=(Train_configuration const&) = delete;
        Base_initializer * initiailizer;
        Base_updater_factory * updater_factory;
};


#endif
