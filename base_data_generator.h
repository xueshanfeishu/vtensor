#ifndef __BASE_DATA_GENERATOR_H__
#define __BASE_DATA_GENERATOR_H__

#include "types.h"
#include "node.h"

class Base_data_generator:public Node{
    public:
        Base_data_generator()
        {
            name = "input";
            op_name = "input";
        };
        
        ~Base_data_generator(){};

        virtual vim_int32 generate(const char * path, vim_uint32 n) {return 0;};
        virtual vim_int32 reset() {return 0;};

        vim_int32 forward(){return 0;};                // forward
        vim_int32 backward(vim_uint64 k){return 0;};
    protected:
    private:
};


#endif
