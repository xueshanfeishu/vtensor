
#ifndef __DATA_GEBERATOR_H__
#define __DATA_GEBERATOR_H__

#include <fstream>
#include "base_data_generator.h"
#include "tensor.h"

using namespace std;

class Data_generator:public Base_data_generator{
    public:
        Data_generator();
        ~Data_generator();
        vim_int32 reset();
        vim_int32 generate(const char * path, vim_uint32 n);
        vector<vector <vim_uint32>> labels;
        vector<vim_uint32> label_length;
        vector<vim_uint32> logit_length;
    private:
        vim_int32 strsplit(char* str, const char * sep, vector<unsigned long> * vec);
        ifstream label_file;
        string mfcc_path;
};

#endif
