
#ifndef __WAVWNET_H__
#define __WAVWNET_H__

#include "network.h"

class Res_block:public Network{
    public:
        Res_block();
        ~Res_block();
        vector <VTensor *>  build(VTensor * input);
    private:
};


class Wavenet:public Network{
    public:
        Wavenet();
        ~Wavenet();
        vector <VTensor *>  build(vector<vector <vim_uint32>> *labels, 
                                  VTensor * input ,
                                  vector <vim_uint32> *label_length, 
                                  vector <vim_uint32> *logit_length);
    private:
};

#endif
