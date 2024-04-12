#include <fstream>
#include <iostream>
#include <vector>
#include "data_generator.h"

using namespace std;
#define INPUT_HMAX 800
#define INPUT_CHAN 20
Data_generator::Data_generator()
{
    // string mfcc_path = "";
}

Data_generator::~Data_generator()
{
    if(label_file.is_open())
    {
        label_file.close();
    }
}

vim_int32 Data_generator::reset()
{
    label_file.seekg(0,ios::beg);
    return 0;
}

vim_int32 Data_generator::strsplit(char* str, const char * sep, vector<unsigned long> * vec)
{
    char *p;
    vec->clear();
    p = strtok(str, sep);
    while(p){
        vec->push_back(atol(p));
        p = strtok(NULL, sep);
    }  
    return 0;
}

vim_int32 Data_generator::generate(const char * path, vim_uint32 n)
{
    if(path == NULL)
    {
        return 0;
    }
    else 
    {
        string path_ = path;
        if (path_ != mfcc_path)
        {
            if(label_file.is_open())
            {
                label_file.close();
            }
            mfcc_path = path_;
            label_file.open(mfcc_path + "train_out.txt");
            assert(label_file.is_open());
        }
        labels.clear();
        label_length.clear();
        logit_length.clear();
        vector<unsigned long> vec;
        vim_uint32 count = 0;
        Real * data = new Real [INPUT_HMAX*INPUT_CHAN*n]; 
        memset(data, 0 , INPUT_HMAX*INPUT_CHAN*n*sizeof(Real) );
        Real * dat = data;
        for(int i=0;i<n;i++)
        {
            char id[16];
            char label[256];
            memset(id,0,16);
            memset(label,0,256);
            label_file >> id;
            label_file >> label;
            cout << id << "   "<< label << endl;
            if(strlen(id)>0 && strlen(label)>0)
            {
                strsplit(label,",",&vec);
                labels.push_back(vec);
                label_length.push_back(vec.size());

                ifstream fmfcc(mfcc_path + "mfcc32/" + id + ".bin",std::ifstream::binary);
                assert(fmfcc.is_open());
                fmfcc.read(reinterpret_cast<char*>(data + INPUT_HMAX*INPUT_CHAN*i), sizeof(Real)*INPUT_HMAX*INPUT_CHAN);
                dat += INPUT_HMAX*INPUT_CHAN;
                logit_length.push_back( (fmfcc.gcount() +3) /4);
                fmfcc.close();
                count++;
            }
            else
            {
                label_file.clear();
                label_file.seekg(0,ios::beg);
                break;
            }
        }
        output->copy_from(data, {count,INPUT_HMAX,1,INPUT_CHAN});
        delete [] data;
        return count;
    }
}

// int main()
// {
//     Data_generator dg;
//     dg.generate(2);
//     return 0;
// }

// g++ datagenerator.cpp dg

