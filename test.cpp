// #include "train.h"
// #include "wavenet.h"
// #include "data_generator.h"
#include <string>
#include <iostream>
#include <assert.h>

using namespace std;

extern int set_param_dir(const char * path);
extern int load(const char * path);
extern int save(const char * path);
extern int generate(const char * path , int k);
extern float forward();
extern int backward(unsigned long long k);
extern int build();
extern int configure_train_env(const char* updater_name);

int main()
{
    string str = "../parameters/";
    set_param_dir(str.c_str());
    build();
    configure_train_env("adam");
    load("../parameters/1000/");
    generate("../data/",1);
    float loss = forward();
    // cout << loss << endl;
    // for(int i=1;i<15;i++)
    // {
    //     int kk = generate("../train_data/",6);
    //     if (kk == 0)
    //     {// to the end
    //         kk = generate("../train_data/",6);
    //     }
    //     loss = forward();
    //     backward(10000);
    //     // float loss = forward();
    //     cout << kk << "  " << loss << endl;
    //     // if ( (i%1000)==0 )
    //     // {
    //     //     string str1 = str + to_string(i);
    //     //     save(str1.c_str());
    //     // }
    // }
    // int k = generate("../data/",55);
    // loss = forward();
    // cout << loss << endl;
}
 // g++ ../test.cpp -o test -lvtensor -L .

