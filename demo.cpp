#include "train.h"
#include "wavenet.h"
#include "data_generator.h"

static Wavenet net;
static Data_generator dg;
static Train train(&net, &dg);
static VTensor * tensor;


int set_param_dir(const char * path)
{
    return train.set_param_dir(path);
}

int load(const char * path)
{
    train.load(path);
}

int save(const char * path)
{
    train.save(path);
}

int generate(const char * path , int k)
{
    return dg.generate(path, k);
}

float forward()
{
    train.forward();
    // cout << * tensor->get_matrix()  << "  " << tensor->get_matrix()->sum() <<  "  " << tensor->get_matrix()->size() << "  "<< tensor->get_matrix()->sum()/tensor->get_matrix()->size() << endl;
    // ofstream fout;
    // string path = "/tmp/vtensor/";
    // path = path + "output.txt";
    // fout.open(path);
    // assert(fout.is_open());
    // fout << * tensor->get_matrix()  << "  " << tensor->get_matrix()->sum() <<  "  " << tensor->get_matrix()->size() << "  "<< tensor->get_matrix()->sum()/tensor->get_matrix()->size() << endl;
    // fout.close();
    
    return ( tensor->get_matrix()->sum()/tensor->get_matrix()->size() );
}

int backward(unsigned long long k)
{
    return train.backward(k);
}

int configure_train_env(const char* updater_name)
{
    train.configure_train_env("randn", updater_name);
    return 0;
}

int build()
{
    vector <VTensor *> tenvec = train.build();
    tensor = tenvec[0];
    train.configure_train_env("randn", "adam");
}





// int main()
// {
//     string str = "../parameters/";
//     train.set_param_dir(str.c_str());
//     build();
//     // load("../parameters/32400/");
//     for(int i=1;i<10000;i++)
//     {
//         generate("../data/",32);
//         forward();
//         // cout << "output data:" << endl;
//         // cout << *tensor->get_matrix() <<endl;
//         backward(10000);
//         if ( (i%1000)==0 )
//         {
//             string str1 = str + to_string(i);
//             save(str1.c_str());
//         }
//     }
    

// }


