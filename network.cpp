#include <iostream>
#include <fstream>
#include "network.h"

using namespace std;

Network::Network():Node()
{

}

Network::~Network()
{
    while( !nodes.empty() )
    {
        Node * node = nodes.back();
        nodes.pop_back();
        delete node;
    }
}

vim_int32 Network::prerun()
{//
    for(vector<Node*>::iterator it = nodes.begin(); it!= nodes.end(); it++)
        (*it)->prerun();
    return 0;
}

vim_int32 Network::forward()
{
    for(vector<Node*>::iterator it = nodes.begin(); it!= nodes.end(); it++)
    {
        // cout << (*it)->name << " forwarding ..." << endl;        
        (*it)->forward();
        ofstream fout;
        string path = "/tmp/vtensor/";
        path = path + (*it)->name + ".txt";
        fout.open(path);
        assert(fout.is_open());
        fout << * (*it)->output->get_matrix() << endl;   
        fout.close();
    }
    return 0;
}

vim_int32 Network::postrun()
{
    for(vector<Node*>::iterator it = nodes.begin(); it!= nodes.end(); it++)
        (*it)->postrun();
    return 0;
}

vim_int32 Network::backward(vim_uint64 k)
{
    for(vector<Node*>::reverse_iterator rit = nodes.rbegin(); rit!= nodes.rend(); rit++)
    {
        // cout << (*rit)->name << " backwarding ..." << endl;
        (*rit)->backward(k);
    }
    return 0;
}

vim_int32 Network::alloc_trainable_variable_updaters()
{
    for(vector<Node*>::iterator it = nodes.begin(); it!= nodes.end(); it++)
    {
        // cout << "\t" << " alloc_trainable_variable_updaters for "<< (*it)->name  << endl;
        (*it)->alloc_trainable_variable_updaters();
    }
    return 0;
}

vim_int32 Network::initialize_trainable_variables()
{
    for(vector<Node*>::iterator it = nodes.begin(); it!= nodes.end(); it++)
    {
        // cout << "\t" << (*it)->name << " initialize trainable variables for" << (*it)->name << endl;
        (*it)->initialize_trainable_variables();
    }
    return 0;
}

void Network::sort_nodes()
{
    for (vim_int32 i = nodes.size()-2; i>=0; i--)
    {
        for (vim_int32 j = nodes.size()-2; j>i; j--)
        {
            for(vim_int32 k=nodes[i]->inputs.size() - 1; k>=0; k--)
            {
                for(Node* node=nodes[i]->inputs[k]->owner; node!=0;node=node->parent)
                {
                    if(node == nodes[j])
                    {// find one of node j's input is behind node j
                        // it is dangerous . be careful
                        // cout << "node["<<i<<"] k:" <<k << nodes[i]->name <<"  nodes[j]:" << "node["<<j<<"]" << nodes[j]->name <<endl;
                        
                        Node* pNode = nodes[j];
                        nodes.erase(nodes.begin()+j);
                        nodes.insert(nodes.begin()+i,pNode);
                        i++;
                        break;
                    }
                }
            }
        }
    }
}

vim_int32 Network::register_node(Node * node, const char* name_)
{
    string tmp_name;
    if(this->name.length() >0 )
        tmp_name = this->name + '.' + name_;
    else
        tmp_name = name_;
    for (vim_int32 i = nodes.size()-1; i>=0; i--)
    {
        if(nodes[i]->name == tmp_name )
        {
            assert(nodes[i]->name != tmp_name);
            return -1;
        }
    }
    node->set_name(tmp_name.c_str());
    nodes.push_back(node);
    node->parent = this;
    return 0;
}


vim_int32 Network::load(const char * path)
{
    for(vector<Node*>::iterator it = nodes.begin(); it!= nodes.end(); it++)
    {
        // cout << "\t loading parameters for " << (*it)->name << endl;
        (*it)->load(path);
    }
    return 0;
}

vim_int32 Network::save(const char * path)
{
    for(vector<Node*>::iterator it = nodes.begin(); it!= nodes.end(); it++)
        (*it)->save(path);
    return 0;
} 

