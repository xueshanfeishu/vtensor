
#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "node.h"
#include "tensor.h"
#include "ndarray.h"

#define Define_Node(obj_ptr, Class_name, Class_param)    \
        Class_name * obj_ptr = new Class_name Class_param; \
        assert(obj_ptr != NULL);   \
        res = register_node( obj_ptr,  #obj_ptr); \
        assert(res  == 0); 

class Network: public Node{
    public:
        Network();
        ~Network();
        
        vim_int32 initialize_trainable_variables();
        vim_int32 alloc_trainable_variable_updaters();
        vim_int32 prerun();                 // make graph
        vim_int32 forward();                // forward
        vim_int32 postrun();
        vim_int32 backward(vim_uint64 k);
        vim_int32 load(const char * path);                    // 恢复参数
        vim_int32 save(const char * path);                    // 保存参数
    protected:
        void sort_nodes();
        vim_int32 register_node(Node * node, const char* name_);
    private:
        vector <Node * > nodes;
};


#endif
