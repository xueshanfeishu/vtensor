#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <vector>
#include <string>
#include "types.h"
#include "ndarray.h"
#include "node.h"

class Node;

class VTensor: public Ndarray{
    public:
        VTensor();
        VTensor(vector<vim_uint32> shape_);
        ~VTensor();

        void set_back_data_reuse();
        inline void * get_back_data(){return m_back.data();};
        inline MatrixRm * get_back_matrix(){return &m_back;};
        vim_int32 set_back_data(Node *node,MatrixRm * m_back_);

        Node* owner;                              // which node this tensor belongs to, only from one
        vector <Node*> to_nodes;                           // a tensor can to more than one nodes 
        string name;
    private:
        vim_uint32 to_node_flags;
        MatrixRm m_back;    
};


#endif