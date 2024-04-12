#include "tensor.h"
#include "types.h"
#include <assert.h> 


VTensor ::VTensor():Ndarray()
{
    to_node_flags = 0xFFFFFFFF;
}

VTensor::VTensor(vector<vim_uint32> shape_):Ndarray(shape_)
{
    to_node_flags = 0;
}

void VTensor::set_back_data_reuse()
{
    
}
 
VTensor ::~VTensor()
{
   
}

/*
 * set_back_data(Node * node, void * data)
 * return -1: exception. the node cannot be found
 * return 0:  the back_data has not update by the node
 *            but they have been updated by other node,
 *            so the back_data need to be updated by being added by a 
 *            gradient but not replaced.then or the flag
 * return 1:  the back_data have not been updated by the node,
 *            and this node is the first node to update them.
 *            so just replace them with the new gradient
 *            then set the flag with just this node
 * 
 * caution: not thread safe. 
 *  
 */
vim_int32 VTensor::set_back_data(Node *node,MatrixRm * m_back_)
{
    vim_int32 i;

    for(i=to_nodes.size()-1; i>=0; i--)
        if(to_nodes[i] == node)
            break;
    if(i<0)
    {
        assert( i>=0 );
        return -1;
    } 
    
    vim_uint32 flag = (to_node_flags & (1<<i))? 1:0 ;
    if( flag )
    {
        to_node_flags = 1<<i;
        m_back = *m_back_;
    }
    else
    {
        to_node_flags |= 1<<i;
        m_back += *m_back_;
    }    
    return flag;
}

// int main()
// {
//     vector<vim_uint32> shape = {3,4,5};
//     Real * data = new Real [80];
//     cout <<"11111"<<endl;
//     VTensor vt;
//     cout <<"2222 "<< data << endl;
//     vt.copy_from(data,shape);

//     vt.print_shape();
    
//     vt.reshape({5,4,3});
//     vt.print_shape();
//     vt.print_data();
// }

// g++ tensor.cpp ndarray.cpp -o test -I eigen/
