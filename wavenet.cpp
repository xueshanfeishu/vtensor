#include <vector>
#include "conv2d.h"
#include "wavenet.h"
#include "tanh.h"
#include "sigmoid.h"
#include "add2.h"
#include "mul2.h"
#include "ctc_loss.h"
#include "depthwise_conv2d.h"


Res_block::Res_block(){

}

Res_block::~Res_block(){
    
}

vector <VTensor *> Res_block::build( VTensor * input)
{
    vector<VTensor * > x1, x1_1;
    vim_int32 res;

    register_input_tensor(input);
    Define_Node(depthwise1, Depthwise_conv2d, (128, 1, {7,1}, {1,1}, (HW) {6,0}, {1,1}));
    Define_Node(tanh1, Tanh, ());
    Define_Node(conv1, Conv2d, (128, 128, {1,1}, {1,1}, (HW){0,0}, {1,1}));
    Define_Node(depthwise1_1, Depthwise_conv2d, (128, 1, {7,1}, {1,1}, (HW) {6,0}, {1,1}));
    Define_Node(sigmoid1_1, Sigmoid, ());
    Define_Node(conv1_1, Conv2d, (128, 128, {1,1}, {1,1}, (HW){0,0}, {1,1}));
    Define_Node(mul, Mul2,());
    Define_Node(conv2, Conv2d, (128, 128, {1,1}, {1,1}, (HW){0,0}, {1,1}));
    Define_Node(tanh2, Tanh, ());

    x1 = depthwise1->build(input);
    x1 = conv1->build(x1[0]);
    x1 = tanh1->build(x1[0]);

    x1_1 = depthwise1_1->build(input);
    x1_1 = conv1_1->build(x1_1[0]);
    x1_1 = sigmoid1_1->build(x1_1[0]);

    x1 = mul->build(x1[0], x1_1[0]);
    x1 = conv2->build(x1[0]);
    x1 = tanh2->build(x1[0]);

    sort_nodes();
    return x1;
}

Wavenet::Wavenet(){

}

Wavenet::~Wavenet(){
    
}

vector <VTensor *> Wavenet::build(  vector<vector <vim_uint32>> *labels, 
                                    VTensor * input ,
                                    vector <vim_uint32> *label_length, 
                                    vector <vim_uint32> *logit_length)
{
    vim_int32 res;
    vim_uint32 blank_index = 0;
    vector<VTensor * > x, res3, res4, res5, res6, res7;
    register_input_tensor(input);
    Define_Node(conv1, Conv2d, (20, 64, {3,1}, {2,1}, (HW) {2,0}, {1,1}));
    Define_Node(tanh1, Tanh, ());
    Define_Node(conv2, Conv2d, (64, 128, {3,1}, {2,1}, (HW) {2,0}, {1,1}));
    Define_Node(tanh2, Tanh, ());

    Define_Node(addx0, Add2, ());
    Define_Node(addx1, Add2, ());
    Define_Node(addx2, Add2, ());
    Define_Node(addx3, Add2, ());
    Define_Node(add3, Add2, ());
    Define_Node(add4, Add2, ());
    Define_Node(add5, Add2, ());
    Define_Node(add6, Add2, ());

    Define_Node(resblk3, Res_block, ());
    Define_Node(resblk4, Res_block, ());
    Define_Node(resblk5, Res_block, ());
    Define_Node(resblk6, Res_block, ());
    Define_Node(resblk7, Res_block, ());
    Define_Node(conv8, Conv2d, (128, 128, {1,1}, {1,1}, (HW){0,0}, {1,1}));
    Define_Node(tanh8, Tanh, ());
    Define_Node(conv9, Conv2d, (128, 46, {1,1}, {1,1}, (HW){0,0}, {1,1}));

    Define_Node(ctc, Ctc_loss, (0, blank_index));

    x = conv1->build(input);
    x = tanh1->build(x[0]);
    x = conv2->build(x[0]);
    x = tanh2->build(x[0]);

    res3 = resblk3->build(x[0]);
    x = add3->build(x[0], res3[0]);
    res4 = resblk4->build(x[0]);
    x = add4->build(x[0], res4[0]);
    res5 = resblk5->build(x[0]);
    x = add5->build(x[0], res5[0]);
    res6 = resblk6->build(x[0]);
    x = add6->build(x[0], res6[0]);
    res7 = resblk7->build(x[0]);

    x = addx0->build(res3[0], res4[0]);  
    x = addx1->build(x[0], res5[0]);
    x = addx2->build(x[0], res6[0]);
    x = addx3->build(x[0], res7[0]);        // x = res3 + res4 + res5 + res6 + res7

    x = conv8->build(x[0]);
    x = tanh8->build(x[0]);
    x = conv9->build(x[0]);
    x = ctc->build(labels, x[0], label_length, logit_length);

    sort_nodes();
    return x;
}



