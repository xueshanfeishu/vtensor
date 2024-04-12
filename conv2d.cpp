
#include "conv2d.h"
#include "train_configuration.h"

Conv2d::Conv2d(vim_uint32 in_chan_, vim_uint32 out_chan_, HW ksize_, HW stride_, string padding_, HW delation_):Node()
{
    assert(padding_ == "SAME" || padding_ == "VALID");
    ksize = ksize_;
    stride = stride_;
    delation = delation_;
    padding_str = padding_;
    weight.preset_shape({ksize.h, ksize.w, in_chan_, out_chan_});
    bias.preset_shape({1, out_chan_});
    this->op_name = "Conv2d";
    weight_updater = NULL;
    bias_updater = NULL;
}

Conv2d::Conv2d(vim_uint32 in_chan_, vim_uint32 out_chan_, HW ksize_, HW stride_, HW padding_, HW delation_):Node()
{
    ksize = ksize_;
    stride = stride_;
    delation = delation_;
    padding = padding_;
    padding_str = "none";
    weight.preset_shape({ksize.h, ksize.w, in_chan_, out_chan_});
    bias.preset_shape({1, out_chan_});
    this->op_name = "Conv2d";
    weight_updater = NULL;
    bias_updater = NULL;
}

Conv2d::~Conv2d()
{
    if(weight_updater)
        callback_trainable_variable_updater(weight_updater);
    if(bias_updater)
        callback_trainable_variable_updater(bias_updater);
}


vim_int32 Conv2d::prerun()
{//set output dimision, because forward() is processed partly, so the dimision of output must be set in advance
    // assert (inputs[0]->shape.size() == 4);
    // assert (inputs[0]->shape[3] == weight.shape[2]);

    // in_shape = inputs[0]->shape;
    // vim_uint32 out_h = (in_shape[1] -1) / stride.h + 1;
    // vim_uint32 out_w = (in_shape[2] -1) / stride.h + 1;
    // out_shape = {in_shape[0], out_h, out_w, weight.shape[3]};
    // output->setZero(out_shape);
    // cal_padding();
    return 0;
}

void Conv2d::cal_padding()
{
    if(padding_str == "VALID")
    {
        padding = HW {0,0};
    }
    else if(padding_str == "SAME")
    {
        HW eff_ksize = get_eff_ksize();
        padding.h = ((out_shape[1] -1) * stride.h + eff_ksize.h - in_shape[1]) >> 1;  
        padding.w = ((out_shape[2] -1) * stride.w + eff_ksize.w - in_shape[2]) >> 1;  
    }
}

vim_int32 Conv2d::im2col(Real* in_data, Real* tmp_data)
{
    Map <MatrixRm> img(in_data, in_shape[1]*in_shape[2], in_shape[3]);
    Map <MatrixRm> mat(tmp_data, out_shape[1]*out_shape[2]*ksize.h*ksize.w, in_shape[3]);
    vim_int32 in_idx,out_idx;
    for(vim_uint32 oh=0; oh<out_shape[1]; oh++)
    {
        vim_int32 ih1 = oh*stride.h - padding.h;
        vim_int32 out_idx1 = oh*out_shape[2];
        for(vim_uint32 ow=0; ow<out_shape[2]; ow++)
        {
            vim_int32 iw1 = ow*stride.w - padding.w;
            vim_int32 out_idx2 = (out_idx1 + ow)*ksize.h;
            for(vim_uint32 kh=0; kh<ksize.h; kh++)
            {
                vim_int32 ih2 = ih1 + delation.h*kh;
                vim_int32 out_idx3 = (out_idx2 + kh)*ksize.w;
                for(vim_uint32 kw=0; kw<ksize.w; kw++)
                {
                    vim_int32 iw2 = iw1 + delation.w*kw;
                    out_idx = out_idx3 + kw;
                    if(ih2>=0 && ih2<in_shape[1] && iw2>=0 && iw2<in_shape[2])
                    {
                        in_idx = ih2*in_shape[2] + iw2;
                        mat.row(out_idx) = img.row(in_idx);
                    }
                    else
                    {
                        mat.row(out_idx).setZero();
                    }
                }
            }
        }
    }
    return 0;
}

vim_int32 Conv2d::forward()
{
    //  根据input_tensor和weight bias来计算output_tensor
    assert (inputs[0]->shape.size() == 4);
    assert (inputs[0]->shape[3] == weight.shape[2]);

    in_shape = inputs[0]->shape;
    vim_uint32 out_h = (in_shape[1] -1) / stride.h + 1;
    vim_uint32 out_w = (in_shape[2] -1) / stride.h + 1;
    out_shape = {in_shape[0], out_h, out_w, weight.shape[3]};
    output->setZero(out_shape);
    cal_padding();

    Real * in_data = inputs[0]->get_matrix()->data();
    Real * out_data = output->get_matrix()->data();
    vim_uint32 in_ninterval = in_shape[1]*in_shape[2]*in_shape[3];
    vim_uint32 out_ninterval = out_shape[1]*out_shape[2]*out_shape[3];
    MatrixRm m_tmp(out_shape[1]*out_shape[2], ksize.h*ksize.w*in_shape[3]);

    for(vim_uint32 n=0;n<in_shape[0];n++)
    {
        im2col(in_data + n*in_ninterval, m_tmp.data());
        Map <MatrixRm> out_tmp(out_data + n*out_ninterval, out_shape[1]*out_shape[2], out_shape[3]);
        out_tmp = m_tmp * ( * (weight.get_matrix()) );
    }
    Map <RowVectorRm> bias_rvect(bias.get_matrix()->data(), out_shape[3]);
    output->get_matrix()->rowwise() += bias_rvect;
    // cout << output->get_matrix()->col(0) <<endl;
    return 0;
}

vim_int32 Conv2d::postrun()
{
    return 0;
}

vim_int32 Conv2d::back_col2im(Real* in_data, Real* tmp_data)
{
    Map <MatrixRm> img(in_data, in_shape[1]*in_shape[2], in_shape[3]);
    Map <MatrixRm> mat(tmp_data, out_shape[1]*out_shape[2]*ksize.h*ksize.w, in_shape[3]);
    img.setZero();
    vim_int32 in_idx,out_idx;
    for(vim_uint32 oh=0; oh<out_shape[1]; oh++)
    {
        vim_int32 ih1 = oh*stride.h - padding.h;
        vim_int32 out_idx1 = oh*out_shape[2];
        for(vim_uint32 ow=0; ow<out_shape[2]; ow++)
        {
            vim_int32 iw1 = ow*stride.w - padding.w;
            vim_int32 out_idx2 = (out_idx1 + ow)*ksize.h;
            for(vim_uint32 kh=0; kh<ksize.h; kh++)
            {
                vim_int32 ih2 = ih1 + delation.h*kh;
                vim_int32 out_idx3 = (out_idx2 + kh)*ksize.w;
                for(vim_uint32 kw=0; kw<ksize.w; kw++)
                {
                    vim_int32 iw2 = iw1 + delation.w*kw;
                    out_idx = out_idx3 + kw;
                    if(ih2>=0 && ih2<in_shape[1] && iw2>=0 && iw2<in_shape[2])
                    {
                        in_idx = ih2*in_shape[2] + iw2;
                        img.row(in_idx) += mat.row(out_idx);
                    }
                }
            }
        }
    }
    return 0;
}

vim_int32 Conv2d::backward(vim_uint64 k)
{
    MatrixRm bias_nabla = output->get_back_matrix()->colwise().sum();
    bias_updater->update( &bias_nabla, k, in_shape[0] );

    MatrixRm weight_nabla = MatrixRm::Zero(weight.shape[0]*weight.shape[1]*weight.shape[2],weight.shape[3]);
    Real * in_data = inputs[0]->get_matrix()->data();
    MatrixRm in_back = MatrixRm::Zero(inputs[0]->rows(), inputs[0]->cols());
    Real * out_back_data = output->get_back_matrix()->data();
    vim_uint32 in_ninterval = in_shape[1]*in_shape[2]*in_shape[3];
    vim_uint32 out_ninterval = out_shape[1]*out_shape[2]*out_shape[3];
    MatrixRm m_tmp(out_shape[1]*out_shape[2], ksize.h*ksize.w*in_shape[3]);   
    MatrixRm m_back_tmp;   
    for(vim_uint32 n=0;n<in_shape[0];n++)
    {
        im2col(in_data + n*in_ninterval, m_tmp.data());
        Map <MatrixRm> out_back_tmp(out_back_data + n*out_ninterval, out_shape[1]*out_shape[2], out_shape[3]);
        weight_nabla += m_tmp.transpose() * out_back_tmp;
        m_back_tmp = out_back_tmp * weight.get_matrix()->transpose();
        back_col2im(in_back.data() + n*in_ninterval, m_back_tmp.data());
    }
    weight_updater->update( &weight_nabla , k, in_shape[0]);
    inputs[0]->set_back_data(this, &in_back);
    return 0;
}

vim_int32 Conv2d::alloc_trainable_variable_updaters()
{
    weight_updater = alloc_trainable_variable_updater(&weight);
    if(weight_updater == NULL)
    {
        assert(weight_updater != NULL);
        return -1;
    }
    bias_updater = alloc_trainable_variable_updater(&bias);
    if(bias_updater == NULL)
    {
        assert(bias_updater != NULL);
        callback_trainable_variable_updater(weight_updater);
        return -1;
    }
    return 0;
}

vim_int32 Conv2d::initialize_trainable_variables()
{
    vim_int32 res = initialize_variable(&weight);
    assert( res == 0 );
    res = initialize_variable(&bias);
    assert( res == 0 );
    return 0;
}

vim_int32 Conv2d::load(const char * path)
{
    string str = path;
    str += name; 
    weight.load((str+".weight.bin").c_str());
    bias.load((str+".bias.bin").c_str());
    // if(name == "conv9")
    // {
    //     cout << *bias.get_matrix() << endl;
    // }
    return 0;
}

vim_int32 Conv2d::save(const char * path)
{
    string str = path;
    str += name; 
    weight.save((str+".weight.bin").c_str());
    bias.save((str+".bias.bin").c_str());
    return 0;
}                    
