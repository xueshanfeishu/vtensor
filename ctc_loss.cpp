#include <math.h>
#include "ctc_loss.h"

using namespace std;

Ctc_loss::Ctc_loss(vim_uint32 logits_time_major_, vim_uint32 blank_index_):Node()
{
    this->op_name = "Ctc_loss";
    logits_time_major = logits_time_major_;
    blank_index = blank_index_;
}

Ctc_loss::~Ctc_loss()
{
}


/**************************************************************
 * labels:labels
 * logits  T,N,C
 * label_length
 * logit_length
 **************************************************************/
vim_int32 Ctc_loss::forward()
{
    log_softmax_logits = inputs[0]->get_matrix()->array().exp();
    log_softmax_logits = (log_softmax_logits.array().colwise() / log_softmax_logits.rowwise().sum().array()).log();
    T = logits_time_major? inputs[0]->shape[0]:inputs[0]->shape[1];     //logits_time_major == true
    N = logits_time_major? inputs[0]->shape[1]:inputs[0]->shape[0];     //logits_time_major == true
    C = inputs[0]->shape[3];            //  XXXXXXXX  input NHWC to be fixed
    *output->get_matrix() = MatrixRm::Zero(N,1);
    for_blanks.resize(N);
    alphas.resize(N);
    for(vim_uint32 n=0;n<N;n++)
    {
        vector <vim_uint32> label = (*labels)[n];
        vector<vim_uint32>::iterator it=label.begin();
        while(it != label.end()){
            if(*it == blank_index){
                it = label.erase(it);
            }else{
                it++;
            }
        }
        vim_uint32 L = label.size();

        for_blanks[n] = MatrixRm::Constant(T, L+1, -numeric_limits<Real>::infinity());
        alphas[n] = MatrixRm::Constant(T, L, -numeric_limits<Real>::infinity());
        if(logits_time_major)
        {
            for_blanks[n](0,0) = log_softmax_logits(n, blank_index); //T N C
            alphas[n](0,0) = log_softmax_logits(n, label[0]);
        }
        else
        {
            for_blanks[n](0,0) = log_softmax_logits(n*T, blank_index);//N T C
            alphas[n](0,0) = log_softmax_logits(n*T, label[0]);
        }

        for(vim_uint32 t=1;t<T;t++)
        {
            for_blanks[n].row(t) = for_blanks[n].row(t-1);  
            for_blanks[n].block(t,1,1,L) = log_sum_exp2m(alphas[n].row(t-1), for_blanks[n].block(t,1,1,L));
            if(logits_time_major)
                for_blanks[n].row(t).array() += log_softmax_logits(t*N+n, blank_index);
            else
                for_blanks[n].row(t).array() += log_softmax_logits(n*T+t, blank_index);
            
            alphas[n].row(t) = log_sum_exp2m(for_blanks[n].block(t-1, 0, 1, L), alphas[n].row(t-1));
            
            for(vim_uint32 l=1;l<L;l++)
                if(label[l] != label[l-1])
                    alphas[n](t, l) = log_sum_exp2(alphas[n](t-1, l-1) , alphas[n](t, l));

            for(vim_uint32 l=0;l<L;l++)
                if(logits_time_major)
                    alphas[n](t, l) += log_softmax_logits(t*N+n, label[l]);
                else
                    alphas[n](t, l) += log_softmax_logits(n*T+t, label[l]);
        }
        (*output->get_matrix())(n,0) = -log_sum_exp2(for_blanks[n](T-1, L) , alphas[n](T-1, L-1));
    }  
    cout << *output->get_matrix()  << endl;
    return 0;       
}

/**************************************************************
 * 
 **************************************************************/
vim_int32 Ctc_loss::backword_ctc()
{
    vector<Real> loss(N);   // this line is for debug
    back_blanks.resize(N);
    betas.resize(N);

    for(vim_uint32 n=0;n<N;n++)
    {
        vector <vim_uint32> label = (*labels)[n];
        vector<vim_uint32>::iterator it=label.begin();
        while(it != label.end()){
            if(*it == blank_index){
                it = label.erase(it);
            }else{
                it++;
            }
        }
        vim_uint32 L = label.size();

        back_blanks[n] = MatrixRm::Constant(T, L+1, -numeric_limits<Real>::infinity());
        betas[n] = MatrixRm::Constant(T, L, -numeric_limits<Real>::infinity());
        if(logits_time_major)
        {
            back_blanks[n](T-1,L) = log_softmax_logits((T-1)*N+n, blank_index); //T N C
            betas[n](T-1,L-1) = log_softmax_logits((T-1)*N+n, label[L-1]);
        }
        else
        {
            back_blanks[n](T-1,L) = log_softmax_logits(n*T+T-1, blank_index);//N T C
            betas[n](T-1,L-1) = log_softmax_logits(n*T+T-1, label[L-1]);
        }


        for(vim_int32 t=T-2;t>=0;t--)
        {
            back_blanks[n].row(t) = back_blanks[n].row(t+1); 
            back_blanks[n].block(t,0,1,L) = log_sum_exp2m(betas[n].row(t+1), back_blanks[n].block(t,0,1,L));
            if(logits_time_major)
                back_blanks[n].row(t).array() += log_softmax_logits(t*N+n, blank_index);
            else
                back_blanks[n].row(t).array() += log_softmax_logits(n*T+t, blank_index);
            
            betas[n].row(t) = log_sum_exp2m(back_blanks[n].block(t+1, 1, 1, L), betas[n].row(t+1));
            
            for(vim_int32 l=L-2;l>=0;l--)
                if(label[l] != label[l+1])
                    betas[n](t, l) = log_sum_exp2(betas[n](t+1, l+1) , betas[n](t, l));

            for(vim_uint32 l=0;l<L;l++)
                if(logits_time_major)
                    betas[n](t, l) += log_softmax_logits(t*N+n, label[l]);
                else
                    betas[n](t, l) += log_softmax_logits(n*T+t, label[l]);
        }
        loss[n] = -log_sum_exp2( back_blanks[n](0,0) , betas[n](0,0));
    }  
    return 0;      
}


vim_int32 Ctc_loss::backward(vim_uint64 k)
{    
    backword_ctc();  //?????
    MatrixRm log_alpha_beta_matrix = MatrixRm::Constant(log_softmax_logits.rows(), log_softmax_logits.cols(), -numeric_limits<Real>::infinity());
    for(vim_uint32 n=0;n<N;n++)
    {
        vector <vim_uint32> label = (*labels)[n];
        vector<vim_uint32>::iterator it=label.begin();
        while(it != label.end()){
            if(*it == blank_index){
                it = label.erase(it);
            }else{
                it++;
            }
        }
        vim_uint32 L = label.size();

        if(logits_time_major)
        {
            for(vim_uint32 t=0;t<T;t++)
            {
                log_alpha_beta_matrix(t*N+n, blank_index) = log_sum_exp1v(for_blanks[n].row(t) + back_blanks[n].row(t));
                for(vim_uint32 l=0;t<L;l++)
                    log_alpha_beta_matrix(t*N+n, label[l]) = log_sum_exp2(alphas[n](t, l) + betas[n](t, l) , log_alpha_beta_matrix(t*N+n, label[l]));
                log_alpha_beta_matrix.row(t*N+n).array() += (*output->get_matrix())(n,0) - log_softmax_logits.row(t*N+n).array();
            }
        }
        else
        {
            for(vim_uint32 t=0;t<T;t++)
            {
                log_alpha_beta_matrix(n*T+t, blank_index) = log_sum_exp1v(for_blanks[n].row(t) + back_blanks[n].row(t));
                for(vim_uint32 l=0;l<L;l++)
                    log_alpha_beta_matrix(n*T+t, label[l]) = log_sum_exp2(alphas[n](t, l) + betas[n](t, l) , log_alpha_beta_matrix(n*T+t, label[l]));
                log_alpha_beta_matrix.row(n*T+t).array() += (*output->get_matrix())(n,0) - log_softmax_logits.row(n*T+t).array();
            }
        }
    } 
    log_alpha_beta_matrix =  log_softmax_logits.array().exp() - log_alpha_beta_matrix.array().exp();
    inputs[0]->set_back_data(this, &log_alpha_beta_matrix);
    return 0;
}


Real Ctc_loss::log_sum_exp2(Real x1, Real x2)
{
    if(x1 == -numeric_limits<Real>::infinity())
        return x2;
    else if(x2 == -numeric_limits<Real>::infinity())
        return x1;
    else if(x2>x1)
        return x2 + log(1.0 + exp(x1-x2));
    else
        return x1 + log(1.0 + exp(x2-x1));
}

MatrixRm Ctc_loss::log_sum_exp2m(MatrixRm m1, MatrixRm m2)
{
    assert(m1.rows() == m2.rows());
    assert(m1.cols() == m2.cols());

    return (m1.array()==-numeric_limits<Real>::infinity()).select(m2.array(),
            (m2.array()==-numeric_limits<Real>::infinity()).select(m1.array(), 
            (m1.array()<m2.array()).select(m2.array()+(m1-m2).array().exp().log1p(),
            m1.array()+(m2-m1).array().exp().log1p())));
}

Real Ctc_loss::log_sum_exp1v(MatrixRm m)
{
    Map <RowVectorRm> v(m.data(), m.size());
    Real res = v(v.size() - 1 );
    for(vim_int32 i = v.size()-2; i>=0;i--)
    {
        res = log_sum_exp2(res,v(i));
    }
    return res;
}

vector<VTensor * > Ctc_loss::build(vector<vector <vim_uint32>> *labels_, 
                        VTensor * logits_ ,
                        vector <vim_uint32> *label_length_, 
                        vector <vim_uint32> *logit_length_)
{
    register_input_tensor(logits_);

    labels = labels_; 
    label_length = label_length_;
    logit_length = logit_length_;
    output->owner = this;

    return {output};
}
