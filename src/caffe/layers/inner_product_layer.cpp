/*继承了Layer基类，定义了了InnerProductLayer类
K: 输入的向量维度
N: 输出的向量维度
M: 向量的个数
this->blob_[0]  weight项      N*K
this->blob_[1]  bias项        1*N
bottom_data                   M*K
top_data                      M*N 
*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  K_ = bottom[0]->count() / bottom[0]->num();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {               //判断是否有bias项。如果有，则blobs_长度置为2，第一项为weight，第二项为bias
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(    //参考Filler类定义，生成proto中规定的Filler对象，
        this->layer_param_.inner_product_param().weight_filler()));     //并取得Filler对象指针
    weight_filler->Fill(this->blobs_[0].get());                   //利用多态性对Filler用Fill方法初始化weight值
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());                   //利用多态性对Filler用Fill方法初始化bias值
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Figure out the dimensions
  M_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with inner product parameters.";
  (*top)[0]->Reshape(bottom[0]->num(), N_, 1, 1);                 //对top_data的大小做reshape，成为M*N
  // Set up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, M_);                        //bias_multiplier意义不明......
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data()); //将bias_multiplier_填充为M_个1
  }
}

//参考math_function.cpp中caffe_cpu_gemm()方法
/*
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C)
    
整理它的功能其实很直观，即C←αA×B+βC
const CBLAS_TRANSPOSE TransA  # A是否转置
const CBLAS_TRANSPOSE TransB  # B是否转置

# 其中A维度是MxK，B维度是KxN，C维度为MxN
*/
template <typename Dtype>           
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  //先计算y=bottom*weight'
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,   
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    //再计算y=y+bias
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),                  //bias_multiplier   M*1
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);   //this->blobs_[1]   1*N
  }
}

/*这段代码有疑惑*/
/*具体backward公式可以参考http://zhangliliang.com/2014/09/15/about-caffe-code/
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) 

# 实现的功能类似 Y←αAX + βY
# A MxN
# X Nx1
# Y Mx1
*/
template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)0.,             //在这里，bias_multiplier被定义成为N*1，由于N<M所以没问题？？
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe
