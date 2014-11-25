#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();    //通过source读取输入文件的文件名
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());     
  string filename;
  int label;
  while (infile >> filename >> label) {                   //文件中filename和label成行排列
    lines_.push_back(std::make_pair(filename, label));    //hpp定义vector<std::pair<std::string, int> > lines_;
  }

  if (this->layer_param_.image_data_param().shuffle()) {  //如果shuffle参数为true，需要将文件随机打乱
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();        //利用Fisher–Yates洗牌算法将从lines_begin到lines_end的文件打乱
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0; 
  // Check if we would need to randomly skip a few data points    
  if (this->layer_param_.image_data_param().rand_skip()) {  //如果rand_skip参数为true，需要跳过开头一部分data。这是为了避免多个用户异步训练的情况。
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";   //避免跳过的数据点比整个数据规模还要大
    lines_id_ = skip;                                       //跳过skip个数据点，从skip个数据点之后开始
  }
  // Read a data point, and use it to initialize the top blob.    //只读入一张图片，用于确定top blob的大小
  Datum datum;                                              //protobuf中定义的Datum类
  CHECK(ReadImageToDatum(lines_[lines_id_].first, lines_[lines_id_].second,   //io.hpp 用opencv读入图片和label，存入datum中
                         new_height, new_width, &datum));
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();     
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {                                      //protobuf是否定义crop的大小，若有则需要裁剪
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);                  //protobuf定义的一次训练batch的大小
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

template <typename Dtype>     //利用Fisher–Yates洗牌算法将从lines_begin到lines_end的文件打乱
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);  //Fisher–Yates洗牌算法实现
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {      //读batch_size规模数量的图片
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    if (!ReadImageToDatum(lines_[lines_id_].first,
          lines_[lines_id_].second,
          new_height, new_width, &datum)) {
      continue;
    }

    // Apply transformations (mirror, crop...) to the data
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    top_label[item_id] = datum.label();
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
}

INSTANTIATE_CLASS(ImageDataLayer);

}  // namespace caffe
