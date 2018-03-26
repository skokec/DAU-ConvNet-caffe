#include <boost/smart_ptr/shared_ptr.hpp>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/filler.hpp>
#include <caffe/layers/dau_conv_layer.hpp>
#include <caffe/layers/cudnn_conv_layer.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/test/test_caffe_main.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>

namespace caffe {

template <typename Dtype>
void compare_blobs(Blob<Dtype>& a, Blob<Dtype>& b, bool compare_diff, Dtype eps) {
	Dtype* data_a = compare_diff ? a.mutable_cpu_diff() : a.mutable_cpu_data();
	Dtype* data_b = compare_diff ? b.mutable_cpu_diff() : b.mutable_cpu_data();
	for (int i = 0; i < a.count(); ++i) {
		EXPECT_NEAR(data_a[i], data_b[i], eps);
	}
}

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int kernel_d, pad_d, stride_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
  } else {
    kernel_d = stride_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r;
                      int in_y = y * stride_h - pad_h + p;
                      int in_x = x * stride_w - pad_w + q;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class DAUConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DAUConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
		blob_bottom_3_(new Blob<Dtype>(1, 1, 4, 4)),
		//blob_bottom_3_(new Blob<Dtype>(2, 3, 32, 48)),
        blob_top_(new Blob<Dtype>()),
		blob_top_2_(new Blob<Dtype>()),
		blob_top_3_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    filler.Fill(this->blob_bottom_3_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DAUConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_top_;
    delete blob_top_2_;
    delete blob_top_3_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  Blob<Dtype>* const blob_top_3_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DAUConvolutionLayerTest, TestDtypesAndDevices);



#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

TYPED_TEST(DAUConvolutionLayerTest, TestFastGaussForward) {

    Caffe::SetDevice(0);

    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;

    // evaluate size settings
    const int N = 128;
    const int F = 64;
    const int S = 32;
    const int G = 2;
    const int W = 64;
    const int H = 32;

    const bool use_interpolation = true;

    const int kernel_size = 17;

    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_output(N,F,H,W);
    Blob<float> blob_output_cpu(N,F,H,W);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_std(1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);

    const_zero_float_filer.Fill(&blob_output);

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(1);
    offset_filler_param.set_max(kernel_size-2);
    UniformFiller<float> offset_filler(offset_filler_param);

    LayerParameter layer_param;

    DAUConvolutionParameter* dau_convolution_param = layer_param.mutable_dau_conv_param();

    dau_convolution_param->add_kernel_size(kernel_size);
    dau_convolution_param->add_stride(1);
    dau_convolution_param->add_pad(kernel_size/2);

    dau_convolution_param->add_number_units(G);
    dau_convolution_param->add_number_units(1);

    dau_convolution_param->set_num_output(F);

    dau_convolution_param->mutable_weight_filler()->set_type("gaussian");
    dau_convolution_param->mutable_weight_filler()->set_std(0.1);

    dau_convolution_param->mutable_bias_filler()->set_type("constant");
    dau_convolution_param->mutable_bias_filler()->set_value(0);

    dau_convolution_param->mutable_mu_filler()->set_type("constant");
    dau_convolution_param->mutable_mu_filler()->set_value(0);

    dau_convolution_param->mutable_sigma_filler()->set_type("constant");
    dau_convolution_param->mutable_sigma_filler()->set_value(0.8);

    dau_convolution_param->set_component_border_bound(0);
    dau_convolution_param->set_sigma_lower_bound(0.5);

    dau_convolution_param->set_unit_normalization(true);
    dau_convolution_param->set_square_unit_normalization(false);

    DAUConvolutionLayer<float> layer(layer_param);

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_cpu;

    blob_bottom_vec.push_back(&blob_input);

    blob_top_vec.push_back(&blob_output);
    blob_top_vec_cpu.push_back(&blob_output_cpu);

    layer.SetUp(blob_bottom_vec, blob_top_vec);

    // override offset data with random values - must be done after SetUp
    offset_filler.Fill(layer.dau_compute.param_buffer_mu1_.get());
    offset_filler.Fill(layer.dau_compute.param_buffer_mu2_.get());

    // perform forward pass with CPU version
    layer.set_processing_on_gpu(false);
    layer.Forward_cpu(blob_bottom_vec, blob_top_vec_cpu);

    // perform forward pass with GPU version
    layer.set_processing_on_gpu(true);
    layer.Forward_gpu(blob_bottom_vec, blob_top_vec);

    const float* output_cpu = blob_top_vec_cpu[0]->cpu_data();
    const float* output_gpu = blob_top_vec[0]->cpu_data();

    // verify data with CPU version
    int found_invalid = 0;

    double diff = 0;
    double max_diff = 0;

    for (int n = 0; n < N; ++n){
        for (int f = 0; f < F; ++f) {
            for (int i = 0; i < H * W; ++i) {
                int index = (n * F + f )* H * W + i;
                float val = output_gpu[index];
                float GT_VALUE = output_cpu[index];

                if (val != val) {
                    printf("error: got NaN at loc (%d=%d,%d,%d,%d) - should be %f\n",  index, n, f, i / W, i % W, GT_VALUE);
                }

                // interpolation at the right edge excludes one pixel so ignore those pixels
                if (std::abs(val - GT_VALUE) / std::abs(GT_VALUE) > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && i % W < (W -1)) {
                    if (found_invalid < 10)
                        printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index,  n, f, i / W, i % W, GT_VALUE);
                    found_invalid++;

                    double current_diff = std::abs(val - GT_VALUE) / std::abs(GT_VALUE);

                    diff += current_diff;

                    max_diff = std::max(max_diff, current_diff);
                }
            }
        }
    }
    if (found_invalid > 0) {
        diff /= found_invalid;
        printf("found num of invalid output vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid, blob_output.count(), diff, max_diff);
    }

    // report fail only if enough samples with big enough diff
    EXPECT_NEAR(diff, 0, 1e-3);
    EXPECT_NEAR(found_invalid/(float)blob_output.count(), 0, 1e-2);
}

TYPED_TEST(DAUConvolutionLayerTest, TestFastGaussBackward) {


    typedef typename TypeParam::Dtype Dtype;

    Caffe::SetDevice(0);

    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;
    // evaluate size settings

    const int N = 32;
    const int F = 128;
    const int S = 96;
    const int G = 2;
    const int W = 32;
    const int H = 64;

    // number of Guassian learning parameters we have (w,mu1,mu2,sigma)
    // for each parameter we need convolution of input data with specific kernel
    const int K = 4;
    const bool use_interpolation = true;
    const bool ignore_edge_gradients = true; // for cpu/gpu compatability

    const int kernel_size = 11;

    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_error(N,F,H,W);
    Blob<float> blob_output(K, S, G, F);
    Blob<float> blob_output_cpu(K, S, G, F);

    Blob<float> blob_output_error_cpu(N,S,H,W);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_value(1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);
    input_filler.Fill(&blob_error);
    caffe_rng_gaussian<float>(blob_error.count(), float(0), float(0.1), blob_error.mutable_cpu_diff());

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(4);
    offset_filler_param.set_max(kernel_size - 4);

    UniformFiller<float> offset_filler(offset_filler_param);

    const_zero_float_filer.Fill(&blob_output);

    LayerParameter layer_param;
    DAUConvolutionParameter* dau_convolution_param = layer_param.mutable_dau_conv_param();

    dau_convolution_param->add_kernel_size(kernel_size);
    dau_convolution_param->add_stride(1);
    dau_convolution_param->add_pad(kernel_size/2);

    dau_convolution_param->add_number_units(G);
    dau_convolution_param->add_number_units(1);

    dau_convolution_param->set_num_output(F);

    dau_convolution_param->mutable_weight_filler()->set_type("gaussian");
    dau_convolution_param->mutable_weight_filler()->set_std(0.1);

    dau_convolution_param->mutable_bias_filler()->set_type("constant");
    dau_convolution_param->mutable_bias_filler()->set_value(0);

    dau_convolution_param->mutable_mu_filler()->set_type("constant");
    dau_convolution_param->mutable_mu_filler()->set_value(0);

    dau_convolution_param->mutable_sigma_filler()->set_type("constant");
    dau_convolution_param->mutable_sigma_filler()->set_value(0.8);

    dau_convolution_param->set_component_border_bound(0);
    dau_convolution_param->set_sigma_lower_bound(0.5);

    dau_convolution_param->set_unit_normalization(true);
    dau_convolution_param->set_square_unit_normalization(false);

    DAUConvolutionLayer<float> layer(layer_param, ignore_edge_gradients);

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_gt;
    std::vector<bool > propagate_down;
    propagate_down.push_back(true);

    blob_bottom_vec.push_back(&blob_input);
    blob_top_vec.push_back(&blob_error);

    layer.SetUp(blob_bottom_vec, blob_top_vec);

    // override offset data with random values within valid range
    offset_filler.Fill(layer.dau_compute.param_buffer_mu1_.get());
    offset_filler.Fill(layer.dau_compute.param_buffer_mu2_.get());

    caffe_gpu_set(layer.dau_compute.param_buffer_w_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_w_->mutable_gpu_diff());
    caffe_gpu_set(layer.dau_compute.param_buffer_mu1_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_mu1_->mutable_gpu_diff());
    caffe_gpu_set(layer.dau_compute.param_buffer_mu2_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_mu2_->mutable_gpu_diff());
    caffe_gpu_set(layer.dau_compute.param_buffer_sigma_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_sigma_->mutable_gpu_diff());

    layer.set_processing_on_gpu(false);
    layer.Backward_cpu(blob_top_vec, propagate_down, blob_bottom_vec);

    // save backproped error values
    float *backprop_error_cpu = blob_output_error_cpu.mutable_cpu_data();
    {
        caffe_copy(blob_output_error_cpu.count(), blob_bottom_vec[0]->cpu_diff(), backprop_error_cpu);
    }

    // save accumulated gradient values
    float *gradients_cpu = blob_output_cpu.mutable_cpu_data();
    {
        int num_params = layer.dau_compute.param_buffer_w_->count();
        if (K > 0) caffe_copy(num_params, layer.dau_compute.param_buffer_w_->cpu_diff(), gradients_cpu + 0 * num_params );
        if (K > 1) caffe_copy(num_params, layer.dau_compute.param_buffer_mu1_->cpu_diff(), gradients_cpu + 1 * num_params );
        if (K > 2) caffe_copy(num_params, layer.dau_compute.param_buffer_mu2_->cpu_diff(), gradients_cpu + 2 * num_params );
        if (K > 3) caffe_copy(num_params, layer.dau_compute.param_buffer_sigma_->cpu_diff(), gradients_cpu + 3 * num_params);
    }

    // reset values for GPU run
    caffe_gpu_set(layer.dau_compute.param_buffer_w_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_w_->mutable_gpu_diff());
    caffe_gpu_set(layer.dau_compute.param_buffer_mu1_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_mu1_->mutable_gpu_diff());
    caffe_gpu_set(layer.dau_compute.param_buffer_mu2_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_mu2_->mutable_gpu_diff());
    caffe_gpu_set(layer.dau_compute.param_buffer_sigma_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_sigma_->mutable_gpu_diff());

    layer.set_processing_on_gpu(true);
    layer.Backward_gpu(blob_top_vec, propagate_down, blob_bottom_vec);

    // get ptr to backproped error on gpu
    const float *backprop_error_gpu = blob_bottom_vec[0]->cpu_diff();

    // save accumulated gradient values
    float *gradients_gpu_g = blob_output.mutable_gpu_data();

    int num_params = layer.dau_compute.param_buffer_w_->count();
    if (K > 0) caffe_gpu_memcpy(num_params * sizeof(float), layer.dau_compute.param_buffer_w_->gpu_diff(), gradients_gpu_g + 0 * num_params );
    if (K > 1) caffe_gpu_memcpy(num_params * sizeof(float), layer.dau_compute.param_buffer_mu1_->gpu_diff(), gradients_gpu_g + 1 * num_params );
    if (K > 2) caffe_gpu_memcpy(num_params * sizeof(float), layer.dau_compute.param_buffer_mu2_->gpu_diff(), gradients_gpu_g + 2 * num_params );
    if (K > 3) caffe_gpu_memcpy(num_params * sizeof(float), layer.dau_compute.param_buffer_sigma_->gpu_diff(), gradients_gpu_g + 3 * num_params);

    // get gradients from GPU, but read them on cpu
    float *gradients_gpu = blob_output.mutable_cpu_data();

    {
        // verify accumulated gradients
        int found_invalid_backprop = 0;

        double diff_gradient = 0;
        double max_diff = 0;
        for (int k = 0; k < K - 1; ++k) { // do not test last K since it is optional by default (i.e. it will not be computed unless there is no computation penalty)
            for (int s = 0; s < S; ++s) {
                for (int g = 0; g < G; ++g) {
                    for (int f = 0; f < F; ++f) {
                        int idx = OFFSET(k,s,g,f, K, S,  G, F);
                        float val = gradients_gpu[idx];
                        float GT_VALUE = gradients_cpu[idx];

                        if (val != val) {
                            printf("error: got NaN at loc (%d=%d,%d,%d,%d) - should be %f\n", idx, k, s,g,f, GT_VALUE);
                        }

                        if (std::abs(val - GT_VALUE) / std::abs(GT_VALUE) > 1e-4 && std::abs(val - GT_VALUE) > 1e-7) {
                            if (found_invalid_backprop < 10)
                                printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, idx, k, s,g,f, GT_VALUE);
                            found_invalid_backprop++;
                            double current_diff = std::abs(val - GT_VALUE) / std::abs(GT_VALUE);

                            diff_gradient += current_diff;

                            max_diff = std::max(max_diff, current_diff);

                        }
                    }
                }
            }
        }

        if (found_invalid_backprop > 0) {
            diff_gradient /= found_invalid_backprop;
            printf("found num of invalid accumulated gradient vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid_backprop, K * S * G * F, diff_gradient, max_diff);
        }
        // report fail only if enough samples with big enough diff
        EXPECT_NEAR(diff_gradient, 0, 1e-2);
        EXPECT_NEAR(found_invalid_backprop/(float)(K * S * G * F), 0, 1e-2);
    }
    {
        // verify accumulated gradients
        int found_invalid_backprop = 0;

        double diff_backprop = 0;
        double max_diff = 0;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < S; ++s) {
                for (int i = 0; i < H * W; ++i) {
                    int index = (n * S + s )* H * W + i;
                    float val = backprop_error_gpu[index];
                    float GT_VALUE = backprop_error_cpu[index];

                    if (val != val) {
                        printf("error: got NaN at loc (%d=%d,%d,%d,%d) - should be %f\n",  index, n, s, i / W, i % W, GT_VALUE);
                    }

                    if (std::abs(val - GT_VALUE) / GT_VALUE > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && i % W < (W -1)) {
                        if (found_invalid_backprop < 10)
                            printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index, n, s, i / W, i % W, GT_VALUE);
                        found_invalid_backprop++;
                        diff_backprop += std::abs(val - GT_VALUE) / GT_VALUE;

                        double current_diff = std::abs(val - GT_VALUE) / GT_VALUE;

                        diff_backprop += current_diff;

                        max_diff = std::max(max_diff, current_diff);


                    }
                }
            }
        }

        if (found_invalid_backprop > 0) {
            diff_backprop /= found_invalid_backprop;
            printf("found num of invalid backproped-error vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid_backprop, N * S * H * W, diff_backprop, max_diff);
        }
        // report fail only if enough samples with big enough diff
        EXPECT_NEAR(diff_backprop, 0, 1e-3);
        EXPECT_NEAR(found_invalid_backprop/(float)(N * S * H * W), 0, 1e-2);
    }
}


TYPED_TEST(DAUConvolutionLayerTest, ProfileFastDAUConvolution) {

    Caffe::SetDevice(0);

    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;

    // evaluate size settings
    const int N = 128;
    const int F = 128;
    const int S = 128;
    const int G = 2;
    const int W = 64;
    const int H = 64;

    const bool use_interpolation = true;

    const int kernel_size = 17;

    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_weights(1, S, G, F);
    Blob<float> blob_output(N,F,H,W);
    Blob<float> blob_output_cpu(N,F,H,W);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_std(0.1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(2);
    offset_filler_param.set_max(kernel_size-2);
    UniformFiller<float> offset_filler(offset_filler_param);

    const_zero_float_filer.Fill(&blob_output);

    LayerParameter layer_param;

    DAUConvolutionParameter* dau_convolution_param = layer_param.mutable_dau_conv_param();

    dau_convolution_param->add_kernel_size(kernel_size);
    dau_convolution_param->add_stride(1);
    dau_convolution_param->add_pad(kernel_size/2);

    dau_convolution_param->add_number_units(G);
    dau_convolution_param->add_number_units(1);

    dau_convolution_param->set_num_output(F);


    dau_convolution_param->mutable_weight_filler()->set_type("gaussian");
    dau_convolution_param->mutable_weight_filler()->set_std(0.1);

    dau_convolution_param->mutable_bias_filler()->set_type("constant");
    dau_convolution_param->mutable_bias_filler()->set_value(0);

    dau_convolution_param->mutable_mu_filler()->set_type("constant");
    dau_convolution_param->mutable_mu_filler()->set_value(0);

    dau_convolution_param->mutable_sigma_filler()->set_type("constant");
    dau_convolution_param->mutable_sigma_filler()->set_value(0.8);

    dau_convolution_param->set_component_border_bound(0);
    dau_convolution_param->set_sigma_lower_bound(0.5);

    dau_convolution_param->set_unit_normalization(true);
    dau_convolution_param->set_square_unit_normalization(false);

    DAUConvolutionLayer<float> layer(layer_param);

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;

    blob_bottom_vec.push_back(&blob_input);
    blob_top_vec.push_back(&blob_output);

    for (int ii = 0; ii < 1; ++ii) {

        layer.SetUp(blob_bottom_vec, blob_top_vec);

        offset_filler.Fill(layer.dau_compute.param_buffer_mu1_.get());
        offset_filler.Fill(layer.dau_compute.param_buffer_mu2_.get());


        for (int i = 0; i < 30; ++i) {
            cudaDeviceSynchronize();

            clock_t start_t = clock();
            layer.Forward_gpu(blob_bottom_vec, blob_top_vec);

            cudaDeviceSynchronize();
            clock_t end_t = clock();

            std::cout << "fast_gauss_forward in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
        }
    }
}

TYPED_TEST(DAUConvolutionLayerTest, ProfileFastGaussBackward) {


    typedef typename TypeParam::Dtype Dtype;

    Caffe::SetDevice(0);

    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;


    const int N = 128;
    const int F = 128;
    const int S = 128;
    const int G = 2;
    const int W = 64;
    const int H = 64;


    // number of Guassian learning parameters we have (w,mu1,mu2,sigma)
    // for each parameter we need convolution of input data with specific kernel
    const int K = 4;
    const bool use_interpolation = true;
    const bool ignore_edge_gradients = true; // for cpu/gpu compatability

    const int kernel_size = 11;

    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_error(N,F,H,W);
    Blob<float> blob_weights(1, S, G, F);
    Blob<float> blob_output(K, S, G, F);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_value(1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);
    input_filler.Fill(&blob_weights);
    input_filler.Fill(&blob_error);
    caffe_rng_gaussian<float>(blob_error.count(), float(0), float(0.1), blob_error.mutable_cpu_diff());

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(3);
    offset_filler_param.set_max(kernel_size - 3);

    UniformFiller<float> offset_filler(offset_filler_param);

    const_zero_float_filer.Fill(&blob_output);

    LayerParameter layer_param;
    DAUConvolutionParameter* dau_convolution_param = layer_param.mutable_dau_conv_param();

    dau_convolution_param->add_kernel_size(kernel_size);
    dau_convolution_param->add_stride(1);
    dau_convolution_param->add_pad(kernel_size/2);

    dau_convolution_param->add_number_units(G);
    dau_convolution_param->add_number_units(1);

    dau_convolution_param->set_num_output(F);

    dau_convolution_param->mutable_weight_filler()->set_type("gaussian");
    dau_convolution_param->mutable_weight_filler()->set_std(0.1);

    dau_convolution_param->mutable_bias_filler()->set_type("constant");
    dau_convolution_param->mutable_bias_filler()->set_value(0);

    dau_convolution_param->mutable_mu_filler()->set_type("constant");
    dau_convolution_param->mutable_mu_filler()->set_value(0);

    dau_convolution_param->mutable_sigma_filler()->set_type("constant");
    dau_convolution_param->mutable_sigma_filler()->set_value(0.8);

    dau_convolution_param->set_component_border_bound(0);
    dau_convolution_param->set_sigma_lower_bound(0.5);

    dau_convolution_param->set_unit_normalization(true);
    dau_convolution_param->set_square_unit_normalization(false);

    DAUConvolutionLayer<float> layer(layer_param, ignore_edge_gradients);

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;

    std::vector<bool > propagate_down;
    propagate_down.push_back(true);

    blob_bottom_vec.push_back(&blob_input);
    blob_top_vec.push_back(&blob_error);


    for (int ii = 0; ii < 1; ++ii) {

        layer.SetUp(blob_bottom_vec, blob_top_vec);

        offset_filler.Fill(layer.dau_compute.param_buffer_mu1_.get());
        offset_filler.Fill(layer.dau_compute.param_buffer_mu2_.get());

        // reset values for GPU run
        caffe_gpu_set(layer.dau_compute.param_buffer_w_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_w_->mutable_gpu_diff());
        caffe_gpu_set(layer.dau_compute.param_buffer_mu1_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_mu1_->mutable_gpu_diff());
        caffe_gpu_set(layer.dau_compute.param_buffer_mu2_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_mu2_->mutable_gpu_diff());
        caffe_gpu_set(layer.dau_compute.param_buffer_sigma_->count(), (Dtype)0, (Dtype*)layer.dau_compute.param_buffer_sigma_->mutable_gpu_diff());

        cudaDeviceSynchronize();

        for (int i = 0; i < 30; ++i) {
            clock_t start_t = clock();

            layer.Backward_gpu(blob_top_vec, propagate_down, blob_bottom_vec);

            cudaDeviceSynchronize();
            clock_t end_t = clock();

            std::cout << "fast_gauss_backward in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
        }
    }
}

TYPED_TEST(DAUConvolutionLayerTest, DebugFastDAUConvolution1x1) {

    Caffe::SetDevice(1);

    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;

    // evaluate size settings
    const int N = 22;
    const int F = 64;
    const int S = 16;
    const int G = 4;
    const int W_in = 7;
    const int H_in = 7;

    const int kernel_size = 7;

    //const int pad = kernel_size/2;
    const int pad = 0;
    //const int pad = kernel_size/2  * 2;

    const int W = (W_in + 2 * pad - kernel_size) + 1;
    const int H = (H_in + 2 * pad - kernel_size) + 1;

    const bool use_interpolation = true;

    Blob<float> blob_input(N,S,H_in,W_in);
    Blob<int> blob_offsets_x(1, S, G, F);
    Blob<int> blob_offsets_y(1, S, G, F);
    Blob<float> blob_offsets_float_x(1, S, G, F);
    Blob<float> blob_offsets_float_y(1, S, G, F);
    Blob<float> blob_weights(1, S, G, F);
    Blob<float> blob_output(N,F,H,W);
    Blob<float> blob_output_cpu(N,F,H,W);
    Blob<float> blob_output_cudnn(N,F,H,W);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_std(0.1);
    GaussianFiller<float> input_filler(rand_filler_param);

    //input_filler.Fill(&blob_input);
    //input_filler.Fill(&blob_weights);

    const_one_filer.Fill(&blob_input);
    const_one_filer.Fill(&blob_weights);


    input_filler.Fill(&blob_input);

    float* data = blob_input.mutable_cpu_data();
    for (int n = 0; n < N; ++n){
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < H_in * W_in; ++i) {
                data[(n * S + s )* H_in * W_in + i] = 1;
                //data[(n * S + s )* H * W + i] = 2;
                //data[(n * S + s )* H * W + i] = n + (i % W + 1);
                //data[(n * S + s )* H * W + i] = n + (i / W + 1);
                //data[(n * S + s )* H * W + i] = (i % W);

            }
        }
    }

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(2);
    offset_filler_param.set_max(kernel_size-2);
    UniformFiller<float> offset_filler(offset_filler_param);

    //offset_filler.Fill(&blob_offsets);
    const_zero_filer.Fill(&blob_offsets_x);
    const_zero_filer.Fill(&blob_offsets_y);

    const_zero_float_filer.Fill(&blob_offsets_float_x);
    const_zero_float_filer.Fill(&blob_offsets_float_y);

    const_zero_float_filer.Fill(&blob_output);

    float* output = Caffe::mode() == Caffe::CPU ? blob_output.mutable_cpu_data() : blob_output.mutable_gpu_data();

    LayerParameter layer_param;

    DAUConvolutionParameter* dau_convolution_param = layer_param.mutable_dau_conv_param();

    dau_convolution_param->add_kernel_size(kernel_size);
    dau_convolution_param->add_stride(1);
    dau_convolution_param->add_pad(pad);

    dau_convolution_param->add_number_units(G);
    dau_convolution_param->add_number_units(1);

    dau_convolution_param->set_num_output(F);

    //dau_convolution_param->mutable_weight_filler()->set_type("constant");
    //dau_convolution_param->mutable_weight_filler()->set_value(2);

    dau_convolution_param->mutable_weight_filler()->set_type("gaussian");
    dau_convolution_param->mutable_weight_filler()->set_std(0.1);

    dau_convolution_param->mutable_bias_filler()->set_type("constant");
    dau_convolution_param->mutable_bias_filler()->set_value(0);

    dau_convolution_param->mutable_mu_filler()->set_type("constant");
    dau_convolution_param->mutable_mu_filler()->set_value(0);

    dau_convolution_param->mutable_sigma_filler()->set_type("constant");
    dau_convolution_param->mutable_sigma_filler()->set_value(0.8);

    dau_convolution_param->set_component_border_bound(0);
    dau_convolution_param->set_sigma_lower_bound(0.5);

    dau_convolution_param->set_unit_normalization(true);
    dau_convolution_param->set_square_unit_normalization(false);

    DAUConvolutionLayer<float> layer(layer_param);


    LayerParameter cudnn_layer_param;

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_cudnn;
    std::vector<Blob<float>* > blob_top_vec_cpu;


    blob_bottom_vec.push_back(&blob_input);

    blob_top_vec.push_back(&blob_output);
    blob_top_vec_cudnn.push_back(&blob_output_cudnn);
    blob_top_vec_cpu.push_back(&blob_output_cpu);

    for (int ii = 0; ii < 1; ++ii) {

        layer.SetUp(blob_bottom_vec, blob_top_vec);

        // override offset data with zeros
        float* filter_offsets_float_mu1 = layer.dau_compute.param_buffer_mu1_->mutable_gpu_data();
        float* filter_offsets_float_mu2 = layer.dau_compute.param_buffer_mu2_->mutable_gpu_data();

        cudaMemset(filter_offsets_float_mu1, 0, layer.dau_compute.param_buffer_mu1_->count() * sizeof(float));
        cudaMemset(filter_offsets_float_mu2, 0, layer.dau_compute.param_buffer_mu2_->count() * sizeof(float));

        offset_filler.Fill(layer.dau_compute.param_buffer_mu1_.get());
        offset_filler.Fill(layer.dau_compute.param_buffer_mu2_.get());

        float* mu1_data = layer.dau_compute.param_buffer_mu1_->mutable_cpu_data();
        float* mu2_data = layer.dau_compute.param_buffer_mu2_->mutable_cpu_data();
        float* w_data = layer.dau_compute.param_buffer_w_->mutable_cpu_data();


        for (int s = 0; s < S; s++) {
            for (int g = 0; g < G; g++) {
                for (int f = 0; f < F; f++) {
                    //w_data[OFFSET(0,s,g,f, 1, S,G,F)] = floor(w_data[OFFSET(0,s,g,f, 1, S,G,F)] * 100) / 100.0f ;
                    //w_data[OFFSET(0,s,g,f, 1, S,G,F)] = s;
                    w_data[OFFSET(0,s,g,f, 1, S,G,F)] = 1;
                    //mu1_data[OFFSET(0,s,g,f, 1, S,G,F)] = -2.2 + ((f+1)*(1+s)*(g+1)) % 5 ;
                    //mu2_data[OFFSET(0,s,g,f, 1, S,G,F)] = 3.1 - ((f+1)*(1+s)*(g+1)) % 5 ;
                    mu1_data[OFFSET(0,s,g,f, 1, S,G,F)] = kernel_size/2;
                    mu2_data[OFFSET(0,s,g,f, 1, S,G,F)] = kernel_size/2;
                }
            }
        }

        layer.set_processing_on_gpu(false);
        layer.Forward_cpu(blob_bottom_vec, blob_top_vec_cpu);

        layer.set_processing_on_gpu(true);
        layer.Forward_gpu(blob_bottom_vec, blob_top_vec);

        float* output_cpu = blob_top_vec_cpu[0]->mutable_cpu_data();
        float* output_gpu = blob_top_vec[0]->mutable_cpu_data();


        // verify data - since we use 1 for input and wights and 0 for offsets we should get S as output value for all

        const bool compare_by_cpu = true;
        int found_invalid = 0;
        //double valid_value = S *G;
        double valid_value = (S-1) * (S)/2 * G;
        double diff = 0;
        double max_diff = 0;

        for (int n = 0; n < N; ++n){
            for (int f = 0; f < F; ++f) {
                for (int i = 0; i < H * W; ++i) {
                    int index = (n * F + f )* H * W + i;
                    float val = output_gpu[index];
                    float GT_VALUE = (n + i % W +1 )*G*S;  // for data[] = n + (i % W + 1)
                    //float GT_VALUE = (n + i / W +1 )*G*S;  // for data[] = n + (i  / W + 1)
                    //float GT_VALUE = (i % W  ) *G*S;       // for data[] = s
                    //float GT_VALUE = n + (i % W + 1);      // for just copy

                    if (compare_by_cpu) {
                        GT_VALUE = output_cpu[index];
                    }
                    if (val != val) {
                        printf("error: got NaN at loc (%d=%d,%d,%d,%d) - should be %f\n",  index, n, f, i / W, i % W, GT_VALUE);
                    }
                    // interpolation at the right edge excludes one pixel so ignore those pixels

                    if (std::abs(val - GT_VALUE) / GT_VALUE > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && (i % W < (W -1) || (W == 1 && H == 1))) {
                        if (found_invalid < 10)
                            printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index,  n, f, i / W, i % W, GT_VALUE);
                        found_invalid++;

                        double current_diff = std::abs(val - GT_VALUE) / GT_VALUE;

                        diff += current_diff;

                        max_diff = std::max(max_diff, current_diff);
                    }
                    //if (i % W == 0)
                    //    std::cout << std::endl;
                    //std::cout << val << " ";
                    //std::cout << GT_VALUE << " ";
                }
                //std::cout << std::endl;
            }
        }
        //std::cout << std::endl;
        diff /= found_invalid;

        if (found_invalid > 0)
            printf("found num of invalid output vals: %d/%d with mean diff val %f and max diff val %f\n",found_invalid, blob_output.count(), diff, max_diff);
    }
}


TYPED_TEST(DAUConvolutionLayerTest, DebugFastGaussMemtest) {

    Caffe::SetDevice(1);

    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;

    SolverParameter solver_param;
    NetParameter net_param;

    solver_param.set_test_initialization(false);
    solver_param.set_max_iter(10000);
    solver_param.set_display(100000);
    solver_param.set_stepsize(100000);
    solver_param.set_base_lr(0.01);
    solver_param.set_snapshot(100000);
    solver_param.set_momentum(0.9);
    solver_param.set_solver_mode(solver_param.GPU);
    solver_param.set_allocated_net_param(&net_param);


    // evaluate size settings
    const int N = 16;
    const int F = 64;
    const int S = 64;
    const int G = 2;
    const int W = 32;
    const int H = 32;

    const bool use_interpolation = true;

    const int kernel_size = 9;

    Blob<float> blob_input(N,S,H,W);
    Blob<int> blob_offsets_x(1, S, G, F);
    Blob<int> blob_offsets_y(1, S, G, F);
    Blob<float> blob_offsets_float_x(1, S, G, F);
    Blob<float> blob_offsets_float_y(1, S, G, F);
    Blob<float> blob_weights(1, S, G, F);
    Blob<float> blob_output(N,F,H,W);
    Blob<float> blob_output_cpu(N,F,H,W);
    Blob<float> blob_output_cudnn(N,F,H,W);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_std(0.1);
    GaussianFiller<float> input_filler(rand_filler_param);

    //input_filler.Fill(&blob_input);
    //input_filler.Fill(&blob_weights);

    const_one_filer.Fill(&blob_input);
    const_one_filer.Fill(&blob_weights);


    input_filler.Fill(&blob_input);

    float* data = blob_input.mutable_cpu_data();
    for (int n = 0; n < N; ++n){
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < H * W; ++i) {
                //data[(n * S + s )* H * W + i] = 1;
                //data[(n * S + s )* H * W + i] = s;
                //data[(n * S + s )* H * W + i] = n + (i % W + 1);
                //data[(n * S + s )* H * W + i] = n + (i / W + 1);
                //data[(n * S + s )* H * W + i] = (i % W);

            }
        }
    }

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(2);
    offset_filler_param.set_max(kernel_size-2);
    UniformFiller<float> offset_filler(offset_filler_param);

    //offset_filler.Fill(&blob_offsets);
    const_zero_filer.Fill(&blob_offsets_x);
    const_zero_filer.Fill(&blob_offsets_y);

    const_zero_float_filer.Fill(&blob_offsets_float_x);
    const_zero_float_filer.Fill(&blob_offsets_float_y);

    const_zero_float_filer.Fill(&blob_output);

    float* output = Caffe::mode() == Caffe::CPU ? blob_output.mutable_cpu_data() : blob_output.mutable_gpu_data();

    LayerParameter layer_param;

    DAUConvolutionParameter* dau_convolution_param = layer_param.mutable_convolution_param();

    dau_convolution_param->add_kernel_size(kernel_size);
    dau_convolution_param->add_stride(1);
    dau_convolution_param->add_pad(kernel_size/2);

    dau_convolution_param->add_number_units(G);
    dau_convolution_param->add_number_units(1);

    dau_convolution_param->set_num_output(F);

    //dau_convolution_param->mutable_weight_filler()->set_type("constant");
    //dau_convolution_param->mutable_weight_filler()->set_value(2);

    dau_convolution_param->mutable_weight_filler()->set_type("gaussian");
    dau_convolution_param->mutable_weight_filler()->set_std(0.1);

    dau_convolution_param->mutable_bias_filler()->set_type("constant");
    dau_convolution_param->mutable_bias_filler()->set_value(0);

    dau_convolution_param->mutable_mu_filler()->set_type("constant");
    dau_convolution_param->mutable_mu_filler()->set_value(0);

    dau_convolution_param->mutable_sigma_filler()->set_type("constant");
    dau_convolution_param->mutable_sigma_filler()->set_value(0.8);

    dau_convolution_param->set_component_border_bound(0);
    dau_convolution_param->set_sigma_lower_bound(0.5);

    dau_convolution_param->set_unit_normalization(true);
    dau_convolution_param->set_square_unit_normalization(false);

    DAUConvolutionLayer<float> layer(layer_param);


    LayerParameter cudnn_layer_param;

    DAUConvolutionParameter* convolution_param =
            cudnn_layer_param.mutable_dau_conv_param();

    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(1);
    convolution_param->add_pad(1);

    convolution_param->set_num_output(F);

    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_weight_filler()->set_std(0.1);

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_cudnn;
    std::vector<Blob<float>* > blob_top_vec_cpu;


    blob_bottom_vec.push_back(&blob_input);

    blob_top_vec.push_back(&blob_output);
    blob_top_vec_cudnn.push_back(&blob_output_cudnn);
    blob_top_vec_cpu.push_back(&blob_output_cpu);


    std::cout << std::endl;
    for (int ii = 0; ii < 1; ++ii) {

        layer.SetUp(blob_bottom_vec, blob_top_vec);

        // override offset data with zeros
        float* filter_offsets_float_mu1 = layer.dau_compute.param_buffer_mu1_->mutable_gpu_data();
        float* filter_offsets_float_mu2 = layer.dau_compute.param_buffer_mu2_->mutable_gpu_data();

        cudaMemset(filter_offsets_float_mu1, 0, layer.dau_compute.param_buffer_mu1_->count() * sizeof(float));
        cudaMemset(filter_offsets_float_mu2, 0, layer.dau_compute.param_buffer_mu2_->count() * sizeof(float));

        offset_filler.Fill(layer.dau_compute.param_buffer_mu1_.get());
        offset_filler.Fill(layer.dau_compute.param_buffer_mu2_.get());

        for (int jj = 0; jj < 10000; ++jj) {
            layer.Forward_gpu(blob_bottom_vec, blob_top_vec);
        }

        float* output_cpu = blob_top_vec_cpu[0]->mutable_cpu_data();
        float* output_gpu = blob_top_vec[0]->mutable_cpu_data();

        // verify data - since we use 1 for input and wights and 0 for offsets we should get S as output value for all
    }
}

#ifdef USE_CUDNN

#endif

}  // namespace caffe
