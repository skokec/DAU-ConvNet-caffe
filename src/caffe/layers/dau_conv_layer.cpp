#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <caffe/layers/dau_conv_layer.hpp>

namespace caffe {

template <typename Dtype>
void DAUComponentInitializerCaffe<Dtype>::InitializeParameters(const DAUConvSettings& settings, Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
                                                               int num_units_per_x, int num_units_per_y, int num_units_ignore,
                                                               int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const {


    int units_per_channel =  num_units_per_x * num_units_per_y;

    Blob<Dtype> tmp_w(1, conv_in_channels,  units_per_channel, conv_out_channels);
    Blob<Dtype> tmp_sigma(1, conv_in_channels, units_per_channel, conv_out_channels);
    Blob<Dtype> tmp_mu1(1, conv_in_channels, units_per_channel, conv_out_channels);
    Blob<Dtype> tmp_mu2(1, conv_in_channels, units_per_channel, conv_out_channels);

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->weight_filler_));
    shared_ptr<Filler<Dtype> > sigma_filler(GetFiller<Dtype>(this->sigma_filler_));

    weight_filler->Fill(&tmp_w);
    sigma_filler->Fill(&tmp_sigma);

    const int outer_size = conv_in_channels;
    const int middle_size = units_per_channel;
    const int inner_size = conv_out_channels;

    Dtype* w_buf = tmp_w.mutable_cpu_data();

    for (int i1 = 0; i1 < outer_size; ++i1) {
        for (int i2 = 0; i2 < middle_size; ++i2) {
            for (int i3 = 0; i3 < inner_size; ++i3) {
                const int offset_idx = (i1 * middle_size + i2 )* inner_size + i3;

                // if we need to ignore last few values then set weights to zero
                if (i2 >= units_per_channel - num_units_ignore) {
                    w_buf[offset_idx] = 0;
                }
            }
        }
    }

    Dtype* mu1_buf = tmp_mu1.mutable_cpu_data();
    Dtype* mu2_buf = tmp_mu2.mutable_cpu_data();

    const FillerParameter& mu_filler_param = this->mu_filler_;

    // NOTE: we use "uniform-random" to have legacy compatability where old code assumes "uniform" to mean equally spaced
    // grid pattern; "uniform-random" now actually initializes them uniformly
    if (mu_filler_param.type() == "uniform-random") {
        FillerParameter uniform_mu1_filler_param,uniform_mu2_filler_param;

        uniform_mu1_filler_param.set_min(mu_filler_param.min() > 0 ? mu_filler_param.min() : settings.component_border_bound);
        uniform_mu1_filler_param.set_max(mu_filler_param.max() > 0 ? mu_filler_param.max() : kernel_w - settings.component_border_bound);

        uniform_mu2_filler_param.set_min(mu_filler_param.min() > 0 ? mu_filler_param.min() : settings.component_border_bound);
        uniform_mu2_filler_param.set_max(mu_filler_param.max() > 0 ? mu_filler_param.max() : kernel_h - settings.component_border_bound);

        UniformFiller<Dtype> mu1_filler(uniform_mu1_filler_param);
        UniformFiller<Dtype> mu2_filler(uniform_mu1_filler_param);

        mu1_filler.Fill(&tmp_mu1);
        mu2_filler.Fill(&tmp_mu2);

    } else {
        //int num_gauss_per_axis = units_per_channel /2;
        Dtype* offset_x = new Dtype[num_units_per_x];
        Dtype* offset_y = new Dtype[num_units_per_y];

        // use unit_border_bound as start and stop position of where components are allowed to be within the kernel
        Dtype gmm_mu_bounds_h_ = (Dtype)kernel_h - 2*settings.component_border_bound;
        Dtype gmm_mu_bounds_w_ = (Dtype)kernel_w - 2*settings.component_border_bound;

        Dtype gmm_mu_border_bound = settings.component_border_bound;

        // if filler for mu is "gmm_mu_bounds" then use min/max values as bounds instead of default (if it is not gmm_mu_bounds then just ignore filler)
        if (mu_filler_param.type() == "gmm_mu_bounds") {
            gmm_mu_bounds_h_ = mu_filler_param.max() -  mu_filler_param.min();
            gmm_mu_bounds_w_ = mu_filler_param.max() -  mu_filler_param.min();

            gmm_mu_border_bound = mu_filler_param.min();
        }

        for (int i = 0; i < num_units_per_x; i++) {
            offset_x[i] = gmm_mu_border_bound + (i)*gmm_mu_bounds_w_ /(Dtype)(num_units_per_x) + (- 0.5+(gmm_mu_bounds_w_)/(Dtype)(2*num_units_per_x));
        }
        for (int i = 0; i < num_units_per_y; i++) {
            offset_y[i] = gmm_mu_border_bound + (i)*gmm_mu_bounds_h_ /(Dtype)(num_units_per_y) + (- 0.5+(gmm_mu_bounds_h_)/(Dtype)(2*num_units_per_y));
        }

        // add offset to mean so that (0,0) is at center (we do not do this any more)
        int kernel_center_w = settings.offsets_already_centered ? kernel_w / 2 : 0;
        int kernel_center_h = settings.offsets_already_centered ? kernel_h / 2 : 0;
        //int kernel_center_w = 0;
        //int kernel_center_h = 0;


        for (int i1 = 0; i1 < outer_size; ++i1) {
            for (int i2 = 0; i2 < middle_size; ++i2) {
                for (int i3 = 0; i3 < inner_size; ++i3) {
                    const int gauss_idx = i2;
                    const int offset_idx = (i1 * middle_size + i2 )* inner_size + i3;
                    mu1_buf[offset_idx] = offset_x[gauss_idx / num_units_per_y] - kernel_center_w;
                    mu2_buf[offset_idx] = offset_y[gauss_idx %  num_units_per_y] - kernel_center_h;
                }
            }
        }

        delete [] offset_x;
        delete [] offset_y;
    }

    const size_t param_size = conv_in_channels * conv_out_channels * units_per_channel;

    if (is_gpu_ptr) {
        caffe_gpu_memcpy(sizeof(Dtype) * param_size, tmp_w.gpu_data(), w);
        caffe_gpu_memcpy(sizeof(Dtype) * param_size, tmp_mu1.gpu_data(), mu1);
        caffe_gpu_memcpy(sizeof(Dtype) * param_size, tmp_mu2.gpu_data(), mu2);
        caffe_gpu_memcpy(sizeof(Dtype) * param_size, tmp_sigma.gpu_data(), sigma);
    } else {
        memcpy(w, tmp_w.gpu_data(), sizeof(Dtype) * param_size);
        memcpy(mu1, tmp_mu1.gpu_data(), sizeof(Dtype) * param_size);
        memcpy(mu2, tmp_mu2.gpu_data(), sizeof(Dtype) * param_size);
        memcpy(sigma, tmp_sigma.gpu_data(), sizeof(Dtype) * param_size);
    }
}

template <typename Dtype>
DAUConvLayerCaffe<Dtype>::~DAUConvLayerCaffe(){
    this->deallocate_workspace_mem();
}

template <typename Dtype>
void* DAUConvLayerCaffe<Dtype>::allocate_workspace_mem(size_t bytes) {
    // deallocate existing workspace memory
    deallocate_workspace_mem();

    // then allocate new one
    cudaError_t err = cudaMalloc(&(this->own_workspace_data), bytes);
    if (err != cudaSuccess) {
        // NULL out underlying data
        this->own_workspace_data = NULL;
    }
    return this->own_workspace_data;
}

template <typename Dtype>
void DAUConvLayerCaffe<Dtype>::deallocate_workspace_mem() {
    if (this->own_workspace_data == NULL)
        CUDA_CHECK(cudaFree(this->own_workspace_data));

}

template <typename Dtype>
void DAUConvLayerCaffe<Dtype>::reshape_params(const vector<int>& shape) {
    // initialize DAU parameters (learnable)
    this->param_buffer_w_.reset(new Blob<Dtype>(shape));
    this->param_buffer_mu1_.reset(new Blob<Dtype>(shape));
    this->param_buffer_mu2_.reset(new Blob<Dtype>(shape));
    this->param_buffer_sigma_.reset(new Blob<Dtype>(shape));

    // If necessary, initialize the biases.
    if (this->bias_term_) {
        vector<int> bias_shape(1, this->conv_out_channels_);
        this->param_buffer_bias_.reset(new Blob<Dtype>(bias_shape));
    }
}

template <typename Dtype>
void DAUConvLayerCaffe<Dtype>::LayerSetUp(const DAUConvSettings& settings,
                                             const BaseDAUComponentInitializer<Dtype>& param_initializer,
                                             BaseDAUKernelCompute<Dtype>* kernel_compute,
                                             BaseDAUKernelParams<Dtype>* kernel_param,
                                             BaseDAUKernelOutput<Dtype>* kernel_output,
                                             const vector<int>& bottom_shape, bool in_train) {

    // call parent to compute all the shape variables and call initialize of parameter shape
    BaseDAUConvLayer<Dtype>::LayerSetUp(settings, param_initializer,
                                        kernel_compute, kernel_param, kernel_output,
                                        bottom_shape, in_train);


    // we use actual (learnable) sigma parameter when computing kernels so connect that param with the sigma for aggregation
    static_cast<DAUKernelParamsCaffe<Dtype>* >(kernel_param)->sigma_ = this->param_buffer_sigma_;

}

template <typename Dtype>
vector<int> DAUConvLayerCaffe<Dtype>::Reshape(const vector<int>& bottom_shape, const vector<int>& top_shape) {

    // call parent to compute all the shape variables
    const vector<int> new_top_shape = BaseDAUConvLayer<Dtype>::Reshape(bottom_shape, top_shape);

    const int max_width = std::max(this->width_out_,this->width_);
    const int max_height = std::max(this->height_out_,this->height_);

    // Set up the all ones "bias multiplier" for adding biases
    if (this->bias_term_) {
        vector<int> bias_multiplier_shape(1, this->height_out_ * this->width_out_);
        this->bias_multiplier_.Reshape(bias_multiplier_shape);
        caffe_set(this->bias_multiplier_.count(), Dtype(1), this->bias_multiplier_.mutable_cpu_data());
    }

    // make sure col_buffer is big enough
    this->col_buffer_.Reshape(this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_);

    // use inter buffer for both fwd and bwd passes so allocate buffer with suitable size for both
    this->interm_buffer_.Reshape(this->batch_num_, std::max(this->conv_in_channels_ * this->NUM_K, this->conv_out_channels_), max_height, max_width);

    this->bwd_gradients_.Reshape(this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_);

    // temporary buffer used during the back-propagation of the error where we rotate mu1 and mu2
    this->tmp_param_buffer_.Reshape(2, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_);

    return new_top_shape;

}

template <typename Dtype>
bool DAUConvLayerCaffe<Dtype>::update_prefiltering_kernels(cudaStream_t stream) {
    bool updated = BaseDAUConvLayer<Dtype>::update_prefiltering_kernels(stream);


    if (updated) {
        //for debug write kernel with 1 only at center i.e. identity convolution kernel
        if (0) {
            DAUKernelOutputCaffe<Dtype>* kernels_output = static_cast<DAUKernelOutputCaffe<Dtype>*>(this->aggregation.kernels);

            Dtype*  gauss_kernel = kernels_output->weight_.mutable_cpu_data();

            int deriv_count = this->conv_in_channels_ * this->units_per_channel * this->conv_out_channels_ *
                              this->aggregation.kernel_h_ * this->aggregation.kernel_w_;

            Dtype*  deriv_weight_kernel = kernels_output->d_params_.mutable_cpu_data() + 0 * deriv_count;
            Dtype*  deriv_mu1_kernel = kernels_output->d_params_.mutable_cpu_data() + 1 * deriv_count;
            Dtype*  deriv_mu2_kernel = kernels_output->d_params_.mutable_cpu_data() + 2 * deriv_count;
            Dtype*  deriv_sigma_kernel = kernels_output->d_params_.mutable_cpu_data() + 3 * deriv_count;
            Dtype*  deriv_error_kernel = kernels_output->d_error_.mutable_cpu_data();


            int h_half = this->aggregation.kernel_h_/2;
            int w_half = this->aggregation.kernel_w_/2;
            int index = 0;
            for (int j = -h_half; j <= h_half; ++j) {
                for (int i = -w_half; i <= w_half; ++i) {

                    Dtype val = (i == 0 && j == 0 ? 1 : 0);

                    gauss_kernel[index] = val;
                    deriv_weight_kernel[index] = val;
                    deriv_mu1_kernel[index] = val;
                    deriv_mu2_kernel[index] = val;
                    deriv_sigma_kernel[index] = val;
                    deriv_error_kernel[index] = val;

                    index++;
                }
            }
        }
    }
}

template DAUConvLayerCaffe<double>::~DAUConvLayerCaffe();
template DAUConvLayerCaffe<float>::~DAUConvLayerCaffe();

template void* DAUConvLayerCaffe<double>::allocate_workspace_mem(size_t bytes);
template void* DAUConvLayerCaffe<float>::allocate_workspace_mem(size_t bytes);


template void DAUConvLayerCaffe<double>::deallocate_workspace_mem();
template void DAUConvLayerCaffe<float>::deallocate_workspace_mem();

template vector<int> DAUConvLayerCaffe<double>::Reshape(const vector<int>& bottom_shape, const vector<int>& top);
template vector<int> DAUConvLayerCaffe<float>::Reshape(const vector<int>& bottom_shape, const vector<int>& top);

template void DAUConvLayerCaffe<float>::LayerSetUp(const DAUConvSettings& settings, const BaseDAUComponentInitializer<float>& param_initializer,
                                                      BaseDAUKernelCompute<float>* kernel_compute, BaseDAUKernelParams<float>* kernel_param, BaseDAUKernelOutput<float>* kernel_output,
                                                      const vector<int>& bottom_shape, bool in_train);
template void DAUConvLayerCaffe<double>::LayerSetUp(const DAUConvSettings& settings, const BaseDAUComponentInitializer<double>& param_initializer,
                                                       BaseDAUKernelCompute<double>* kernel_compute, BaseDAUKernelParams<double>* kernel_param, BaseDAUKernelOutput<double>* kernel_output,
                                                       const vector<int>& bottom_shape, bool in_train);

template <typename Dtype>
DAUKernelComputeCaffe<Dtype>::~DAUKernelComputeCaffe()  {
    for (int i = 0; i < this->kernels_buffers_.size(); i++)
        delete this->kernels_buffers_[i];

    for (int i = 0; i < this->param_buffers_.size(); i++)
        delete this->param_buffers_[i];
}
template <typename Dtype>
void DAUKernelComputeCaffe<Dtype>::get_kernels(BaseDAUKernelParams<Dtype> &input, BaseDAUKernelOutput<Dtype> &output, cublasHandle_t cublas_handle) {
    // when using CPU we must ensure input data is on GPU
    // since get_kernels() is implemented only for GPU !!

    if (use_gpu_ == false) {
        static_cast<DAUKernelParamsCaffe<Dtype>&>(input).set_processing_on_gpu(true);
        static_cast<DAUKernelOutputCaffe<Dtype>&>(output).set_processing_on_gpu(true);
    }

    // call parent implementation
    BaseDAUKernelCompute<Dtype>::get_kernels(input, output, cublas_handle);

    if (use_gpu_ == false) {
        static_cast<DAUKernelParamsCaffe<Dtype>&>(input).set_processing_on_gpu(false);
        static_cast<DAUKernelOutputCaffe<Dtype>&>(output).set_processing_on_gpu(false);
    }
}

template <typename Dtype>
void DAUKernelComputeCaffe<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w){

    this->num_in_channels = num_in_channels;
    this->num_out_channels = num_out_channels;
    this->num_gauss = num_gauss;
    this->kernel_h = kernel_h;
    this->kernel_w = kernel_w;

    // allocate and prepare temporary buffers for kernels
    if (this->kernels_buffers_.size() != 5) {

        for (int i = 0; i < this->kernels_buffers_.size(); i++)
            delete this->kernels_buffers_[i];

        this->kernels_buffers_.resize(5);
        for (int i = 0; i < 5; i++)
            this->kernels_buffers_[i] = new Blob<Dtype>();
    }

    for (int i = 0; i < 5; ++i)
        this->kernels_buffers_[i]->Reshape(num_in_channels, num_gauss, num_out_channels, kernel_h * kernel_w);

    // allocate and prepare temporary buffers for parameters
    if (this->param_buffers_.size() != 7){
        for (int i = 0; i < this->param_buffers_.size(); i++)
            delete this->param_buffers_[i];

        this->param_buffers_.resize(7);
        for (int i = 0; i < 7; i++)
            this->param_buffers_[i] = new Blob<Dtype>();
    }

    for (int i = 0; i < 7; ++i)
        this->param_buffers_[i]->Reshape(1, num_in_channels, num_gauss, num_out_channels);

    // pre-computed offset indexes for batched sums (when using caffe_gpu_sum)
    this->create_precompute_index(num_in_channels * num_gauss * num_out_channels, kernel_h * kernel_w);

}

template <typename Dtype>
void DAUKernelParamsCaffe<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss) {

    if (this->weight_ == false) this->weight_.reset(new Blob<Dtype>());
    if (this->mu1_ == false) this->mu1_.reset(new Blob<Dtype>());
    if (this->mu2_ == false) this->mu2_.reset(new Blob<Dtype>());
    if (this->sigma_ == false) this->sigma_.reset(new Blob<Dtype>());

    this->weight_->Reshape(1, num_in_channels, num_gauss, num_out_channels);
    this->mu1_->Reshape(1,num_in_channels, num_gauss, num_out_channels);
    this->mu2_->Reshape(1,num_in_channels, num_gauss, num_out_channels);
    this->sigma_->Reshape(1,num_in_channels, num_gauss, num_out_channels);
}

template <typename Dtype>
void DAUKernelOutputCaffe<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w) {
    this->weight_.Reshape(num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w);

    this->d_error_.Reshape(num_in_channels, num_out_channels, kernel_h, kernel_w);

    // four params == [w,mu1,mu2,sigma]
    this->d_params_.Reshape(4, num_in_channels * num_gauss, num_out_channels, kernel_h * kernel_w);
}

template DAUKernelComputeCaffe<float>::~DAUKernelComputeCaffe();
template DAUKernelComputeCaffe<double>::~DAUKernelComputeCaffe();

template void DAUKernelComputeCaffe<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);
template void DAUKernelComputeCaffe<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

template void DAUKernelParamsCaffe<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss);
template void DAUKernelParamsCaffe<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss);

template void DAUKernelOutputCaffe<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);
template void DAUKernelOutputCaffe<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

template <typename Dtype>
void DAUKernelComputeCaffe<Dtype>::create_precompute_index(const int index_size, const int kernel_size) {

    tmp_precomp_index_.Reshape(1, 1, 1, index_size + 1);

    int* tmp_precomp_index_cpu = tmp_precomp_index_.mutable_cpu_data();

    tmp_precomp_index_cpu[0] = 0;

    for (int i = 0; i < tmp_precomp_index_.count()-1; i++)
        tmp_precomp_index_cpu[i+1] = kernel_size * (i+1);

}


template <typename Dtype>
void DAUConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const DAUConvolutionParameter& param = this->layer_param().dau_conv_param();

    // verify validity of parameters
    CHECK_EQ(param.kernel_size_size(),1) << "Expecting only single kernel_size value in DAUConvolutionParameter";
    CHECK_EQ(param.pad_size(),1) << "Expecting only single pad value in DAUConvolutionParameter";
    CHECK_EQ(param.stride_size(),1) << "Expecting only single stride value in DAUConvolutionParameter";

    // copy them to DAUConvSettings
    DAUConvSettings dau_settings;

    dau_settings.bias_term = param.bias_term();

    dau_settings.num_output = param.num_output();
    dau_settings.number_units.assign(param.number_units().begin(), param.number_units().end());

    dau_settings.kernel_size = param.kernel_size(0);
    dau_settings.pad = param.pad(0);
    dau_settings.stride = param.stride(0);

    dau_settings.unit_normalization = param.unit_normalization();
    dau_settings.square_unit_normalization = param.square_unit_normalization();

    dau_settings.mean_iteration_step = param.mean_iteration_step();
    dau_settings.sigma_iteration_step = param.sigma_iteration_step();

    dau_settings.merge_iteration_step = param.merge_iteration_step();
    dau_settings.merge_threshold = param.merge_threshold();

    dau_settings.sigma_lower_bound = param.sigma_lower_bound();
    dau_settings.component_border_bound = param.component_border_bound();

    dau_settings.offsets_already_centered = param.use_already_centered_offsets();

    // define which param initializer will be used
    DAUComponentInitializerCaffe<Dtype> param_initializer(param.weight_filler(),
                                                          param.mu_filler(),
                                                          param.sigma_filler());

    // setup layer for DAU-ConvNet object
    dau_compute.LayerSetUp(dau_settings, param_initializer,
                           &this->dau_kernel_compute, &this->dau_kernel_params, &this->dau_kernel_output,
                           bottom[0]->shape(), this->phase_ == TRAIN);


    // we need to manually initialize bias
    if (dau_settings.bias_term) {
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(param.bias_filler()));
        bias_filler->Fill(this->dau_compute.param_buffer_bias_.get());
    }

    // finally connect param blobs with actual learnable blobs
    this->blobs_.resize(4 + (dau_settings.bias_term ? 1 : 0));

    // we use shared_ptr for params so just asign them to this->blobs_ array
    this->blobs_[0] = this->dau_compute.param_buffer_w_;
    this->blobs_[1] = this->dau_compute.param_buffer_mu1_;
    this->blobs_[2] = this->dau_compute.param_buffer_mu2_;
    this->blobs_[3] = this->dau_compute.param_buffer_sigma_;

    if (dau_settings.bias_term)
        this->blobs_[4] =  this->dau_compute.param_buffer_bias_;

    // Propagate gradients to the parameters (as directed by backward pass).
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const vector<int> new_top_shape = this->dau_compute.Reshape(bottom[0]->shape(), top[0]->shape());

    for (int i = 0; i < top.size(); ++i)
        top[i]->Reshape(new_top_shape);

}

template <typename Dtype>
DAUConvolutionLayer<Dtype>::~DAUConvolutionLayer() {


}
template <typename Dtype>
void plot_blob_data(Blob<Dtype>& b) {
    const Dtype* d = b.cpu_data();
    for (int n = 0;  n< b.shape(0); ++n) {
        for (int c = 0;  c< b.shape(1); ++c) {
            for (int j = 0;  j< b.shape(2); ++j) {
                for (int i = 0;  i< b.shape(3); ++i) {
                    printf("%.2f ", d[b.offset(n,c,j,i)]);
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

    template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    for (int i = 0; i < bottom.size(); ++i) {
        this->dau_compute.Forward_gpu(bottom[i]->gpu_data(), bottom[i]->shape(),
                                      top[i]->mutable_gpu_data(), top[i]->shape());
    }

}
template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

    for (int i = 0; i < top.size(); ++i) {
        this->dau_compute.Backward_gpu(top[i]->gpu_data(), top[i]->gpu_diff(), top[i]->shape(), propagate_down[i],
                                       bottom[i]->gpu_data(), bottom[i]->mutable_gpu_diff(), bottom[i]->shape(),
                                       this->param_propagate_down_);

    }
}

    template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    for (int i = 0; i < bottom.size(); ++i) {
        this->dau_compute.Forward_cpu(bottom[i]->cpu_data(), bottom[i]->shape(),
                                      top[i]->mutable_cpu_data(), top[i]->shape());
    }
}

template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < top.size(); ++i) {
        this->dau_compute.Backward_cpu(top[i]->cpu_data(), top[i]->cpu_diff(), top[i]->shape(), propagate_down[i],
                                       bottom[i]->cpu_data(), bottom[i]->mutable_cpu_diff(), bottom[i]->shape(),
                                       this->param_propagate_down_);

    }
}


INSTANTIATE_CLASS(DAUConvolutionLayer);

}   // namespace caffe
