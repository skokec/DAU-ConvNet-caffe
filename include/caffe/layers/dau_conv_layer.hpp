#ifndef CAFFE_DAU_CONV_LAYER_HPP_
#define CAFFE_DAU_CONV_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"

#include "dau_conv/base_dau_conv_layer.hpp"

#include "caffe/util/device_alternate.hpp"

namespace caffe {

// we will be using base classes from DAUConvNet
using DAUConvNet::DAUConvSettings;

using DAUConvNet::BaseDAUConvLayer;
using DAUConvNet::BaseDAUComponentInitializer;

using DAUConvNet::BaseDAUKernelCompute;
using DAUConvNet::BaseDAUKernelOutput;
using DAUConvNet::BaseDAUKernelParams;



////////////////////////////////////////////////////////////////////////////////
// Caffe implementation of buffers used in DAUKernel*

template <typename Dtype>
class DAUKernelParamsCaffe : public  BaseDAUKernelParams<Dtype> {
public:
	explicit DAUKernelParamsCaffe(bool use_gpu = true) : use_gpu_(use_gpu) {}

	void reshape(int num_in_channels, int num_out_channels, int num_gauss);

	virtual Dtype* weight() { return use_gpu_ ? this->weight_->mutable_gpu_data() : this->weight_->mutable_cpu_data(); }
	virtual Dtype* mu1() { return use_gpu_ ? this->mu1_->mutable_gpu_data() : this->mu1_->mutable_cpu_data(); }
	virtual Dtype* mu2() { return use_gpu_ ? this->mu2_->mutable_gpu_data() : this->mu2_->mutable_cpu_data(); }
	virtual Dtype* sigma() { return use_gpu_ ? this->sigma_->mutable_gpu_data() : this->sigma_->mutable_cpu_data(); }

	void set_processing_on_gpu(bool do_on_gpu) { use_gpu_ = do_on_gpu; }

	shared_ptr<Blob<Dtype> > weight_, mu1_, mu2_, sigma_; // CPU for setting (once) GPU for computing, except for sigma
private:
	bool use_gpu_;
};


template <typename Dtype>
class DAUKernelOutputCaffe : public BaseDAUKernelOutput<Dtype> {
public:
	explicit DAUKernelOutputCaffe(bool use_gpu = true) : use_gpu_(use_gpu) {}

	virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

	virtual Dtype* weight() { return use_gpu_ ? this->weight_.mutable_gpu_data() : this->weight_.mutable_cpu_data(); }
	virtual Dtype* d_error() { return use_gpu_ ? this->d_error_.mutable_gpu_data() : this->d_error_.mutable_cpu_data(); }
	virtual Dtype* d_params() { return use_gpu_ ? this->d_params_.mutable_gpu_data() : this->d_params_.mutable_cpu_data(); }

	void set_processing_on_gpu(bool do_on_gpu) { use_gpu_ = do_on_gpu; }

	// main filter weights
	Blob<Dtype> weight_;

	// derivative weights for back-propagation and all four parameters
	Blob<Dtype> d_error_;
	Blob<Dtype> d_params_; // four params == [w,mu1,mu2,sigma]
private:
	bool use_gpu_;
};

template <typename Dtype>
class DAUKernelComputeCaffe : public BaseDAUKernelCompute<Dtype> {
public:
	explicit  DAUKernelComputeCaffe(bool use_gpu = true) : use_gpu_(use_gpu) {}

	virtual ~DAUKernelComputeCaffe();

	virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss,
						 int kernel_h, int kernel_w);

	virtual void get_kernels(BaseDAUKernelParams<Dtype> &input, BaseDAUKernelOutput<Dtype> &output, cublasHandle_t cublas_handle);

	virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { return this->param_buffers_[index]->mutable_gpu_data(); }
	virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { return this->param_buffers_[index]->mutable_gpu_data(); }
	virtual int* precomp_index() { return this->tmp_precomp_index_.mutable_gpu_data(); }

	void set_processing_on_gpu(bool do_on_gpu) { use_gpu_ = do_on_gpu; }

protected:
	void create_precompute_index(const int index_size, const int kernel_size);

	// intermediate buffers when computing derivative kernels in precompute_guassian_weights_gpu
	// temporary buffers for pre-computed sigma^2, sigma^3 and 1/2*sigma^2
	vector<Blob<Dtype>* > param_buffers_;
	vector<Blob<Dtype>* > kernels_buffers_;

	Blob<int> tmp_precomp_index_;// pre-computed indexes for caffe_gpu_sum in get_kernels
private:
	bool use_gpu_;
};


////////////////////////////////////////////////////////////////////////////////
// GPU version of Caffe buffers used in DAUKernel*
/*
template <typename Dtype>
class DAUKernelParamsGPU : public  DAUKernelParams<Dtype> {
public:

};

template <typename Dtype>
class DAUKernelOutputGPU : public DAUKernelOutput<Dtype> {
public:

};

template <typename Dtype>
class DAUKernelCompute : public DAUKernelCompute<Dtype> {
public:


};

//
template <typename Dtype>
class DAUKernelParamsCPU : public  DAUKernelParams<Dtype> {
public:

    virtual Dtype* weight() { return this->weight_->mutable_cpu_data(); }
    virtual Dtype* mu1() { return this->mu1_->mutable_cpu_data(); }
    virtual Dtype* mu2() { return this->mu2_->mutable_cpu_data(); }
    virtual Dtype* sigma() { return this->sigma_->mutable_cpu_data(); }
};

template <typename Dtype>
class DAUKernelOutputCPU : public DAUKernelOutput<Dtype> {
public:
    virtual Dtype* weight() { return this->weight_.mutable_cpu_data(); }
    virtual Dtype* d_error() { return this->d_error_.mutable_cpu_data(); }
    virtual Dtype* d_params() { return this->d_params_.mutable_cpu_data(); }
};

template <typename Dtype>
class DAUKernelComputeCPU : public DAUKernelCompute<Dtype> {
public:

    virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { return this->param_buffers_[index]->mutable_cpu_data(); }
    virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { return this->param_buffers_[index]->mutable_cpu_data(); }
    virtual int* precomp_index() { return this->tmp_precomp_index_.mutable_gpu_data(); }

};
*/
////////////////////////////////////////////////////////////////////////////////
// Caffe GPU version of DAUConvolution layer (BaseDAUConvLayer)

template <typename Dtype>
class DAUComponentInitializerCaffe : public BaseDAUComponentInitializer<Dtype> {
public:

	DAUComponentInitializerCaffe(const FillerParameter& weight_filler,
								 const FillerParameter& mu_filler,
								 const FillerParameter& sigma_filler) :
			weight_filler_(weight_filler), mu_filler_(mu_filler), sigma_filler_(sigma_filler) {

	}

	virtual void InitializeParameters(const DAUConvSettings& settings, Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
                                      int num_units_per_x, int num_units_per_y, int num_units_ignore,
									  int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const;

private:
	// param fillers
	FillerParameter weight_filler_;
	FillerParameter mu_filler_;
	FillerParameter sigma_filler_;
};

template <typename Dtype>
class DAUConvLayerCaffe : public  BaseDAUConvLayer<Dtype> {
public:

	explicit DAUConvLayerCaffe(cublasHandle_t cublas_handle, bool ignore_edge_gradients = false)
			: BaseDAUConvLayer<Dtype>(cublas_handle, ignore_edge_gradients), own_workspace_data(0), do_on_gpu_(true) {

	}

	virtual ~DAUConvLayerCaffe();

	virtual void LayerSetUp(const DAUConvSettings& settings,
							const BaseDAUComponentInitializer<Dtype>& param_initializer,
							BaseDAUKernelCompute<Dtype>* kernel_compute,
							BaseDAUKernelParams<Dtype>* kernel_param,
							BaseDAUKernelOutput<Dtype>* kernel_output,
							const vector<int>& bottom_shape, bool in_train = true);

	virtual vector<int> Reshape(const vector<int>& bottom_shape, const vector<int>& top);

	// make compute_output_shape() public
	virtual void compute_output_shape() { return BaseDAUConvLayer<Dtype>::compute_output_shape(); }

	void set_processing_on_gpu(bool do_on_gpu) { do_on_gpu_ = do_on_gpu; }

	// parameters to learn
	shared_ptr<Blob<Dtype> > param_buffer_w_;
	shared_ptr<Blob<Dtype> > param_buffer_mu1_;
	shared_ptr<Blob<Dtype> > param_buffer_mu2_;
	shared_ptr<Blob<Dtype> > param_buffer_sigma_;
	shared_ptr<Blob<Dtype> > param_buffer_bias_;
protected:
	virtual bool is_data_on_gpu() { return do_on_gpu_; }

    virtual void reshape_params(const vector<int>& shape) ;

	virtual bool update_prefiltering_kernels(cudaStream_t stream);

	// learnable parameters of size
	virtual Dtype* param_w() { return is_data_on_gpu() ? param_buffer_w_->mutable_gpu_data() : param_buffer_w_->mutable_cpu_data(); }
	virtual Dtype* param_mu1() { return is_data_on_gpu() ? param_buffer_mu1_->mutable_gpu_data() : param_buffer_mu1_->mutable_cpu_data(); }
	virtual Dtype* param_mu2() { return is_data_on_gpu() ? param_buffer_mu2_->mutable_gpu_data() : param_buffer_mu2_->mutable_cpu_data(); }
	virtual Dtype* param_sigma() { return is_data_on_gpu() ? param_buffer_sigma_->mutable_gpu_data() : param_buffer_sigma_->mutable_cpu_data(); }
	virtual Dtype* param_bias() { return is_data_on_gpu() ? param_buffer_bias_->mutable_gpu_data() : param_buffer_bias_->mutable_cpu_data(); }

	// gradient buffers for learnable parameters
	virtual Dtype* param_w_grad() { return is_data_on_gpu() ? param_buffer_w_->mutable_gpu_diff() : param_buffer_w_->mutable_cpu_data(); }
	virtual Dtype* param_mu1_grad() { return is_data_on_gpu() ? param_buffer_mu1_->mutable_gpu_diff() : param_buffer_mu1_->mutable_cpu_data(); }
	virtual Dtype* param_mu2_grad() { return is_data_on_gpu() ? param_buffer_mu2_->mutable_gpu_diff() : param_buffer_mu2_->mutable_cpu_data(); }
	virtual Dtype* param_sigma_grad(){ return is_data_on_gpu() ? param_buffer_sigma_->mutable_gpu_diff() : param_buffer_sigma_->mutable_cpu_data(); }
	virtual Dtype* param_bias_grad() { return is_data_on_gpu() ? param_buffer_bias_->mutable_gpu_diff() : param_buffer_bias_->mutable_cpu_data(); }

	// remaining intermediate/temporary buffers
	virtual Dtype* temp_bwd_gradients() { return is_data_on_gpu() ? bwd_gradients_.mutable_gpu_data() : bwd_gradients_.mutable_cpu_data() ; }
	virtual Dtype* temp_interm_buffer() { return is_data_on_gpu() ? interm_buffer_.mutable_gpu_data() : interm_buffer_.mutable_cpu_data() ; }
	virtual Dtype* temp_param_buffer() { return is_data_on_gpu() ? tmp_param_buffer_.mutable_gpu_data() : tmp_param_buffer_.mutable_cpu_data() ; }
	virtual Dtype* temp_col_buffer() { return is_data_on_gpu() ? col_buffer_.mutable_gpu_data() : col_buffer_.mutable_cpu_data() ; }
	virtual Dtype* temp_bias_multiplier() { return is_data_on_gpu() ? bias_multiplier_.mutable_gpu_data() : bias_multiplier_.mutable_cpu_data() ; }

	virtual void* allocate_workspace_mem(size_t bytes);
	virtual void deallocate_workspace_mem();

	// accumulated gradients
	Blob<Dtype> bwd_gradients_;

	// additional buffers
	Blob<Dtype> interm_buffer_; // GPU only
	Blob<Dtype> tmp_param_buffer_; // GPU and CPU

	Blob<Dtype> col_buffer_; // CPU only
	Blob<Dtype> bias_multiplier_; // GPU and CPU

	// workspace memory that we have allocated
	void* own_workspace_data;

	bool do_on_gpu_;
};


/**
 * DAUConvolutionLayer
 *
 * Implementation of Deep Compositional Layer that introduces two constraints which results in Displaced Aggregation
 * Units (DAU) as presented in CVPR18. This implementation is efficient and allows for 3-5 times faster computation
 * of inference and learning compared to Deep Compositional Layer from ICPR16 paper. This does introduces a slight
 * loss of information and is only an aproximation of the original GaussianConvLayer, but perofrmance is not impacted.
 *
 * DAUConvolutionLayer implements two constraints on composition/units :
 *  - single sigma/variance for the whole layer (shared across all features)
 *  - internal discretization of mu1/mu2 position values
 * Due to discretization of mu1/mu2 values this implementation handles sub-pixel offsets using bilinear interpolation
 * of input channels.
 *
 * Due to CUDA implementation this method does not compute accuretely on bottom/right border (1px). Those values
 * are used in gradient accumulation unless ignore_edge_gradients_ is set to true. Border values are back-propagated
 * nevertheless.
 *
 *
 * TODO:
 *  - add sharing of GPU memory accross layers that are computed in sequence
 *  - add stride>1 (currently allows only stride=1)
 *  - improve cudaStream for forward and backward pass
 *  - combine convolve and input preparation forward and backward pass (might shave 5-10% off the whole computation time)
 *
 *
 * @tparam Dtype
 */
template <typename Dtype>
class DAUConvolutionLayer : public Layer<Dtype>{
public:

	explicit DAUConvolutionLayer(const LayerParameter& param, bool ignore_edge_gradients = false)
	  : Layer<Dtype>(param), dau_compute(Caffe::cublas_handle(), ignore_edge_gradients) {}

	virtual ~DAUConvolutionLayer();

	virtual inline const char* type() const { return "DAUConvolutionLayer"; }

	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline bool EqualNumBottomTopBlobs() const { return true; }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	void set_processing_on_gpu(bool do_on_gpu) {
		dau_compute.set_processing_on_gpu(do_on_gpu);
		dau_kernel_compute.set_processing_on_gpu(do_on_gpu);
		dau_kernel_params.set_processing_on_gpu(do_on_gpu);
		dau_kernel_output.set_processing_on_gpu(do_on_gpu);
	}
protected:

	virtual void compute_output_shape() { return dau_compute.compute_output_shape(); }
	virtual inline bool reverse_dimensions() { return false; }

private:
	// compute obj and buffers (param and output) for our Gaussian kernel
	// (we actually have only one kernel in buffer but data structure is general)
	DAUKernelComputeCaffe<Dtype> dau_kernel_compute;
	DAUKernelParamsCaffe<Dtype> dau_kernel_params;
	DAUKernelOutputCaffe<Dtype> dau_kernel_output;

public:
    // must be public only for testing reasons
    DAUConvLayerCaffe<Dtype> dau_compute;
};

}  // namespace caffe

#endif  // CAFFE_DAU_CONV_LAYER_HPP_
