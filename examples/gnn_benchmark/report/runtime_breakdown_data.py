
fwd_strs = {
    'dace_csr':"""
Name	Start	Duration	GPU	Context
examples_gnn_benchmark_implementations_gcn_implementations_torch_geometricDOTnnDOTconvDOTgcn_convDOTGCNConv_0_expansion_154_1_0_0(float const*, float*)	77,694s	99,616 μs	GPU 0	Stream 7
volta_sgemm_128x64_tn	77,6941s	485,820 μs	GPU 0	Stream 7
void cusparse::partition_kernel<128, int, int>(int const*, int, int, int, int, int*)	77,6946s	10,592 μs	GPU 0	Stream 7
void cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<int, float, float, float>, true, false, false, int, int, int, int, float, float, float>(int, int, int, int, int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, int const*, float const*, long, float const*, int, long, float*, int, long)	77,6946s	1,476 ms	GPU 0	Stream 7
_elementwise__map_0_0_14(float const*, float*)	77,6961s	222,943 μs	GPU 0	Stream 7
examples_gnn_benchmark_implementations_gcn_implementations_torch_geometricDOTnnDOTconvDOTgcn_convDOTGCNConv_2_expansion_154_2_0_0(float const*, float*)	77,6963s	43,680 μs	GPU 0	Stream 7
void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*)	77,6964s	304,158 μs	GPU 0	Stream 7
void cusparse::partition_kernel<128, int, int>(int const*, int, int, int, int, int*)	77,6967s	10,592 μs	GPU 0	Stream 7
void cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<int, float, float, float>, true, false, false, int, int, int, int, float, float, float>(int, int, int, int, int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, int const*, float const*, long, float const*, int, long, float*, int, long)	77,6967s	559,356 μs	GPU 0	Stream 7
grid_3_0_0(float const*, float*)	77,6973s	295,550 μs	GPU 0	Stream 7
outer_fused_0_0_30(float const*, float const*, float*, float*)	77,6976s	110,624 μs	GPU 0	Stream 7
grid_4_0_0(float const*, float*)	77,6977s	295,517 μs	GPU 0	Stream 7
_numpy_log__map_0_0_23(float const*, float*)	77,698s	12,032 μs	GPU 0	Stream 7
_Sub__map_0_0_27(float*, float const*, float const*)	77,698s	83,712 μs	GPU 0	Stream 7""",
#     'dace_coo':"""Name	Start	Duration	TID	GPU	Context
# volta_sgemm_128x64_tn	184,96s	510,108 μs	-	GPU 0	Stream 13
# dace_coo single run. [3,036 ms]	184,961s	3,036 ms	1895293	GPU 0	Stream 14
# examples_gnn_benchmark_implementations_gcn_implementations_torch_geometricDOTnnDOTconvDOTgcn_convDOTGCNConv_0_expansion_317_1_0_4(float const*, float*)	184,961s	132,895 μs	-	GPU 0	Stream 14
# void cusparse::coomm_v2_kernel<cusparse::CooMMPolicy<int, float, float, float>, true, false, false, int, int, float, float, float>(int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, float const*, long, float const*, int, long, float*, int, long, int)	184,961s	1,462 ms	-	GPU 0	Stream 13
# _elementwise__map_0_0_14(float const*, float*)	184,963s	224,639 μs	-	GPU 0	Stream 13
# examples_gnn_benchmark_implementations_gcn_implementations_torch_geometricDOTnnDOTconvDOTgcn_convDOTGCNConv_2_expansion_317_2_0_4(float const*, float*)	184,963s	42,527 μs	-	GPU 0	Stream 14
# void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*)	184,963s	302,909 μs	-	GPU 0	Stream 13
# void cusparse::coomm_v2_kernel<cusparse::CooMMPolicy<int, float, float, float>, true, false, false, int, int, float, float, float>(int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, float const*, long, float const*, int, long, float*, int, long, int)	184,963s	516,188 μs	-	GPU 0	Stream 13
# grid_3_0_0(float const*, float*)	184,964s	295,518 μs	-	GPU 0	Stream 14
# outer_fused_0_0_30(float const*, float const*, float*, float*)	184,964s	109,759 μs	-	GPU 0	Stream 13
# grid_4_0_0(float const*, float*)	184,964s	295,486 μs	-	GPU 0	Stream 13
# _numpy_log__map_0_0_23(float const*, float*)	184,964s	11,744 μs	-	GPU 0	Stream 13
# _Sub__map_0_0_27(float*, float const*, float const*)	184,964s	83,551 μs	-	GPU 0	Stream 13""",
'torch_edge_list': """
Name	Start	Duration	TID	GPU	Context
volta_sgemm_128x64_tn	62,6235s	484,701 μs	-	GPU 0	Stream 7
void at::native::<unnamed>::indexSelectLargeIndex<float, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T2, T3>, int, int, T3, T3, long)	62,624s	2,675 ms	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	62,6267s	1,684 ms	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	62,6283s	99,199 μs	-	GPU 0	Stream 7
void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)4, void at::native::_cuda_scatter_gather_internal_kernel<(bool)1, float>::operator ()<at::native::ReduceAdd>(at::TensorIterator &, long, long, long, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	62,6284s	1,941 ms	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	62,6304s	215,743 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)	62,6306s	212,446 μs	-	GPU 0	Stream 7
void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, const T1 *, const T1 *, T1, T1, int, cublasLtEpilogue_t, int, const void *, long)	62,6308s	326,110 μs	-	GPU 0	Stream 7
void at::native::<unnamed>::indexSelectLargeIndex<float, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T2, T3>, int, int, T3, T3, long)	62,6311s	881,689 μs	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	62,632s	534,236 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	62,6326s	33,184 μs	-	GPU 0	Stream 7
void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)4, void at::native::_cuda_scatter_gather_internal_kernel<(bool)1, float>::operator ()<at::native::ReduceAdd>(at::TensorIterator &, long, long, long, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	62,6326s	620,603 μs	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	62,6332s	70,752 μs	-	GPU 0	Stream 7
void <unnamed>::softmax_warp_forward<float, float, float, (int)6, (bool)1, (bool)0>(T2 *, const T1 *, int, int, int, const bool *, int, bool)	62,6333s	72,671 μs	-	GPU 0	Stream 7""",
'torch_edge_list_compiled': """
Name	Start	Duration	TID	GPU	Context
volta_sgemm_128x64_tn	62,9748s	483,580 μs	-	GPU 0	Stream 7
triton__0d1d	62,9753s	100,063 μs	-	GPU 0	Stream 7
triton__0d1d2d3d4d	62,9754s	1,390 ms	-	GPU 0	Stream 7
triton__0d1d2d3d	62,9768s	213,439 μs	-	GPU 0	Stream 7
void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, const T1 *, const T1 *, T1, T1, int, cublasLtEpilogue_t, int, const void *, long)	62,977s	328,798 μs	-	GPU 0	Stream 7
triton__0d1	62,9774s	33,728 μs	-	GPU 0	Stream 7
triton__0d1d2d3d4d	62,9774s	535,324 μs	-	GPU 0	Stream 7
triton__0d1d2d34	62,9779s	86,623 μs	-	GPU 0	Stream 7""",
'torch_csr': """
Name	Start	Duration	GPU	Context
volta_sgemm_128x64_tn	77,5661s	484,252 μs	GPU 0	Stream 7
void spmm_kernel<float, (ReductionType)0, true>(long const*, long const*, float const*, float const*, float*, long*, int, int, int, int)	77,5665s	2,916 ms	GPU 0	Stream 7
void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int, false>, OffsetCalculator<1, unsigned int, false>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int, false>, OffsetCalculator<1, unsigned int, false>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)	77,5695s	214,750 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::clamp_min_scalar_kernel_impl(at::TensorIterator&, c10::Scalar)::{lambda()#1}::operator()() const::{lambda()#8}::operator()() const::{lambda(float)#1}, at::detail::Array<char*, 2> >(int, at::native::(anonymous namespace)::clamp_min_scalar_kernel_impl(at::TensorIterator&, c10::Scalar)::{lambda()#1}::operator()() const::{lambda()#8}::operator()() const::{lambda(float)#1}, at::detail::Array<char*, 2>)	77,5697s	212,575 μs	GPU 0	Stream 7
void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*)	77,5699s	307,806 μs	GPU 0	Stream 7
void spmm_kernel<float, (ReductionType)0, true>(long const*, long const*, float const*, float const*, float*, long*, int, int, int, int)	77,5702s	1,747 ms	GPU 0	Stream 7
void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int, false>, OffsetCalculator<1, unsigned int, false>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int, false>, OffsetCalculator<1, unsigned int, false>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)	77,5719s	70,719 μs	GPU 0	Stream 7
void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, true>(float*, float const*, int, int, int)	77,572s	72,064 μs	GPU 0	Stream 7
"""
}

bwd_strs = {
    'dace_csr': """
Name	Start	Duration	GPU	Context
volta_sgemm_128x64_tn	78,2162s	444,732 μs	GPU 0	Stream 7
void cusparse::partition_kernel<128, int, int>(int const*, int, int, int, int, int*)	78,2166s	10,367 μs	GPU 0	Stream 7
void cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<int, float, float, float>, true, false, false, int, int, int, int, float, float, float>(int, int, int, int, int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, int const*, float const*, long, float const*, int, long, float*, int, long)	78,2166s	1,473 ms	GPU 0	Stream 7
_elementwise__map_0_0_14(float const*, float*)	78,2181s	223,230 μs	GPU 0	Stream 7
examples_gnn_benchmark_implementations_gcn_implementations_torch_geometricDOTnnDOTconvDOTgcn_convDOTGCNConv_2_expansion_154_2_0_0(float const*, float*)	78,2183s	39,808 μs	GPU 0	Stream 7
void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*)	78,2184s	283,805 μs	GPU 0	Stream 7
void cusparse::partition_kernel<128, int, int>(int const*, int, int, int, int, int*)	78,2186s	9,984 μs	GPU 0	Stream 7
void cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<int, float, float, float>, true, false, false, int, int, int, int, float, float, float>(int, int, int, int, int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, int const*, float const*, long, float const*, int, long, float*, int, long)	78,2186s	547,357 μs	GPU 0	Stream 7
grid_0_0_50(float const*, float*)	78,2192s	268,413 μs	GPU 0	Stream 7
outer_fused_0_0_30(float const*, float const*, float*, float*)	78,2195s	109,759 μs	GPU 0	Stream 7
grid_0_0_33(float*, float const*)	78,2196s	268,734 μs	GPU 0	Stream 7
_numpy_log__map_0_0_23(float const*, float*)	78,2198s	11,775 μs	GPU 0	Stream 7
_Sub__map_0_0_27(float*, float const*, float const*)	78,2199s	81,023 μs	GPU 0	Stream 7
void at::native::(anonymous namespace)::nll_loss_forward_reduce_cuda_kernel_2d<float, float, long>(float*, float*, float*, long*, float*, bool, int, int, int, long)	78,2199s	3,977 ms	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)	78,2239s	3,232 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)	78,2239s	44,255 μs	GPU 0	Stream 7
void at::native::(anonymous namespace)::nll_loss_backward_reduce_cuda_kernel_2d<float, long>(float*, float*, long*, float*, float*, bool, int, int, int, long)	78,224s	2,360 ms	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)	78,2263s	3,040 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)	78,2263s	3,552 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)	78,2263s	2,560 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)	78,2263s	2,592 μs	GPU 0	Stream 7
grid_0_0_37(float*, float const*)	78,2263s	268,158 μs	GPU 0	Stream 7
outer_fused_0_0_29(float const*, float const*, float*, float const*)	78,2266s	107,135 μs	GPU 0	Stream 7
grid_0_0_68(float const*, float*)	78,2267s	600,412 μs	GPU 0	Stream 7
void cusparse::matrix_scalar_multiply_kernel<cusparse::MatrixWiseMulPolicy, true, long, float, float>(long, long, long, cusparse::KernelCoeff<float>, float*)	78,2273s	98,591 μs	GPU 0	Stream 7
void cusparse::partition_kernel<128, int, int>(int const*, int, int, int, int, int*)	78,2274s	10,688 μs	GPU 0	Stream 7
void cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<int, float, float, float>, true, false, false, int, int, int, int, float, float, float>(int, int, int, int, int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, int const*, float const*, long, float const*, int, long, float*, int, long)	78,2274s	1,466 ms	GPU 0	Stream 7
volta_sgemm_64x32_sliced1x4_nt	78,2289s	323,902 μs	GPU 0	Stream 7
void splitKreduce_kernel<float, float, float, float, true, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, void*, long, float*, int*)	78,2292s	7,776 μs	GPU 0	Stream 7
volta_sgemm_128x64_nn	78,2292s	176,318 μs	GPU 0	Stream 7
void cusparse::matrix_scalar_multiply_kernel<cusparse::MatrixWiseMulPolicy, true, long, float, float>(long, long, long, cusparse::KernelCoeff<float>, float*)	78,2294s	99,615 μs	GPU 0	Stream 7
void cusparse::partition_kernel<128, int, int>(int const*, int, int, int, int, int*)	78,2295s	10,784 μs	GPU 0	Stream 7
void cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<int, float, float, float>, false, false, false, int, int, int, int, float, float, float>(int, int, int, int, int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, int const*, float const*, long, float const*, int, long, float*, int, long)	78,2295s	742,811 μs	GPU 0	Stream 7
_elementwise__map_0_0_20(float const*, float*, float const*)	78,2303s	315,742 μs	GPU 0	Stream 7
grid_0_0_54(float const*, float*)	78,2306s	621,467 μs	GPU 0	Stream 7
void cusparse::matrix_scalar_multiply_kernel<cusparse::MatrixWiseMulPolicy, true, long, float, float>(long, long, long, cusparse::KernelCoeff<float>, float*)	78,2312s	98,175 μs	GPU 0	Stream 7
void cusparse::partition_kernel<128, int, int>(int const*, int, int, int, int, int*)	78,2313s	10,657 μs	GPU 0	Stream 7
void cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<int, float, float, float>, true, false, false, int, int, int, int, float, float, float>(int, int, int, int, int, int, cusparse::KernelCoeffs<float>, int, int, int const*, int const*, int const*, float const*, long, float const*, int, long, float*, int, long)	78,2313s	1,523 ms	GPU 0	Stream 7
volta_sgemm_32x128_nt	78,2328s	418,301 μs	GPU 0	Stream 7
void splitKreduce_kernel<float, float, float, float, true, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, void*, long, float*, int*)	78,2333s	8,256 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>)	78,2333s	3,392 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>)	78,2333s	2,880 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>)	78,2333s	3,680 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>)	78,2333s	3,104 μs	GPU 0	Stream 7""",
    'torch_edge_list': """Name	Start	Duration	TID	GPU	Context
volta_sgemm_128x64_tn	63,2764s	448,285 μs	-	GPU 0	Stream 7
void at::native::<unnamed>::indexSelectLargeIndex<float, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T2, T3>, int, int, T3, T3, long)	63,2769s	2,520 ms	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2794s	1,682 ms	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	63,2811s	99,199 μs	-	GPU 0	Stream 7
void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)4, void at::native::_cuda_scatter_gather_internal_kernel<(bool)1, float>::operator ()<at::native::ReduceAdd>(at::TensorIterator &, long, long, long, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2812s	1,945 ms	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2831s	215,934 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)	63,2834s	212,158 μs	-	GPU 0	Stream 7
void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, const T1 *, const T1 *, T1, T1, int, cublasLtEpilogue_t, int, const void *, long)	63,2836s	303,646 μs	-	GPU 0	Stream 7
void at::native::<unnamed>::indexSelectLargeIndex<float, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T2, T3>, int, int, T3, T3, long)	63,2839s	828,185 μs	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2847s	532,893 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	63,2852s	32,640 μs	-	GPU 0	Stream 7
void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)4, void at::native::_cuda_scatter_gather_internal_kernel<(bool)1, float>::operator ()<at::native::ReduceAdd>(at::TensorIterator &, long, long, long, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2853s	609,307 μs	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2859s	70,303 μs	-	GPU 0	Stream 7
void <unnamed>::softmax_warp_forward<float, float, float, (int)6, (bool)1, (bool)0>(T2 *, const T1 *, int, int, int, const bool *, int, bool)	63,286s	73,375 μs	-	GPU 0	Stream 7
void at::native::<unnamed>::nll_loss_forward_reduce_cuda_kernel_2d<float, float, long>(T1 *, T1 *, T1 *, T3 *, T1 *, bool, long, long, long, long)	63,286s	4,171 ms	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	63,2902s	3,168 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	63,2902s	30,976 μs	-	GPU 0	Stream 7
void at::native::<unnamed>::nll_loss_backward_reduce_cuda_kernel_2d<float, long>(T1 *, T1 *, T2 *, T1 *, T1 *, bool, int, int, long, long)	63,2902s	2,365 ms	-	GPU 0	Stream 7
void <unnamed>::softmax_warp_backward<float, float, float, (int)6, (bool)1, (bool)0>(T2 *, const T1 *, const T1 *, int, int, int, const bool *)	63,2926s	98,975 μs	-	GPU 0	Stream 7
Memset	63,2927s	1,344 μs	-	GPU 0	Stream 7
void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, float, (int)4>>(T3)	63,2927s	64,640 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,2928s	3,776 μs	-	GPU 0	Stream 7
void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)4, void at::native::_cuda_scatter_gather_internal_kernel<(bool)0, at::native::OpaqueType<(int)4>>::operator ()<at::native::TensorAssign>(at::TensorIterator &, long, long, long, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2928s	527,740 μs	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2933s	533,276 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	63,2938s	32,959 μs	-	GPU 0	Stream 7
void at::native::indexFuncLargeIndex<float, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1, at::native::<unnamed>::ReduceAdd>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T2, T3>, int, int, T3, T3, long, long, const T8 &, T1)	63,2939s	873,913 μs	-	GPU 0	Stream 7
volta_sgemm_64x32_sliced1x4_nt	63,2948s	330,046 μs	-	GPU 0	Stream 7
void splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)1, (bool)0, (bool)0>(cublasSplitKParams<T6>, const T4 *, const T5 *, T5 *, const T6 *, const T6 *, const T7 *, const T4 *, T7 *, void *, long, T6 *, int *)	63,2951s	8,768 μs	-	GPU 0	Stream 7
volta_sgemm_128x64_nn	63,2951s	184,094 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,2953s	3,776 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, void at::native::<unnamed>::threshold_kernel_impl<float>(at::TensorIteratorBase &, T1, T1)::[lambda(float, float) (instance 1)]>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,2953s	308,606 μs	-	GPU 0	Stream 7
Memset	63,2956s	1,344 μs	-	GPU 0	Stream 7
void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, float, (int)4>>(T3)	63,2956s	213,407 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,2958s	3,584 μs	-	GPU 0	Stream 7
void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)4, void at::native::_cuda_scatter_gather_internal_kernel<(bool)0, at::native::OpaqueType<(int)4>>::operator ()<at::native::TensorAssign>(at::TensorIterator &, long, long, long, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2958s	1,565 ms	-	GPU 0	Stream 7
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)	63,2974s	1,683 ms	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	63,2991s	99,551 μs	-	GPU 0	Stream 7
void at::native::indexFuncLargeIndex<float, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1, at::native::<unnamed>::ReduceAdd>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T2, T3>, int, int, T3, T3, long, long, const T8 &, T1)	63,2992s	2,508 ms	-	GPU 0	Stream 7
volta_sgemm_32x128_nt	63,3017s	424,797 μs	-	GPU 0	Stream 7
void splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)1, (bool)0, (bool)0>(cublasSplitKParams<T6>, const T4 *, const T5 *, T5 *, const T6 *, const T6 *, const T7 *, const T4 *, T7 *, void *, long, T6 *, int *)	63,3021s	8,704 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,3021s	3,584 μs	-	GPU 0	Stream 7
""",
    'torch_edge_list_compiled': """Name	Start	Duration	TID	GPU	Context
volta_sgemm_128x64_tn	63,4154s	443,548 μs	-	GPU 0	Stream 7
triton__0d1d2d	63,4158s	196,318 μs	-	GPU 0	Stream 7
triton__0d1d2d3d4d	63,416s	1,403 ms	-	GPU 0	Stream 7
triton__0d1d2d3d	63,4174s	213,247 μs	-	GPU 0	Stream 7
void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, const T1 *, const T1 *, T1, T1, int, cublasLtEpilogue_t, int, const void *, long)	63,4176s	300,381 μs	-	GPU 0	Stream 7
triton__0d1d2	63,4179s	63,872 μs	-	GPU 0	Stream 7
triton__0d1d2d3d4d	63,418s	525,404 μs	-	GPU 0	Stream 7
triton__0d1d2d34	63,4185s	82,719 μs	-	GPU 0	Stream 7
void at::native::<unnamed>::nll_loss_forward_reduce_cuda_kernel_2d<float, float, long>(T1 *, T1 *, T1 *, T3 *, T1 *, bool, long, long, long, long)	63,4186s	4,156 ms	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	63,4228s	3,488 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)	63,4228s	30,944 μs	-	GPU 0	Stream 7
void at::native::<unnamed>::nll_loss_backward_reduce_cuda_kernel_2d<float, long>(T1 *, T1 *, T2 *, T1 *, T1 *, bool, int, int, long, long)	63,4228s	2,356 ms	-	GPU 0	Stream 7
triton__0d1d23	63,4251s	41,344 μs	-	GPU 0	Stream 7
triton__0d1d2d3d4d5	63,4252s	78,112 μs	-	GPU 0	Stream 7
triton__0d1d23	63,4253s	4,673 μs	-	GPU 0	Stream 7
triton__0d1d2	63,4253s	65,152 μs	-	GPU 0	Stream 7
triton__0d1d2d3d4d5d6d7d	63,4253s	780,762 μs	-	GPU 0	Stream 7
volta_sgemm_64x32_sliced1x4_nt	63,4261s	258,622 μs	-	GPU 0	Stream 7
void splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)1, (bool)0, (bool)0>(cublasSplitKParams<T6>, const T4 *, const T5 *, T5 *, const T6 *, const T6 *, const T7 *, const T4 *, T7 *, void *, long, T6 *, int *)	63,4264s	8,384 μs	-	GPU 0	Stream 7
volta_sgemm_128x64_nn	63,4264s	181,567 μs	-	GPU 0	Stream 7
triton__0d1d2d3d4	63,4266s	210,078 μs	-	GPU 0	Stream 7
triton__0d1d2d3	63,4268s	5,920 μs	-	GPU 0	Stream 7
triton__0d1d2d	63,4268s	208,414 μs	-	GPU 0	Stream 7
triton__0d1d2d3d4d5d6d	63,427s	1,893 ms	-	GPU 0	Stream 7
volta_sgemm_32x128_nt	63,4289s	425,245 μs	-	GPU 0	Stream 7
void splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)1, (bool)0, (bool)0>(cublasSplitKParams<T6>, const T4 *, const T5 *, T5 *, const T6 *, const T6 *, const T7 *, const T4 *, T7 *, void *, long, T6 *, int *)	63,4293s	8,864 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,4293s	3,584 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,4293s	3,040 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,4293s	3,200 μs	-	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)	63,4293s	2,880 μs	-	GPU 0	Stream 7""",
    'torch_csr': """Name	Start	Duration	GPU	Context
volta_sgemm_128x64_tn	77,8321s	445,629 μs	GPU 0	Stream 7
void spmm_kernel<float, (ReductionType)0, true>(long const*, long const*, float const*, float const*, float*, long*, int, int, int, int)	77,8325s	2,750 ms	GPU 0	Stream 7
void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int, false>, OffsetCalculator<1, unsigned int, false>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int, false>, OffsetCalculator<1, unsigned int, false>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)	77,8353s	214,559 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::clamp_min_scalar_kernel_impl(at::TensorIterator&, c10::Scalar)::{lambda()#1}::operator()() const::{lambda()#8}::operator()() const::{lambda(float)#1}, at::detail::Array<char*, 2> >(int, at::native::(anonymous namespace)::clamp_min_scalar_kernel_impl(at::TensorIterator&, c10::Scalar)::{lambda()#1}::operator()() const::{lambda()#8}::operator()() const::{lambda(float)#1}, at::detail::Array<char*, 2>)	77,8355s	212,319 μs	GPU 0	Stream 7
void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*)	77,8357s	286,558 μs	GPU 0	Stream 7
void spmm_kernel<float, (ReductionType)0, true>(long const*, long const*, float const*, float const*, float*, long*, int, int, int, int)	77,836s	1,652 ms	GPU 0	Stream 7
void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int, false>, OffsetCalculator<1, unsigned int, false>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int, false>, OffsetCalculator<1, unsigned int, false>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)	77,8377s	69,696 μs	GPU 0	Stream 7
void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, true>(float*, float const*, int, int, int)	77,8377s	73,375 μs	GPU 0	Stream 7
void at::native::(anonymous namespace)::nll_loss_forward_reduce_cuda_kernel_2d<float, float, long>(float*, float*, float*, long*, float*, bool, int, int, int, long)	77,8378s	4,001 ms	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)	77,8418s	3,136 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)	77,8418s	44,544 μs	GPU 0	Stream 7
void at::native::(anonymous namespace)::nll_loss_backward_reduce_cuda_kernel_2d<float, long>(float*, float*, long*, float*, float*, bool, int, int, int, long)	77,8419s	2,370 ms	GPU 0	Stream 7
void (anonymous namespace)::softmax_warp_backward<float, float, float, 6, true>(float*, float const*, float const*, int, int, int)	77,8442s	98,719 μs	GPU 0	Stream 7
void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>)	77,8443s	62,783 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>)	77,8444s	3,776 μs	GPU 0	Stream 7
void at::native::(anonymous namespace)::indexSelectLargeIndex<float, long, unsigned int, 1, 1, -2, true>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<long, unsigned int>, int, int, unsigned int, unsigned int, long)	77,8444s	39,808 μs	GPU 0	Stream 7
void at::native::(anonymous namespace)::indexSelectLargeIndex<long, long, unsigned int, 1, 1, -2, true>(at::cuda::detail::TensorInfo<long, unsigned int>, at::cuda::detail::TensorInfo<long, unsigned int>, at::cuda::detail::TensorInfo<long, unsigned int>, int, int, unsigned int, unsigned int, long)	77,8444s	65,343 μs	GPU 0	Stream 7
void spmm_kernel<float, (ReductionType)0, true>(long const*, long const*, float const*, float const*, float*, long*, int, int, int, int)	77,8445s	805,242 μs	GPU 0	Stream 7
volta_sgemm_64x32_sliced1x4_nt	77,8453s	326,750 μs	GPU 0	Stream 7
void splitKreduce_kernel<float, float, float, float, true, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, void*, long, float*, int*)	77,8456s	7,872 μs	GPU 0	Stream 7
volta_sgemm_128x64_nn	77,8456s	179,039 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>)	77,8458s	3,712 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::threshold_kernel_impl<float>(at::TensorIteratorBase&, float, float)::{lambda(float, float)#1}>, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::threshold_kernel_impl<float>(at::TensorIteratorBase&, float, float)::{lambda(float, float)#1}>, at::detail::Array<char*, 3>)	77,8458s	309,085 μs	GPU 0	Stream 7
void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>)	77,8461s	213,438 μs	GPU 0	Stream 7
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float> >, at::detail::Array<char*, 3>)	77,8464s	3,553 μs	GPU 0	Stream 7
void at::native::(anonymous namespace)::indexSelectLargeIndex<float, long, unsigned int, 1, 1, -2, true>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<long, unsigned int>, int, int, unsigned int, unsigned int, long)	77,8464s	40,511 μs	GPU 0	Stream 7
void at::native::(anonymous namespace)::indexSelectLargeIndex<long, long, unsigned int, 1, 1, -2, true>(at::cuda::detail::TensorInfo<long, unsigned int>, at::cuda::detail::TensorInfo<long, unsigned int>, at::cuda::detail::TensorInfo<long, unsigned int>, int, int, unsigned int, unsigned int, long)	77,8464s	65,184 μs	GPU 0	Stream 7
void spmm_kernel<float, (ReductionType)0, true>(long const*, long const*, float const*, float const*, float*, long*, int, int, int, int)	77,8465s	1,604 ms	GPU 0	Stream 7
volta_sgemm_32x128_nt	77,8481s	435,165 μs	GPU 0	Stream 7
void splitKreduce_kernel<float, float, float, float, true, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, void*, long, float*, int*)	77,8485s	8,576 μs	GPU 0	Stream 7"""
}
