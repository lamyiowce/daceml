// Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>
namespace dace {

namespace sparse {

static void CheckCusparseError(cusparseStatus_t const& status) {
  std::string error(cusparseGetErrorString(status));
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("cuSPARSE failed with error code: " + error);
  }
}

static cusparseHandle_t CreateCusparseHandle(int device) {
  if (cudaSetDevice(device) != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device.");
  }
  cusparseHandle_t handle;
  CheckCusparseError(cusparseCreate(&handle));
  return handle;
}

/**
 * CUsparse wrapper class for DaCe. Once constructed, the class can be used to
 * get or create a cuSPARSE library handle (cusparseHandle_t) for a given
 * GPU ID. The class is constructed when the cuSPARSE DaCe library is used.
 **/
class CusparseHandle {
 public:
  CusparseHandle() = default;
  CusparseHandle(CusparseHandle const&) = delete;

  cusparseHandle_t& Get(int device) {
    auto f = handles_.find(device);
    if (f == handles_.end()) {
      // Lazily construct new cusparse handle if the specified key does not
      // yet exist
      auto handle = CreateCusparseHandle(device);
      f = handles_.emplace(device, handle).first;
    }
    return f->second;
  }

  void* Buffer(int device, int stream, size_t size) {
    // Lazily allocate and inflate buffers separately for streams as needed.
    // Assumes the device is set already.
    // TODO: Check if the device is the same.
    long long const key = ((long long)device<<32) + stream;
    auto f = buffers_.find(key);
    if (f == buffers_.end()) {
        void* bufferPtr;
        cudaMalloc(&bufferPtr, size);
        f = buffers_.emplace(key, std::make_pair(bufferPtr, size)).first;
    }
    size_t current_size = f->second.second;
    void* bufferPtr = f->second.first;
    if (size > current_size) {
        cudaFree(bufferPtr);
        cudaMalloc(&bufferPtr, size);
        f->second = std::make_pair(bufferPtr, size);
    }
    return f->second.first;
  }

  ~CusparseHandle() {
    for (auto& h : handles_) {
      CheckCusparseError(cusparseDestroy(h.second));
    }
    for (auto& b : buffers_) {
      cudaFree(b.second.first);
    }
  }

  CusparseHandle& operator=(CusparseHandle const&) = delete;

  std::unordered_map<int, cusparseHandle_t> handles_;
  std::unordered_map<long long, std::pair<void*, size_t>> buffers_;
};

}  // namespace sparse

}  // namespace dace
