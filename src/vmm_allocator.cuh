#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include "index/common.cuh"

#define DRIVER_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char* msg; \
            cuGetErrorName(result, &msg); \
            std::cerr << "CUDA Driver Error: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA Driver Error"); \
        } \
    } while (0)

namespace cagra {

class DeviceBufferVMM {
public:
    DeviceBufferVMM(size_t max_capacity_bytes) : capacity_(max_capacity_bytes), current_size_(0), d_ptr_(0) {
        // 1. 初始化 Driver API
        cuInit(0);
        
        // 【新增关键步骤】设置 Runtime API 使用设备 1，并建立 Context
        // 如果不加这两行，cudaMemcpy 可能会尝试去操作设备 0，导致 invalid argument
        cudaSetDevice(CUDA_DEVICE_ID);
        cudaFree(0);      

        // 2. 获取粒度
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        
        // 【修改点 1】这里指定物理内存分配在 Device 1 上
        prop.location.id = CUDA_DEVICE_ID;
        
        DRIVER_CHECK(cuMemGetAllocationGranularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        // 3. 对齐容量
        capacity_ = ((capacity_ + granularity_ - 1) / granularity_) * granularity_;

        // 4. 保留虚拟地址范围
        DRIVER_CHECK(cuMemAddressReserve(&d_ptr_, capacity_, 0, 0, 0));
    }

    ~DeviceBufferVMM() {
        free();
    }

    void free() {
        if (d_ptr_) {
            if (current_size_ > 0) {
                DRIVER_CHECK(cuMemUnmap(d_ptr_, current_size_));
                for (auto handle : handles_) {
                    DRIVER_CHECK(cuMemRelease(handle));
                }
                handles_.clear();
            }
            DRIVER_CHECK(cuMemAddressFree(d_ptr_, capacity_));
            d_ptr_ = 0;
        }
    }

    // 调整物理内存大小
    void resize(size_t new_size) {
        if (new_size <= current_size_) return;
        if (new_size > capacity_) {
            throw std::runtime_error("VMM OOM: Request exceeds reserved capacity");
        }

        size_t aligned_new = ((new_size + granularity_ - 1) / granularity_) * granularity_;
        size_t size_diff = aligned_new - current_size_;

        if (size_diff == 0) return;

        // 1. 分配物理内存
        CUmemGenericAllocationHandle handle;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        
        // 【修改点 2】这里指定新增的物理内存也在 Device 1
        prop.location.id = CUDA_DEVICE_ID;

        DRIVER_CHECK(cuMemCreate(&handle, size_diff, &prop, 0));
        handles_.push_back(handle);

        // 2. 映射
        DRIVER_CHECK(cuMemMap(d_ptr_ + current_size_, size_diff, 0, handle, 0));

        // 3. 设置访问权限
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        
        // 【修改点 3】这里告诉 GPU，Device 1 拥有读写这块内存的权限
        accessDesc.location.id = CUDA_DEVICE_ID;
        
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        DRIVER_CHECK(cuMemSetAccess(d_ptr_ + current_size_, size_diff, &accessDesc, 1));

        current_size_ = aligned_new;
    }

    void* data() const { return (void*)d_ptr_; }
    size_t size() const { return current_size_; }

private:
    CUdeviceptr d_ptr_;
    size_t capacity_;
    size_t current_size_;
    size_t granularity_;
    std::vector<CUmemGenericAllocationHandle> handles_;
};

} // namespace cagra