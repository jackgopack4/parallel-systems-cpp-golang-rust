/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Helper functions for CUDA Driver API error handling (make sure that CUDA_H is included in your projects)
#ifndef HELPER_CUDA_DRVAPI_H
#define HELPER_CUDA_DRVAPI_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_string.h>
#include <drvapi_error_string.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// add a level of protection to the CUDA Samples, let's force samples to explicitly include CUDA.H
#ifdef  __cuda_cuda_h__
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

#ifdef getLastCudaDrvErrorMsg
#undef getLastCudaDrvErrorMsg
#endif

#define getLastCudaDrvErrorMsg(msg)           __getLastCudaDrvErrorMsg  (msg, __FILE__, __LINE__)

inline void __getLastCudaDrvErrorMsg(const char *msg, const char *file, const int line)
{
    CUresult err = cuCtxSynchronize();

    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "getLastCudaDrvErrorMsg -> %s", msg);
        fprintf(stderr, "getLastCudaDrvErrorMsg -> cuCtxSynchronize API error = %04d \"%s\" in file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error_result = cuDeviceGetAttribute(attribute, device_attribute, device);

    if (error_result != CUDA_SUCCESS)
    {
        printf("cuDeviceGetAttribute returned %d\n-> %s\n", (int)error_result, getCudaDrvErrorString(error_result));
        exit(EXIT_SUCCESS);
    }
}
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2CoresDRV(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x30, 192},
        { 0x32, 192},
        { 0x35, 192},
        { 0x37, 192},
        { 0x50, 128},
        { 0x52, 128},
        { 0x53, 128},
        { 0x60, 64 },
        { 0x61, 128},
        { 0x62, 128},
        { 0x70, 64 },
        { 0x72, 64 },
        { 0x75, 64 },
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

#ifdef __cuda_cuda_h__
// General GPU Device CUDA Initialization
inline int gpuDeviceInitDRV(int ARGC, const char **ARGV)
{
    int cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0, __CUDA_API_VERSION);

    if (CUDA_SUCCESS == err)
    {
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, (const char **) ARGV, "device=");

    if (dev < 0)
    {
        dev = 0;
    }

    if (dev > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> cudaDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
        fprintf(stderr, "\n");
        return -dev;
    }

    checkCudaErrors(cuDeviceGet(&cuDevice, dev));
    char name[100];
    cuDeviceGetName(name, 100, cuDevice);

    int computeMode;
    getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);

    if (computeMode == CU_COMPUTEMODE_PROHIBITED)
    {
        fprintf(stderr, "Error: device is running in <CU_COMPUTEMODE_PROHIBITED>, no threads can use this CUDA Device.\n");
        return -1;
    }

    if (checkCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == false)
    {
        printf("gpuDeviceInitDRV() Using CUDA Device [%d]: %s\n", dev, name);
    }

    return dev;
}

// This function returns the best GPU based on performance
inline int gpuGetMaxGflopsDeviceIdDRV()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count        = 0, sm_per_multiproc = 0;
    int max_compute_perf    = 0, best_SM_arch     = 0;
    int major = 0, minor = 0   , multiProcessorCount, clockRate;
    int devices_prohibited = 0;

    cuInit(0, __CUDA_API_VERSION);
    checkCudaErrors(cuDeviceGetCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceIdDRV error: no devices supporting CUDA\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major > 0 && major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, major);
        }

        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        checkCudaErrors(cuDeviceGetAttribute(&multiProcessorCount,
                                             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                             current_device));
        checkCudaErrors(cuDeviceGetAttribute(&clockRate,
                                             CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                             current_device));
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        int computeMode;
        getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, current_device);

        if (computeMode != CU_COMPUTEMODE_PROHIBITED)
        {
            if (major == 9999 && minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ConvertSMVer2CoresDRV(major, minor);
            }

            int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;

            if (compute_perf  > max_compute_perf)
            {
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
        }
        else
        {
            devices_prohibited++;
        }

        ++current_device;
    }

    if (devices_prohibited == device_count)
    {    
        fprintf(stderr, "gpuGetMaxGflopsDeviceIdDRV error: all devices have compute mode prohibited.\n");
        exit(EXIT_FAILURE);
    }    

    return max_perf_device;
}

// General initialization call to pick the best CUDA Device
inline CUdevice findCudaDeviceDRV(int argc, const char **argv)
{
    CUdevice cuDevice;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = gpuDeviceInitDRV(argc, argv);

        if (devID < 0)
        {
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        char name[100];
        devID = gpuGetMaxGflopsDeviceIdDRV();
        checkCudaErrors(cuDeviceGet(&cuDevice, devID));
        cuDeviceGetName(name, 100, cuDevice);
        printf("> Using CUDA Device [%d]: %s\n", devID, name);
    }

    cuDeviceGet(&cuDevice, devID);

    return cuDevice;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilitiesDRV(int major_version, int minor_version, int devID)
{
    CUdevice cuDevice;
    char name[256];
    int major = 0, minor = 0;

    checkCudaErrors(cuDeviceGet(&cuDevice, devID));
    checkCudaErrors(cuDeviceGetName(name, 100, cuDevice));
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, devID));

    if ((major > major_version) ||
        (major == major_version && minor >= minor_version))
    {
        printf("> Device %d: <%16s >, Compute SM %d.%d detected\n", devID, name, major, minor);
        return true;
    }
    else
    {
        printf("No GPU device was found that can support CUDA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}
#endif

// end of CUDA Helper Functions

#endif
