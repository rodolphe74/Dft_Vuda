#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define _ITERATOR_DEBUG_LEVEL 0
#pragma warning( disable : 6011)
#pragma warning( disable : 4996)
#pragma warning( disable : 6255)
#pragma warning( disable : 6385)
#pragma warning( disable : 4834)
#endif // _MS


#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED

#include <vuda_runtime.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <complex>
#include <cmath>
#include <ostream>
#include <algorithm>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


void normalize(unsigned char *image, float *&normImage, int32_t size) {
    std::cout << "| mem size:" << size << std::endl;
    normImage = (float *) calloc(size, sizeof(float));
    for (int32_t i = 0; i < size; i++) {
        normImage[i] = image[i] / 1.0f;
    }
}


int _main(void)
{
    // load image
    int w, h, c;
    unsigned char *data = stbi_load("img/coin.png", &w, &h, &c, 1);
    size_t sz = (size_t) (w * h);

    std::cout << "Size:" << w << "*" << h << std::endl;
    std::cout << "Comp:" << c << std::endl;

    float *normImage;
    normalize(data, normImage, (int32_t) sz);

    int count;
    cudaGetDeviceCount(&count);
    std::cout << "devices count: " << count << std::endl;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "device name: " << props.name << std::endl;
    std::cout << "device maxSurface1D: " << props.maxSurface1D << std::endl;
    std::cout << "device computeMode: " << props.computeMode << std::endl;
    std::cout << "device integrated: " << props.integrated  << std::endl;
    std::cout << "device totalGlobalMem: " << props.totalGlobalMem << std::endl;
    std::cout << "device sharedMemPerBlock: " << props.sharedMemPerBlock << std::endl;
    std::cout << "device concurrentKernels: " << props.concurrentKernels << std::endl;


    // assign a device to the thread
    cudaSetDevice(0);
    // allocate memory on the device


    const int N = 1000;

    //int *a = new int[N];
    //int *b = new int[N];
    //int *m = new int[N];

    int *mx = new int[1];
    int *mw = new int[1];
    int *mh = new int[1];
    int *mu = new int[1];
    int *mv = new int[1];

    //for (int i = 0; i < N; ++i)
    //{
    //    a[i] = -i;
    //    b[i] = i * i;
    //}

    float *dev_image;
    cudaMalloc((void **) &dev_image, sz * sizeof(float));
    cudaMemcpy(dev_image, normImage, sz * sizeof(float), cudaMemcpyHostToDevice);

    float *dev_sumCols;
    cudaMalloc((void **) &dev_sumCols, h*2 * sizeof(float));


    // reserve memory on device
    int *dev_x, *dev_w, *dev_h, *dev_u, *dev_v;
    cudaMalloc((void **) &dev_x, 1 * sizeof(int));
    cudaMalloc((void **) &dev_w, 1 * sizeof(int));
    cudaMalloc((void **) &dev_h, 1 * sizeof(int));
    cudaMalloc((void **) &dev_u, 1 * sizeof(int));
    cudaMalloc((void **) &dev_v, 1 * sizeof(int));

    // init values
    /**mx = 0;*/
    *mw = w;
    *mh = h;
    *mu = 0;
    *mv = 0;

    // copy the values to the device
    cudaMemcpy(dev_x, mx, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_w, mw, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_h, mh, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_u, mu, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, mv, 1 * sizeof(int), cudaMemcpyHostToDevice);
    

    // run kernel (vulkan shader module)
    const int blocks = 256;
    const int threads = 256;

    const int stream_id = 0;


    //vuda::launchKernel("./Dft_Vuda/sumColsKernel.spv", "main", stream_id, blocks, threads, dev_image, dev_sumCols, dev_x, dev_w, dev_h, dev_u, dev_v, h);

    //// copy device result to host
    float *sumCols = new float[h*2];
    //cudaMemcpy(sumCols, dev_sumCols, h * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "h*2=" << h * 2 << std::endl;

    float re, im;

    std::complex<float> sum(0, 0);

    for (int u = 0; u < w; u++) {
        std::complex<double> sum(0, 0);

        *mu = u;
        cudaMemcpy(dev_u, mu, 1 * sizeof(int), cudaMemcpyHostToDevice);

        for (int x = 0; x < w; x++) {


            *mx = x;
            cudaMemcpy(dev_x, mx, 1 * sizeof(int), cudaMemcpyHostToDevice);
            std::complex<float> currentSum(0, 0);

            vuda::launchKernel("./Dft_Vuda/sumColsKernel.spv", "main", stream_id, blocks, threads, dev_image, dev_sumCols, dev_x, dev_w, dev_h, dev_u, dev_v, h);

            // copy device result to host
            cudaMemcpy(sumCols, dev_sumCols, h * 2 * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < h; i++) {
                re = sumCols[i * 2];
                im = sumCols[i * 2 + 1];
                std::complex<float> current(re, im);
                currentSum += current;
            }
            //std::cout << "  currentSum:" << currentSum << std::endl;
            sum += currentSum;
        }

        std::cout << u << " : " << "Current sum: " << sum << std::endl;
    }
    
    
    

    // free memory on device
    cudaFree(dev_image);
    cudaFree(dev_sumCols);
    cudaFree(dev_x);
    cudaFree(dev_w);
    cudaFree(dev_h);
    cudaFree(dev_u);
    cudaFree(dev_v);


    delete[] normImage;
    delete[] sumCols;
    delete[] mx;
    delete[] mw;
    delete[] mh;
    delete[] mu;
    delete[] mv;
}