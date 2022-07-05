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


inline int32_t index(int32_t x, int32_t y, int32_t w) {
    return (y * w + x);
}


float *crop(const float *normImage, int32_t stride, int32_t x, int32_t y, int32_t w, int32_t h) {
    float *cropped = (float *) calloc((size_t) (w * h), sizeof(float));
    float v;
    int32_t k = 0, l = 0;
    for (int32_t j = y; j < y + h; j++) {
        k = 0;
        for (int32_t i = x; i < x + w; i++) {
            v = normImage[index(i, j, stride)];
            cropped[index(k, l, w)] = v;
            k++;
        }
        l++;
    }
    return cropped;
}

void copy(const float *src, int32_t sw, int32_t sh, float *&target, int32_t x, int32_t y, int32_t tw) {
    for (int j = 0; j < sh; j++) {
        for (int i = 0; i < sw; i++) {
            target[index(x + i, y + j, tw)] = src[index(i, j, sw)];
        }
    }
}


int main(void)
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
    std::cout << "device integrated: " << props.integrated << std::endl;
    std::cout << "device totalGlobalMem: " << props.totalGlobalMem << std::endl;
    std::cout << "device sharedMemPerBlock: " << props.sharedMemPerBlock << std::endl;
    std::cout << "device concurrentKernels: " << props.concurrentKernels << std::endl;


    // assign a device to the thread
    cudaSetDevice(0);


    // allocate memory on the device
    int *mw = new int[1];
    int *mh = new int[1];
    int *mu = new int[1];
    int *mv = new int[1];


    float *dev_image;
    cudaMalloc((void **) &dev_image, sz * sizeof(float));
    cudaMemcpy(dev_image, normImage, sz * sizeof(float), cudaMemcpyHostToDevice);

    float *dev_sumLines;
    cudaMalloc((void **) &dev_sumLines, w * 2 * sizeof(float));


    // reserve memory on device
    int *dev_w, *dev_h, *dev_u, *dev_v;
    cudaMalloc((void **) &dev_w, 1 * sizeof(int));
    cudaMalloc((void **) &dev_h, 1 * sizeof(int));
    cudaMalloc((void **) &dev_u, 1 * sizeof(int));
    cudaMalloc((void **) &dev_v, 1 * sizeof(int));

    // init values
    *mw = w;
    *mh = h;
    *mu = 0;
    *mv = 0;

    // copy the values to the device
    cudaMemcpy(dev_w, mw, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_h, mh, 1 * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_u, mu, 1 * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_v, mv, 1 * sizeof(int), cudaMemcpyHostToDevice);


    // run kernel (vulkan shader module)
    const int blocks = 256;
    const int threads = 256;

    const int stream_id = 0;



    //// copy device result to host
    float *sumLines = new float[w * 2];
    std::cout << "w*2=" << w * 2 << std::endl;

    float re, im;
    float *magnitudes = new float[sz];

    for (int v = 0; v < h; v++) {

        *mv = v;
        cudaMemcpy(dev_v, mv, 1 * sizeof(int), cudaMemcpyHostToDevice);

        for (int u = 0; u < w; u++) {
            std::complex<double> sum(0, 0);

            *mu = u;
            cudaMemcpy(dev_u, mu, 1 * sizeof(int), cudaMemcpyHostToDevice);

            vuda::launchKernel("./Dft_Vuda/sumImageLineKernel.spv", "main", stream_id, blocks, threads, dev_image, dev_sumLines, dev_w, dev_h, dev_u, dev_v, w);

            // copy device result to host
            cudaMemcpy(sumLines, dev_sumLines, w * 2 * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < w; i++) {
                re = sumLines[i * 2];
                im = sumLines[i * 2 + 1];
                std::complex<float> current(re, im);
                // std::cout << current << std::endl;
                sum += current;
            }

            //std::cout << u << " : " << "Current sum: " << sum << std::endl;
            float magnitude = (float) log(abs(sum));
            std::cout << "\r" << "(" << u << "," << v << ")" << " magnitude:" << magnitude << std::flush;
            magnitudes[index(u, v, w)] = magnitude;
        }
    }
    std::cout << std::endl;
    std::cout << "normImage[0] = " << normImage[0] << std::endl;
    std::cout << "magnitudes[0] = " << magnitudes[0] << std::endl;

    std::cout << "normImage[" << sz - 1 << "] = " << normImage[sz - 1] << std::endl;
    std::cout << "magnitudes[" << sz - 1 << "] = " << magnitudes[sz - 1] << std::endl;


    // normalize magnitudes
    std::vector<float> v(sz);
    memcpy(&v[0], magnitudes, sz * sizeof(float));
    auto minimum = std::min_element(std::begin(v), std::end(v));
    auto maximum = std::max_element(std::begin(v), std::end(v));
    std::cout << "min=" << *minimum << std::endl;
    std::cout << "max=" << *maximum << std::endl;
    for (int i = 0; i < sz; i++) {
        magnitudes[i] = (magnitudes[i] - *minimum) / (*maximum - *minimum);
        // cout << magnitudes[i] << "; ";
    }
    // cout << endl;

    float *cropUpperLeft = crop(magnitudes, w, 0, 0, w / 2, h / 2);
    float *cropUpperRight = crop(magnitudes, w, w / 2, 0, w / 2, h / 2);
    float *cropBottomLeft = crop(magnitudes, w, 0, h / 2, w / 2, h / 2);
    float *cropBottomRight = crop(magnitudes, w, w / 2, h / 2, w / 2, h / 2);



    float *rearranged = (float *) calloc(sz, sizeof(double));
    copy(cropUpperLeft, w / 2, h / 2, rearranged, w / 2, h / 2, w);
    copy(cropUpperRight, w / 2, h / 2, rearranged, 0, h / 2, w);
    copy(cropBottomLeft, w / 2, h / 2, rearranged, w / 2, 0, w);
    copy(cropBottomRight, w / 2, h / 2, rearranged, 0, 0, w);


    // create resulting dft image
    uint8_t *dftImage = (uint8_t *) calloc(sz, sizeof(uint8_t));
    for (int i = 0; i < sz; i++) {
        dftImage[i] = (uint8_t) round(magnitudes[i] * 255);
        //std::cout << (int32_t) dftImage[i] << "; ";
    }
    // cout << endl;
    stbi_write_png("magn.png", w, h, 1, dftImage, w);

    //// create the rearranged dft image
    uint8_t *dftRearranged = (uint8_t *) calloc(sz, sizeof(uint8_t));
    for (int i = 0; i < sz; i++) {
        dftRearranged[i] = (uint8_t) round(rearranged[i] * 255);
    }
    stbi_write_png("reamagn.png", w, h, 1, dftRearranged, w);


    // free memory on device
    cudaFree(dev_image);
    cudaFree(dev_sumLines);
    cudaFree(dev_w);
    cudaFree(dev_h);
    cudaFree(dev_u);
    cudaFree(dev_v);


    delete[] normImage;
    delete[] sumLines;
    delete[] mw;
    delete[] mh;
    delete[] mu;
    delete[] mv;
    delete[] magnitudes;

    free(cropUpperLeft);
    free(cropUpperRight);
    free(cropBottomLeft);
    free(cropBottomRight);
}