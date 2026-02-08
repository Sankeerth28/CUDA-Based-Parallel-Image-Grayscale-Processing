#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>

__global__ void rgb_to_gray(unsigned char* input, unsigned char* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    if (idx < size) {
        int i = idx * 3;
        output[idx] = (input[i] + input[i + 1] + input[i + 2]) / 3;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./image_filter <input_dir> <output_dir>\n";
        return 1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        int width = img.cols;
        int height = img.rows;

        unsigned char *d_input, *d_output;
        size_t input_size = width * height * 3;
        size_t output_size = width * height;

        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, output_size);

        cudaMemcpy(d_input, img.data, input_size, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;
        rgb_to_gray<<<blocks, threads>>>(d_input, d_output, width, height);

        cv::Mat gray(height, width, CV_8UC1);
        cudaMemcpy(gray.data, d_output, output_size, cudaMemcpyDeviceToHost);

        std::string out_path = output_dir + "/gray_" + entry.path().filename().string();
        cv::imwrite(out_path, gray);

        cudaFree(d_input);
        cudaFree(d_output);
    }

    std::cout << "CUDA processing completed.\n";
    return 0;
}
