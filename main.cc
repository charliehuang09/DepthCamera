#include <NvInfer.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <librealsense2/h/rs_frame.h>
#include <librealsense2/hpp/rs_frame.hpp>
#include <librealsense2/hpp/rs_internal.hpp>
#include <librealsense2/rs.h>
#include <librealsense2/rs.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

constexpr float fov = 85.2;
constexpr float img_width = 640;
constexpr float degree_to_radian(float angle) { return angle * M_PI / 180.0; }
constexpr float radian_to_degree(float radian) { return radian * 180.0 / M_PI; }
constexpr float focal_length() {
  return (img_width / 2) / std::tan(degree_to_radian(fov));
}

float calculate_angle(size_t pixel) {
  std::cout << "Got " << pixel << std::endl;
  if (pixel == img_width / 2) {
    return 0;
  }

  if (pixel > img_width / 2) {
    pixel -= img_width / 2;
    return radian_to_degree(std::atan(pixel / focal_length()));
  }
  if (pixel < img_width / 2) {
    pixel = img_width / 2 - pixel;
    return -radian_to_degree(std::atan(pixel / focal_length()));
  }
  return -1;
}

typedef struct Model {
  IRuntime *runtime = nullptr;
  ICudaEngine *engine = nullptr;
  IExecutionContext *context = nullptr;
  cudaStream_t inferenceCudaStream;
  size_t output_size = 1;
} model_t;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
};

std::vector<char> loadEngineFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("Engine file not found");
  return std::vector<char>((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
}

// HWC -> CHW
// https://github.com/cyrusbehr/tensorrt-cpp-api/blob/f93f973cd03f1caa710c60d60c0d6feb9ed79e4a/src/engine.h
cv::cuda::GpuMat preprocess(cv::Mat img) {
  cv::cuda::GpuMat img_gpu;
  img_gpu.upload(img);
  cv::cuda::GpuMat gpu_dst(1, img_gpu.rows * img_gpu.cols * 1, CV_8UC3);

  size_t width = img_gpu.cols * img_gpu.rows;

  std::vector<cv::cuda::GpuMat> input_channels{
      cv::cuda::GpuMat(img_gpu.rows, img_gpu.cols, CV_8U,
                       &(gpu_dst.ptr()[width * 0])),
      cv::cuda::GpuMat(img_gpu.rows, img_gpu.cols, CV_8U,
                       &(gpu_dst.ptr()[width * 1])),
      cv::cuda::GpuMat(img_gpu.rows, img_gpu.cols, CV_8U,
                       &(gpu_dst.ptr()[width * 2]))};

  // std::vector<cv::cuda::GpuMat> input_channels{
  //     cv::cuda::GpuMat(img_gpu.rows, img_gpu.cols, CV_8U,
  //                      &(gpu_dst.ptr()[width * 2])),
  //     cv::cuda::GpuMat(img_gpu.rows, img_gpu.cols, CV_8U,
  //                      &(gpu_dst.ptr()[width * 1])),
  //     cv::cuda::GpuMat(img_gpu.rows, img_gpu.cols, CV_8U,
  //                      &(gpu_dst.ptr()[width * 1]))};

  cv::cuda::split(img_gpu, input_channels); // HWC -> CHW

  cv::cuda::GpuMat output;
  gpu_dst.convertTo(output, CV_32FC3, 1.f / 255.f);

  return output;
}

short get_depth(const short *depth_buffer, int tx, int ty, int bx, int by,
                size_t stride = 480) {
  int cx = (tx + bx) / 2;
  int cy = (ty + by) / 2;
  short depth = depth_buffer[cx * stride + cy];
  return depth;
}

int main() {
  Logger logger;
  bool status;

  // std::string engine_path = "models/yolo11n.engine";
  std::string engine_path = "models/game_peice.engine";
  auto engine_data = loadEngineFile(engine_path);

  IRuntime *runtime = createInferRuntime(logger);
  assert(runtime != nullptr);

  ICudaEngine *engine =
      runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
  assert(engine != nullptr);

  IExecutionContext *context = engine->createExecutionContext();
  assert(context != nullptr);
  rs2::colorizer color_map;
  rs2::rates_printer printer;
  rs2::pipeline pipe;

  pipe.start();

  while (true) {
    rs2::frameset frames = pipe.wait_for_frames();
    rs2_frame *frame_ = frames.get();

    rs2::depth_frame depth = frames.get_depth_frame();
    std::cout << "units: " << depth.get_units();
    std::cout << "depth at center " << depth.get_distance(320, 320);
    const short *depth_buffer = static_cast<const short *>(depth.get_data());
    // std::cout << "CENTER DEPTH: "
    //           << rs2_depth_frame_get_distance(
    //                  rs2_extract_frame(frames.get(), 0, nullptr), 320, 320,
    //                  nullptr);

    // std::cout << "depth dimensions " << depth.get_height() <<
    // depth.get_width()
    //           << "\n";

    rs2::video_frame color = frames.get_color_frame();
    const uint8_t *color_buffer =
        static_cast<const uint8_t *>(color.get_data());

    cv::Mat depth_mat =
        cv::Mat(640, 640, CV_8UC1, const_cast<void *>(depth.get_data()));
    cv::imwrite("depth.png", depth_mat);
    // std::cout << color.get_width() << " " << color.get_height() << "\n";

    uint8_t *padded_color_buffer = new uint8_t[640 * 640 * 3]();
    std::memcpy(static_cast<void *>(padded_color_buffer), color_buffer,
                color.get_width() * color.get_height() * 3 * sizeof(uint8_t));

    cv::Mat img =
        cv::Mat(640, 640, CV_8UC3, static_cast<void *>(padded_color_buffer));

    cv::cuda::GpuMat gpu_mat = preprocess(img);
    status = context->setTensorAddress(engine->getIOTensorName(0),
                                       (void *)gpu_mat.ptr<void>());
    Dims output_shape = engine->getTensorShape(engine->getIOTensorName(1));
    size_t output_size = 1;
    for (int i = 0; i < output_shape.nbDims; i++) {
      output_size *= output_shape.d[i];
    }

    void *output;
    cudaMalloc((void **)&output, sizeof(float) * output_size);

    status =
        context->setTensorAddress(engine->getIOTensorName(1), (void *)output);

    cudaStream_t inferenceCudaStream;
    cudaStreamCreate(&inferenceCudaStream);

    status = context->enqueueV3(inferenceCudaStream);

    cudaStreamSynchronize(inferenceCudaStream);

    std::vector<float> featureVector;
    featureVector.resize(output_size);
    cudaMemcpyAsync(featureVector.data(), static_cast<char *>(output),
                    output_size * sizeof(float), cudaMemcpyDeviceToHost,
                    inferenceCudaStream);

    cudaStreamSynchronize(inferenceCudaStream);

    std::cout << "\n";
    for (int i = 0; i < 10; i++) {
      if (featureVector[i * 6 + 5] == 1) {
        for (int j = 0; j < 6; j++) {
          // std::cout << featureVector[i * 6 + j] << " ";
        }

        cv::circle(
            img, cv::Point(featureVector[i * 6 + 0], featureVector[i * 6 + 1]),
            3, cv::Scalar(0, 255, 0),
            -1); // red dot
        cv::circle(
            img, cv::Point(featureVector[i * 6 + 2], featureVector[i * 6 + 3]),
            3, cv::Scalar(255, 0, 0),
            -1); // red dot
        cv::circle(
            img,
            cv::Point((featureVector[i * 6 + 0] + featureVector[i * 6 + 2]) / 2,
                      (featureVector[i * 6 + 1] + featureVector[i * 6 + 3]) /
                          2),
            3, cv::Scalar(0, 0, 0),
            -1); // red dot
        std::cout << "depth "
                  << get_depth(depth_buffer, featureVector[i * 6 + 0],
                               featureVector[i * 6 + 1],
                               featureVector[i * 6 + 2],
                               featureVector[i * 6 + 3])
                  << "\n";
        std::cout << "angle "
                  << calculate_angle(
                         (featureVector[i * 6 + 0] + featureVector[i * 6 + 2]) /
                         2)
                  << "\n";
      }
    }

    // std::cout << img.size << "\n";
    cv::imshow("img", img);
    cv::waitKey(1);
  }

  return 0;
}
