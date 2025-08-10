#include <NvInfer.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
  std::string log_name = "unnamed";
  if (argc == 2) {
    log_name = argv[1];
  }

  if (std::filesystem::exists("logs/" + log_name)) {
    std::filesystem::remove_all("logs/" + log_name);
  }
  std::filesystem::create_directories("logs/" + log_name);

  rs2::pipeline pipe;
  pipe.start();

  int idx = 0;
  while (true) {
    rs2::frameset frames = pipe.wait_for_frames();
    rs2::depth_frame depth = frames.get_depth_frame();
    rs2::video_frame color = frames.get_color_frame();

    cv::Mat depth_mat =
        cv::Mat(480, 640, CV_16UC1, const_cast<void *>(depth.get_data()));
    cv::Mat color_mat =
        cv::Mat(480, 640, CV_8UC3, const_cast<void *>(color.get_data()));
    cv::cvtColor(color_mat, color_mat, cv::COLOR_RGB2BGR);

    const short *data = static_cast<const short *>(depth.get_data());

    std::string color_filename =
        "logs/" + log_name + "/" + std::to_string(idx) + "_color.png";
    cv::imwrite(color_filename, color_mat);

    std::string depth_filename =
        "logs/" + log_name + "/" + std::to_string(idx) + "_depth.png";
    cv::imwrite(depth_filename, depth_mat);

    cv::imshow("depth", depth_mat);
    cv::imshow("color", color_mat);
    cv::waitKey(1);
    idx += 1;
  }

  return 0;
}
