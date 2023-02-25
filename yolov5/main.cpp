#include "yolov5.h"

int main(int argc, char* argv[]) {
  std::string inputImagePath = argv[1];
  std::cout << "Input image path: " << inputImagePath << std::endl;
  std::string onnx_path =
      "/home/limuyi/code/model.toolbox/yolov5/model/yolov5s.onnx";
  cv::Mat img;
  img = cv::imread(inputImagePath);
  std::vector<types::BndBox> detection_boxes;
  YoloV5* yolov5 = new YoloV5(onnx_path);
  yolov5->Predict(img, detection_boxes);
  // 画图
  utils::visualization::draw_boxes_inplace(img, detection_boxes);
  cv::imwrite("/home/limuyi/code/model.toolbox/yolov5/out/res.jpg", img);

  return 0;
}