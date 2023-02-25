#include "yolov5.h"

void YoloV5::PreProcess(const cv::Mat& src_mat,
                        std::vector<float>& inputTensorValues,
                        types::LetterBoxInfo& scale_params) {
  cv::Mat input_mat;
  // letterbox处理
  utils::image::letterbox(src_mat, input_mat, target_height, target_width,
                          scale_params);
  // 分离通道进行HWC->CHW
  cv::Mat channels[3];
  cv::split(input_mat, channels);
  for (int i = 0; i < input_mat.channels(); i++)  // HWC->CHW
  {
    std::vector<float> data = std::vector<float>(
        channels[i].reshape(1, input_mat.cols * input_mat.rows));
    inputTensorValues.insert(inputTensorValues.end(), data.begin(), data.end());
  }
}

void YoloV5::PostProcess(float* output,
                         std::vector<types::BndBox>& detection_boxes,
                         const types::LetterBoxInfo& scale_params) {
  // 过滤模型原始输出
  std::vector<types::BndBox> origin_results;
  utils::yolo::get_original_results(
      output, origin_results, output_node_dims[0][1], output_node_dims[0][2],
      conf_thres, scale_params);
  std::cout << "origin_results.size(): " << origin_results.size() << std::endl;
  // nms操作
  utils::nms::offset_nms(origin_results, detection_boxes, iou_thres);
  std::cout << "detection_boxes.size(): " << detection_boxes.size()
            << std::endl;
  // 转化label显示名
  const unsigned int box_num = detection_boxes.size();
  for (unsigned int i = 0; i < box_num; ++i) {
    detection_boxes[i].label_name = labels[detection_boxes[i].label_index];
  }
}

void YoloV5::Predict(cv::Mat& imgSrc,
                     std::vector<types::BndBox>& detection_boxes) {
  // 前处理
  types::LetterBoxInfo scale_params;
  std::vector<float> inputTensorValues;
  PreProcess(imgSrc, inputTensorValues, scale_params);

  // 1. make input tensor
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_handler, inputTensorValues.data(), inputTensorValues.size(),
      input_node_dims.data(), input_node_dims.size());
  // 2. inference scores & boxes.
  clock_t startTime, endTime;  // 计算推理时间
  std::cout << "Start infer" << std::endl;
  startTime = clock();
  auto output_tensors =
      ort_session->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                       &input_tensor, 1, output_node_names.data(), 1);
  endTime = clock();
  std::cout << "The run time is:"
            << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s"
            << std::endl;

  // 输出处理
  // (1,n,85=5+80=cxcy+cwch+obj_conf+cls_conf)
  float* output = output_tensors[0].GetTensorMutableData<float>();

  // 后处理操作
  PostProcess(output, detection_boxes, scale_params);
}