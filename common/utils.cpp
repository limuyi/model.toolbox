#include "utils.h"

void utils::image::letterbox(const cv::Mat& src, cv::Mat& dst,
                             const int target_height, const int target_width,
                             types::LetterBoxInfo& scale_params) {
  // LetterBox
  int img_width = static_cast<int>(src.cols);
  int img_height = static_cast<int>(src.rows);
  // scale ratio (new / old) new_shape(h,w)
  float w_r = (float)target_width / (float)img_width;
  float h_r = (float)target_height / (float)img_height;
  float r = std::min(w_r, h_r);
  // 计算等比例缩放后的图片尺寸
  int new_unpad_w = static_cast<int>((float)img_width * r);   // floor
  int new_unpad_h = static_cast<int>((float)img_height * r);  // floor
  // 计算padding信息
  int pad_w = target_width - new_unpad_w;   // >=0
  int pad_h = target_height - new_unpad_h;  // >=0
  // 左上位置补边宽高
  int dw = pad_w / 2;
  int dh = pad_h / 2;
  // 记录信息
  scale_params.ratio = r;
  scale_params.left_padding_width = dw;
  scale_params.top_padding_height = dh;
  // 输入处理
  std::cout << "src: " << src.cols << "," << src.rows << std::endl;
  cv::Mat unpadding_mat;
  resize(src, unpadding_mat, cv::Size(new_unpad_w, new_unpad_h),
         cv::InterpolationFlags::INTER_CUBIC);
  std::cout << "unpadding_mat: " << unpadding_mat.cols << ","
            << unpadding_mat.rows << std::endl;
  // 目标图
  cv::Mat padding_mat;
  // 补边操作 top,bottom,left,right
  cv::copyMakeBorder(unpadding_mat, padding_mat, dh, pad_h - dh, dw, pad_w - dw,
                     cv::BorderTypes::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  std::cout << "padding_mat: " << padding_mat.cols << "," << padding_mat.rows
            << std::endl;
  // 贴图背景
  cv::Mat img_rgb = padding_mat;
  cvtColor(padding_mat, img_rgb, cv::COLOR_BGR2RGB);
  // divided by 255转float
  // yolo默认标准化处理可以直接用convertTo的alpha参数处理
  img_rgb.convertTo(dst, CV_32F, 1.0 / 255);
  // // 标准化方法1
  // img_rgb.convertTo(dst, CV_32F);
  // float mean_val = 0.f;
  // float scale_val = 1.0 / 255.f;
  // dst = (dst - mean_val) * scale_val;
  // // 标准化方法2，逐通道进行标准化处理
  // float mean[] = {0.485f, 0.456f, 0.406f};
  // float std_val[] = {0.229f, 0.224f, 0.225f};
  // for (int i = 0; i < input_mat.channels(); i++)
  // {
  //   channels[i] -= mean[i];     // mean均值
  //   channels[i] /= std_val[i];  // std方差
  // }
}

void utils::yolo::get_original_results(
    float* output, std::vector<types::BndBox>& origin_results,
    const unsigned int num_anchors, unsigned int info_length,
    const float conf_thres, const types::LetterBoxInfo& scale_params) {
  float rate = scale_params.ratio;
  unsigned int dw = scale_params.left_padding_width;
  unsigned int dh = scale_params.top_padding_height;
  for (unsigned int i = 0; i < num_anchors; ++i) {
    // 每组数据的长度为 output_dims[2]
    // 4位位置坐标+1位目标置信度+num_classes位类别置信度
    float* ptr = output + i * info_length;
    // 目标置信度
    float obj_conf = ptr[4];
    // 根据置信度阈值进行第一次过滤
    // std::cout << "obj_conf[" << i << "]: " << obj_conf << std::endl;
    if (obj_conf < conf_thres) continue;
    /**
     * 从第5位开始，剩余部分为类别置信度
     * 寻找最大值下标
     * 取值 int max_num = *max_element(nums, nums + 8);
     * 取坐标 int max_num_index = max_element(nums, nums + 8) - nums;
     * 这里运算的时候因为需要-(ptr+5)，因为起始坐标位prt+5，所以抵消直接-ptr
     */
    unsigned int max_conf_index =
        std::max_element(ptr + 5, ptr + info_length) - ptr;
    // std::cout << "label_index: " << label_index << std::endl;
    // 获取类别置信度
    float cls_conf = ptr[max_conf_index];
    // 计算最终置信度
    float score = obj_conf * cls_conf;
    // 根据置信度阈值进行第二次过滤
    if (score < conf_thres) continue;
    // 两次过滤后就可以计算坐标了
    // yolo输出的坐标为中心点坐标+宽高信息
    float cx = ptr[0];
    float cy = ptr[1];
    float w = ptr[2];
    float h = ptr[3];
    // 转化为左上右下坐标并记录
    types::BndBox obj;
    obj.x1 = ((cx - w / 2.f) - (float)dw) / rate;
    obj.y1 = ((cy - h / 2.f) - (float)dh) / rate;
    obj.x2 = ((cx + w / 2.f) - (float)dw) / rate;
    obj.y2 = ((cy + h / 2.f) - (float)dh) / rate;
    obj.score = score;
    obj.label_index = max_conf_index - 5;
    // obj.label_name = labels[max_conf_index - 5];
    origin_results.push_back(obj);
  }
}

float utils::base::compute_iou(types::BndBox box1, types::BndBox box2) {
  float max_x = std::max(box1.x1, box2.x1);  // 找出左上角坐标哪个大
  float min_x = std::min(box1.x2, box2.x2);  // 找出右上角坐标哪个小
  float max_y = std::max(box1.y1, box2.y1);
  float min_y = std::min(box1.y2, box2.y2);
  if (min_x <= max_x || min_y <= max_y)  // 如果没有重叠
    return 0;
  float over_area = (min_x - max_x) * (min_y - max_y);  // 计算重叠面积
  float area_a = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  float area_b = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  float iou = over_area / (area_a + area_b - over_area);
  return iou;
}

// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
void utils::nms::offset_nms(std::vector<types::BndBox>& input,
                            std::vector<types::BndBox>& output,
                            const float iou_thres) {
  const unsigned int topk = 1000;
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::BndBox& a, const types::BndBox& b) {
              return a.score > b.score;
            });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);
  const float offset = 4096.f;
  /** Add offset according to classes.
   * That is, separate the boxes into categories, and each category performs its
   * own NMS operation. The same offset will be used for those predicted to be
   * of the same category. Therefore, the relative positions of boxes of the
   * same category will remain unchanged. Box of different classes will be
   * farther away after offset, because offsets are different. In this way, some
   * overlapping but different categories of entities are not filtered out by
   * the NMS. Very clever!
   */
  for (unsigned int i = 0; i < box_num; ++i) {
    input[i].x1 += static_cast<float>(input[i].label_index) * offset;
    input[i].y1 += static_cast<float>(input[i].label_index) * offset;
    input[i].x2 += static_cast<float>(input[i].label_index) * offset;
    input[i].y2 += static_cast<float>(input[i].label_index) * offset;
  }
  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    std::vector<types::BndBox> buf;
    buf.push_back(input[i]);
    merged[i] = 1;
    for (unsigned int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;
      float iou = utils::base::compute_iou(input[i], input[j]);
      if (iou > iou_thres) {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }
    output.push_back(buf[0]);
    // keep top k
    count += 1;
    if (count >= topk) break;
  }
  /** Substract offset.*/
  if (!output.empty()) {
    for (unsigned int i = 0; i < output.size(); ++i) {
      output[i].x1 -= static_cast<float>(output[i].label_index) * offset;
      output[i].y1 -= static_cast<float>(output[i].label_index) * offset;
      output[i].x2 -= static_cast<float>(output[i].label_index) * offset;
      output[i].y2 -= static_cast<float>(output[i].label_index) * offset;
    }
  }
}

void utils::visualization::draw_boxes_inplace(
    cv::Mat& mat_inplace, const std::vector<types::BndBox>& boxes) {
  if (boxes.empty()) return;
  for (const auto& box : boxes) {
    cv::Point lt(box.x1, box.y1), rb(box.x2, box.y2);
    cv::rectangle(mat_inplace, lt, rb, cv::Scalar(87, 161, 5), 2);
    std::string label_name(box.label_name);
    label_name = label_name + ":" + std::to_string(box.score).substr(0, 4);
    cv::Point text_lb(box.x1, box.y1 - 3);
    cv::putText(mat_inplace, label_name, text_lb, cv::FONT_HERSHEY_SIMPLEX,
                0.6f, cv::Scalar(0, 0, 0), 2);
  }
}