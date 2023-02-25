#include "opencv2/opencv.hpp"
#include "types.h"

namespace utils {
namespace image {
void letterbox(const cv::Mat& src, cv::Mat& dst, const int target_height,
               const int target_width, types::LetterBoxInfo& scale_params);
}
namespace yolo {
void get_original_results(float* output,
                          std::vector<types::BndBox>& origin_results,
                          const unsigned int num_anchors,
                          unsigned int info_length, const float conf_thres,
                          const types::LetterBoxInfo& scale_params);
}
namespace base {
float compute_iou(types::BndBox box1, types::BndBox box2);
}
namespace nms {
void offset_nms(std::vector<types::BndBox>& input,
                std::vector<types::BndBox>& output, const float iou_thres);
}
namespace visualization {
void draw_boxes_inplace(cv::Mat& mat_inplace,
                        const std::vector<types::BndBox>& boxes);
}
}  // namespace utils