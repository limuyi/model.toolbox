#include "utils.h"
#include "ort_handler.h"

class YoloV5 : public core::BasicOrtHandler {
 public:
  explicit YoloV5(const std::string& _onnx_path, unsigned int _num_threads = 1)
      : BasicOrtHandler(_onnx_path, _num_threads){};

  ~YoloV5() override = default;

 public:
  void Predict(cv::Mat& imgSrc, std::vector<types::BndBox>& detection_boxes);

 private:
  void PreProcess(const cv::Mat& src_mat, std::vector<float>& inputTensorValues,
                  types::LetterBoxInfo& scale_params);
  void PostProcess(float* output, std::vector<types::BndBox>& detection_boxes,
                   const types::LetterBoxInfo& scale_params);

 private:
  const unsigned int target_width = 640;
  const unsigned int target_height = 640;
  const float conf_thres = 0.25f;
  const float iou_thres = 0.45f;

  const std::vector<std::string> labels = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
};
