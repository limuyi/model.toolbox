namespace types {
// letterbox操作信息
typedef struct {
  float ratio;              // 缩放比例
  int left_padding_width;   // 左边padding宽度
  int top_padding_height;  // 上边padding高度
} LetterBoxInfo;
// 目标检测结果信息
typedef struct {
  float x1;                  // left
  float y1;                  // top
  float x2;                  // right
  float y2;                  // bottom
  float score;               // 置信度
  unsigned int label_index;  // 原始标签
  std::string label_name;    // 标签名称
} BndBox;

}  // namespace types