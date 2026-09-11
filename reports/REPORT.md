# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** GPU (CUDA / Tesla T4)

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không (giữ nguyên toàn bộ cấu hình, tham số, ngưỡng kiểm thử và mã nguồn chuẩn của bài lab)

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  `{"class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915, "taxonomy_name": "ImageNet-1K"}` (thuộc sample `traffic`, coco_image_id: 210273).
- Record này mô tả toàn ảnh như thế nào?
  Tác vụ phân loại ảnh (image classification) gán một nhãn duy nhất ở cấp độ toàn bức ảnh (image-level prediction).
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Tập dữ liệu huấn luyện và người thiết kế taxonomy, cụ thể ở đây là taxonomy ImageNet-1K gồm 1.000 lớp danh mục chuẩn do cộng đồng ImageNet thiết lập.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    Giữ lại để đảm bảo cho cả máy tính và con người đều có thể hiểu được một cách tường minh vật thể muốn xem xét
  - `class_id` (468): Dành cho máy tính xử lý kỹ thuật số
  - `class_name` ("cab"): Dành cho con người hiểu trực quan ngữ nghĩa của nhãn.
  - `taxonomy_name` ("ImageNet-1K"): Cực kỳ quan trọng để đảm bảo tính toàn vẹn và nguồn gốc dữ liệu (data provenance).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Trong ảnh `traffic`, có rất nhiều chủ thể cùng xuất hiện (xe buýt, taxi, xe con, người đi bộ). Guideline cần quy định rõ ràng:
  - Tiêu chí lựa chọn nhãn đại diện chính: Ưu tiên đối tượng chiếm diện tích lớn nhất (dominant object), đối tượng nằm ở vùng trung tâm (center of focus), hay đối tượng ở tiền cảnh sắc nét nhất.
- Vì sao model score không phải ground truth?
  Model score (confidence score) chỉ là giá trị xác suất toán học (thường qua hàm Softmax) phản ánh mức độ tự tin của mạng nơ-ron dựa trên các trọng số đã được huấn luyện. Model score không phản ánh sự thật khách quan (ground truth) vì mô hình có thể tự tin rất cao (high confidence) vào một dự đoán sai do thiên kiến dữ liệu hoặc góc chụp lạ (có thể là hiện tượng calibration model không tốt), và ngược lại có thể tự tin thấp (low confidence) vào một đối tượng hiển nhiên đúng do nhiễu hạt hoặc ánh sáng yếu. Ground truth chỉ được thiết lập sau khi con người kiểm duyệt và xác nhận theo guideline chuẩn.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  `{"class_name": "bus", "score": 0.912558, "bbox_xyxy": [93.17,187.95,223.01,320.91], "bbox_width": 129.84, "bbox_height": 132.96}` (đối tượng xe buýt).
- Diễn giải vị trí box bằng lời:
  Hộp bao quanh xe buýt nằm trong tọa độ từ pixel (93.17, 187.95) đến (223.01, 320.91) với kích thước ảnh là chiều rộng 129.84 pixel và chiều cao 132.96 pixel.
- So sánh số prediction ở hai threshold:
  Trong notebook khi chạy thực nghiệm trên sample `kitchen`:
  - Tại ngưỡng score `0.20`: mô hình phát hiện 17 vật thể (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']`).
  - Tại ngưỡng score `0.35`: mô hình phát hiện 11 vật thể (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl']`).
  (Số lượng dự đoán giảm từ 17 xuống 11 khi tăng threshold từ 0.20 lên 0.35, loại bỏ các vật thể có điểm tin cậy thấp như spoon, potted plant, dining table, bottle).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Khi đặt threshold thấp (0.20): Độ bao phủ (recall) cao, phát hiện được nhiều vật thể nhỏ và mờ (chiếc bàn lớn, các thìa, chùm lá treo), nhưng kéo theo nhiều dự đoán sai/dương tính giả (false positives). Reviewer phải kiểm tra khối lượng dự đoán lớn hơn nhiều và tốn thời gian xóa/sửa các box rác.
  - Khi đặt threshold cao (0.35 hoặc 0.60): Độ chính xác (precision) của các box còn lại cao hơn, ít báo ảo giúp Reviewer duyệt nhanh hơn ở các box hiển thị, nhưng độ bao phủ giảm mạnh (bỏ sót nhiều vật thể thực tế / false negatives). Khi đó, người gán nhãn và Reviewer vẫn phải tự tay vẽ bổ sung thủ công toàn bộ các vật thể bị bỏ sót (như chiếc bàn `dining table` hay các dụng cụ nấu ăn).
- Đề xuất một quy tắc box chặt:
  "Bounding box phải ôm sát tuyệt đối (tightest bounding box) các pixel khả kiến của vật thể: cả 4 cạnh của hộp (trên, dưới, trái, phải) phải tiếp xúc trực tiếp với các điểm cực trị ngoài cùng của vật thể. Không được để lề trống (margin/padding) quá 2 pixel và tuyệt đối không cắt lẹm vào bất kỳ bộ phận khả kiến nào của vật thể."
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  Guideline và escalation cần quy định rõ:
  1. Với vật thể bị cắt bởi mép ảnh (truncated): Chỉ đóng khung phần hiển thị thực tế trong khung ảnh (visible part), chặn tại đường biên ảnh (không suy đoán vùng vô hình ngoài ảnh) và đánh dấu cờ `is_truncated = True`.
  2. Với vật thể bị che khuất một phần (occluded):
     - Nếu vật thể vẫn nhận biết được liên tục qua vật cản, vẽ 1 box bao phủ toàn bộ phạm vi của vật thể đó và đánh dấu cờ `is_occluded = True`.
     - Nếu vật cản chia vật thể thành 2 phần tách rời rõ rệt hoặc độ che khuất quá lớn (ví dụ trên 70-80% diện tích không thấy được), guideline cần quy định rõ là tách thành 2 box riêng, vẽ 1 box lớn gộp, hay bỏ qua không gán nhãn.
     - Nếu annotator không thể xác định chắc chắn vật thể bị che là gì, bắt buộc phải escalate cho QA Lead / Domain Expert.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  `{"instance_id": "traffic-001", "class_name": "bus", "score": 0.925745, "polygon_point_count": 120, "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ],
      [
        141.0,
        193.0
      ],
      [
        139.0,
        193.0
      ],
      [
        137.0,
        195.0
      ],
      [
        136.0,
        195.0
      ],
      [
        135.0,
        196.0
      ],
      [
        134.0,
        196.0
      ],
      [
        133.0,
        197.0
      ],
      [
        132.0,
        197.0
      ],
      [
        131.0,
        198.0
      ],
      [
        130.0,
        198.0
      ],
      [
        128.0,
        200.0
      ],
      [
        127.0,
        200.0
      ],
      [
        126.0,
        201.0
      ],
      [
        125.0,
        201.0
      ],
      [
        123.0,
        203.0
      ],
      [
        122.0,
        203.0
      ],
      [
        121.0,
        204.0
      ],
      [
        120.0,
        204.0
      ],
      [
        119.0,
        205.0
      ],
      [
        117.0,
        205.0
      ],
      [
        116.0,
        206.0
      ],
      [
        115.0,
        206.0
      ],
      [
        114.0,
        207.0
      ],
      [
        113.0,
        207.0
      ],
      [
        111.0,
        209.0
      ],
      [
        110.0,
        209.0
      ],
      [
        109.0,
        210.0
      ],
      [
        108.0,
        210.0
      ],
      [
        105.0,
        213.0
      ],
      [
        104.0,
        213.0
      ],
      [
        98.0,
        219.0
      ],
      [
        98.0,
        220.0
      ],
      [
        96.0,
        222.0
      ],
      [
        96.0,
        312.0
      ],
      [
        97.0,
        313.0
      ],
      [
        97.0,
        314.0
      ],
      [
        98.0,
        315.0
      ],
      [
        101.0,
        315.0
      ],
      [
        102.0,
        316.0
      ],
      [
        104.0,
        316.0
      ],
      [
        105.0,
        317.0
      ],
      [
        106.0,
        317.0
      ],
      [
        107.0,
        318.0
      ],
      [
        108.0,
        318.0
      ],
      [
        109.0,
        319.0
      ],
      [
        113.0,
        319.0
      ],
      [
        114.0,
        318.0
      ],
      [
        116.0,
        318.0
      ],
      [
        117.0,
        317.0
      ],
      [
        122.0,
        317.0
      ],
      [
        123.0,
        316.0
      ],
      [
        131.0,
        316.0
      ],
      [
        132.0,
        315.0
      ],
      [
        163.0,
        315.0
      ],
      [
        164.0,
        314.0
      ],
      [
        170.0,
        314.0
      ],
      [
        171.0,
        315.0
      ],
      [
        176.0,
        315.0
      ],
      [
        177.0,
        316.0
      ],
      [
        183.0,
        316.0
      ],
      [
        184.0,
        315.0
      ],
      [
        185.0,
        315.0
      ],
      [
        188.0,
        312.0
      ],
      [
        188.0,
        311.0
      ],
      [
        190.0,
        309.0
      ],
      [
        190.0,
        308.0
      ],
      [
        197.0,
        301.0
      ],
      [
        198.0,
        301.0
      ],
      [
        199.0,
        300.0
      ],
      [
        201.0,
        300.0
      ],
      [
        202.0,
        299.0
      ],
      [
        203.0,
        299.0
      ],
      [
        204.0,
        298.0
      ],
      [
        205.0,
        298.0
      ],
      [
        206.0,
        297.0
      ],
      [
        209.0,
        297.0
      ],
      [
        210.0,
        296.0
      ],
      [
        211.0,
        296.0
      ],
      [
        213.0,
        294.0
      ],
      [
        213.0,
        293.0
      ],
      [
        214.0,
        292.0
      ],
      [
        214.0,
        291.0
      ],
      [
        215.0,
        290.0
      ],
      [
        215.0,
        289.0
      ],
      [
        217.0,
        287.0
      ],
      [
        217.0,
        286.0
      ],
      [
        219.0,
        284.0
      ],
      [
        219.0,
        283.0
      ],
      [
        220.0,
        282.0
      ],
      [
        220.0,
        281.0
      ],
      [
        221.0,
        280.0
      ],
      [
        221.0,
        279.0
      ],
      [
        222.0,
        278.0
      ],
      [
        222.0,
        276.0
      ],
      [
        223.0,
        275.0
      ],
      [
        223.0,
        234.0
      ],
      [
        222.0,
        233.0
      ],
      [
        222.0,
        226.0
      ],
      [
        221.0,
        225.0
      ],
      [
        221.0,
        220.0
      ],
      [
        220.0,
        219.0
      ],
      [
        220.0,
        208.0
      ],
      [
        219.0,
        207.0
      ],
      [
        219.0,
        202.0
      ],
      [
        218.0,
        201.0
      ],
      [
        218.0,
        198.0
      ],
      [
        217.0,
        197.0
      ],
      [
        217.0,
        196.0
      ],
      [
        213.0,
        192.0
      ],
      [
        211.0,
        192.0
      ],
      [
        210.0,
        191.0
      ],
      [
        206.0,
        191.0
      ],
      [
        205.0,
        190.0
      ],
      [
        199.0,
        190.0
      ],
      [
        198.0,
        189.0
      ]
    ]` (đối tượng xe bus trong ).
- Polygon bổ sung chi tiết gì so với box?
  Polygon cung cấp thông tin ranh giới pixel chính xác (pixel-level mask/boundary) theo đúng hình dạng thực tế (contour, silhouette) của từng vật thể. Thay vì một hình chữ nhật bao quanh chứa lẫn cả các pixel nền (background) hoặc pixel của các vật thể khác xung quanh, polygon tách biệt hoàn toàn vật thể khỏi môi trường nền, thể hiện rõ các đường nét chi tiết như bờ vai, eo tạp dề, nếp gấp quần áo của người đầu bếp, hoặc đường gờ uốn lượn của khuôn nướng bánh.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` (ví dụ `kitchen-001`, `kitchen-002`,...) dùng để phân biệt và định danh duy nhất từng cá thể đối tượng độc lập trong phạm vi bức ảnh đó, cho phép phân tách giữa các đối tượng có cùng class_name.
  - `instance_id` KHÔNG PHẢI là `class_id` (mã danh mục ngữ nghĩa dùng chung cho toàn bộ lớp đối tượng), và cũng KHÔNG PHẢI là `tracking ID / re-identification ID` (mã định danh liên tục xuyên suốt qua nhiều khung hình video hay camera khác nhau).
- Đề xuất một quy tắc biên mask:
  "Đường biên của đa giác mặt nạ (polygon edge) phải đi sát đường ranh giới tự nhiên giữa vật thể và nền với sai số tối đa 1-2 pixel. Không bao gồm các pixel nền trống, bóng đổ (shadows) hoặc các vật thể lân cận. Đối với các đối tượng có cấu trúc rỗng bên trong (như quai ấm, lỗ hổng), phải tạo đường biên trong (interior ring/hole) để trừ phần nền ra."
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  Guideline và escalation cần quy định:
  1. Vùng mờ do chuyển động hoặc ánh sáng nhòe (motion blur): Quy định vẽ đường biên cắt ngang trung bình vùng chuyển tiếp gradient màu dựa trên cấu trúc vật lý của vật thể, không mở rộng polygon ôm lấy vệt nhòe.
  2. Điểm tiếp xúc giữa các vật thể chồng lấn (touching/overlapping): Ranh giới giữa hai instance phải được phân định dứt khoát, không để mask của hai cá thể đè chồng lấn lên nhau gây xung đột điểm ảnh (trừ khi hệ thống hỗ trợ cơ chế phân lớp depth/layer rõ ràng).
  3. Vật thể bị chia cắt thành nhiều mảng riêng biệt do bị che khuất: Quy định gán gộp các mảng thành một `multi-polygon` thuộc cùng một `instance_id` hay tách thành các instance độc lập. Trong các trường hợp ranh giới tiếp xúc quá phức tạp hoặc bị che khuất trên 50%, annotator cần escalate để reviewer hướng dẫn.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cấp ảnh (Image-level label): `class_id`, `class_name`, `taxonomy_name` | Ảnh `traffic` có nhiều loại xe (bus, car, cab, minibus) nhưng chỉ gán được 1 nhãn; model dự đoán `cab` với score 0.51, bỏ sót bối cảnh giao thông chung. | Quan sát toàn cảnh, đối chiếu guideline để xác định chủ thể chính thống trị (dominant subject) hoặc bối cảnh chung; gán đúng 1 nhãn theo taxonomy. | Kiểm tra nhãn được gán có tuân thủ quy tắc ưu tiên trong guideline không; kiểm tra tính nhất quán giữa các annotator với các ảnh phức tạp. |
| Phát hiện vật thể | Danh sách các hộp bao: `class_name`, `bbox_xyxy` (hoặc `xywh`), kèm cờ thuộc tính (`is_truncated`, `is_occluded`) | Ở ngưỡng 0.35 bị bỏ sót cái bàn (`dining table`), thìa (`spoon`), cây treo (`potted plant`); cánh tay ở góc trái bị nhận diện thành `person` hoàn chỉnh. | Rà soát toàn bộ ảnh để tìm mọi đối tượng trong taxonomy; vẽ bounding box ôm sát 4 cạnh ngoài cùng của vật thể; gán cờ che khuất/cắt mép. | Soi độ ôm khít (tightness) của 4 cạnh hộp; phát hiện các vật thể bị gán nhãn bỏ sót (false negatives) hoặc hộp vẽ dư thừa/trùng lặp (false positives). |
| Instance segmentation | Đa giác kín hoặc mặt nạ pixel cho từng cá thể: `instance_id`, `class_name`, `polygon_xy` | Mặt nạ bàn bị đứt khúc; ranh giới tiếp xúc giữa các bát/đĩa/khuôn trên bàn bị dính chùm hoặc lem ra ngoài; bó cây treo bị dính vào nền tường. | Dùng công cụ polygon/brush viền khít chu vi từng cá thể; khoét lỗ rỗng; gán đúng `instance_id` độc lập cho từng đối tượng cùng lớp. | Phóng to (zoom-in) kiểm tra độ chính xác cấp pixel ở viền mask; kiểm tra vùng tiếp xúc giữa các instance có bị chồng lấn hoặc bỏ sót chi tiết không. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Tuyệt đối không trích xuất, sao chép hoặc phát tán dữ liệu dự án ra bên ngoài hệ thống được cấp phép; không đưa thông tin nhận dạng cá nhân (PII như họ tên, khuôn mặt nhạy cảm, biển số xe, địa chỉ, MSSV, CCCD) vào nhãn gán hoặc báo cáo nộp bài; luôn tuân thủ giấy phép bản quyền nguồn ảnh mở (ghi nhận đầy đủ nguồn gốc và bản quyền trong IMAGE_ATTRIBUTION.md).
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Dừng ngay công việc trên dữ liệu đó và báo cáo trực tiếp cho Giảng viên / Mentor phụ trách lab (hoặc QA Lead / Project Manager) qua kênh hỗ trợ chính thức của khóa học, cung cấp mã định danh ảnh (image ID) và mô tả vắn tắt vấn đề mà không tự ý sao chép hay lan truyền ảnh đó.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
