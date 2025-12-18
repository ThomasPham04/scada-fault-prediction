## 1. Giới Thiệu Dữ Liệu và Chiến Lược Tiền Xử Lý

### 1.1. Đặc Tính Bộ Dữ Liệu (Dataset Overview)
Dữ liệu sử dụng trong dự án là dữ liệu thực tế (real-world data) từ hệ thống SCADA của các trang trại gió đang vận hành. Dữ liệu bao gồm thông tin trạng thái tuabin (WT status), các sự kiện bất thường được dán nhãn (labeled anomalies) với thời gian bắt đầu/kết thúc và mô tả lỗi. Do tính chất bảo mật, dữ liệu đã được ẩn danh (anonymized) nhưng vẫn giữ tối đa thông tin hữu ích cho bài toán bảo trì dự báo (predictive maintenance).

**Các đặc điểm chính và ảnh hưởng đến bài toán:**

1.  **Cấu trúc dữ liệu:**
    *   **Quy mô:** Bao gồm 95 tập dữ liệu từ 3 trang trại gió (Wind Farm A, B, C) với 36 tuabin khác nhau, tổng cộng 89 năm dữ liệu. Tần suất lấy mẫu là **10 phút/lần** (10-minute resolution).
    *   **Đặc trưng (Features):** Số lượng đặc trưng lớn (86 features cho Wind Farm A, lên tới 957 cho Wind Farm C), bao gồm các giá trị trung bình (avg), tối thiểu (min), tối đa (max) và độ lệch chuẩn (std) của các cảm biến.
    *   **Ảnh hưởng:** Độ phân giải cao và số lượng biến lớn cung cấp thông tin chi tiết nhưng cũng tạo ra thách thức về nhiễu và độ phức tạp tính toán. Việc lựa chọn đặc trưng (Feature Selection) là bắt buộc.

2.  **Chất lượng và Ẩn danh:**
    *   **Dữ liệu thực tế:** Chứa các giá trị thiếu (missing values) thường được thay thế bằng số 0, và các nhãn trạng thái (status IDs) đôi khi không nhất quán.
    *   **Ẩn danh:** Tên cảm biến bị mã hóa (trừ Power, Reactive Power, Wind Speed) và thời gian thực bị dịch chuyển ngẫu nhiên (randomly shifted years).
    *   **Ảnh hưởng:** Chúng ta không thể dựa hoàn toàn vào tên cảm biến để hiểu ý nghĩa, mà cần kết hợp phân tích tương quan. Việc xử lý dữ liệu thiếu và lọc nhiễu (như loại bỏ các giá trị 0 "ảo") trở nên quan trọng hàng đầu trong bước **Làm sạch dữ liệu (Data Cleaning)**.

3.  **Hệ thống đánh giá CARE:**
    *   Bộ dữ liệu đi kèm đề xuất về chỉ số CARE (Coverage, Accuracy, Reliability, Earliness) để đánh giá mô hình toàn diện.
    *   **Ảnh hưởng:** Điều này định hướng mục tiêu của mô hình không chỉ là độ chính xác (Accuracy) đơn thuần mà còn phải phát hiện sớm (Earliness) và giảm báo động giả (Reliability), ảnh hưởng trực tiếp đến việc lựa chọn ngưỡng cắt (threshold) trong **Chiến lược chia tập dữ liệu (Data Splitting)**.

### 1.2. Tổng Quan về Chiến Lược Tiền Xử Lý (Preprocessing Strategy)

Trong bài toán phát hiện bất thường (Anomaly Detection) cho hệ thống SCADA tuabin gió, chất lượng của dữ liệu đầu vào đóng vai trò quyết định đến hiệu suất của mô hình. Khác với các mô hình phân loại (Classification) truyền thống, phương pháp **Normal Behavior Model (NBM)** mà nhóm em lựa chọn hoạt động dựa trên nguyên lý: *"Học hành vi vận hành bình thường của tuabin, từ đó phát hiện các sai lệch (reconstruction errors) lớn làm tín hiệu cảnh báo lỗi."*

Do đó, quy trình tiền xử lý dữ liệu được nhóm thiết kế chặt chẽ nhằm đảm bảo mô hình chỉ được học trên dữ liệu "sạch" và "bình thường" nhất, đồng thời giữ nguyên tính chất của dữ liệu lỗi trong tập kiểm tra để đánh giá chính xác khả năng phát hiện.

Quy trình nhóm thực hiện bao gồm 4 giai đoạn chính:
1.  **Làm sạch & Lọc dữ liệu (Data Cleaning & Filtering)**
2.  **Kỹ thuật đặc trưng (Feature Engineering)**
3.  **Lựa chọn đặc trưng (Feature Selection)**
4.  **Chiến lược chia tập dữ liệu (Data Splitting Methodology)**

---

## 2. Chi Tiết Các Bước Tiền Xử Lý

### 2.1. Làm Sạch và Lọc Dữ Liệu (Data Cleaning & Filtering)

Mục tiêu của bước này là loại bỏ nhiễu và các trạng thái vận hành không mong muốn để tạo ra tập dữ liệu huấn luyện (Training Set) đại diện chuẩn xác cho "trạng thái bình thường" (Normal Features).

Nhóm em đã áp dụng quy trình lọc nghiêm ngặt dựa trên nguyên tắc **Boolean AND** (thỏa mãn tất cả điều kiện) cho từng điểm dữ liệu trong tập huấn luyện:

*   **Bước 1: Lọc Trạng thái (Status Filter)**
    *   **Điều kiện:** `status_type_id == 0`
    *   **Ý nghĩa:** Chỉ giữ lại dữ liệu được đánh dấu là "Vận hành bình thường". Loại bỏ hoàn toàn các trạng thái dừng máy, bảo trì, hiệu chỉnh, hoặc lỗi đã biết để tránh làm nhiễu mô hình với các mẫu dữ liệu không chuẩn.

*   **Bước 2: Lọc Môi trường (Wind Speed Filter)**
    *   **Điều kiện:** `wind_speed_3_avg > 4.0 m/s` (Cut-in wind speed)
    *   **Ý nghĩa:** Loại bỏ các dữ liệu ở tốc độ gió thấp khi tuabin chưa phát điện ổn định. Tại vùng này, các cảm biến thường ghi lại tín hiệu nhiễu (noise) hoặc hành vi quay tự do không mang tính đại diện cho vật lý vận hành.

*   **Bước 3: Lọc Công suất (Power Filter)**
    *   **Điều kiện:** `power_29_avg > 0 kW`
    *   **Ý nghĩa:** Loại bỏ các trường hợp tuabin quay (idling) nhưng không hòa lưới/không tải. Hành vi nhiệt và rung động ở trạng thái không tải khác biệt lớn so với trạng thái có tải, do đó không phù hợp để làm mẫu huấn luyện cho NBM.

*   **Bước 4: Kiểm tra Độ dài Dữ liệu (Data Sufficiency Check)**
    *   **Điều kiện:** `len(data) > Window_Size (14 days)`
    *   **Ý nghĩa:** Sau khi áp dụng các bộ lọc trên, nếu chuỗi dữ liệu còn lại của một sự kiện quá ngắn (không đủ để tạo thành một cửa sổ mẫu hoàn chỉnh), sự kiện đó sẽ bị **loại bỏ hoàn toàn** khỏi tập huấn luyện. Điều này đảm bảo tính liên tục và chất lượng của chuỗi thời gian đầu vào.

*   **Bước 5: Xử lý Dữ liệu Thiếu (Missing & NaN Handling)**
    *   **Kiểm tra:** Sau quá trình lọc hàng (filtering), tập dữ liệu vẫn có thể chứa các giá trị trống (`NaN`) ở một số cột cảm biến (do lỗi đường truyền hoặc cảm biến hỏng).
    *   **Chiến lược Xử lý:** Nhóm áp dụng phương pháp lấp đầy 3 bước ưu tiên tính liên tục thời gian:
        1.  `Forward Fill (ffill)`: Ưu tiên lấy giá trị của thời điểm ngay trước đó ($t-1$) điền cho thời điểm hiện tại ($t$). Đây là phương pháp hợp lý nhất cho chuỗi thời gian vật lý vì nhiệt độ/áp suất có tính quán tính.
        2.  `Backward Fill (bfill)`: Đối với các điểm `NaN` ở đầu chuỗi (không có $t-1$), sử dụng giá trị ngay sau đó ($t+1$).
        3.  `Fill Zero`: Nếu vẫn còn sót lại (trường hợp cực hiếm khi toàn bộ cột bị lỗi), điền giá trị 0.
    *   **Ý nghĩa:** Đảm bảo input cho mô hình LSTM là một ma trận dày đặc (dense matrix) hoàn chỉnh, không gây lỗi tính toán NaN trong quá trình nhân ma trận trọng số.

**Kết quả:** Quy trình này hoạt động như một "cái phễu" (tunnel), loại bỏ khoảng 40-50% dữ liệu thô kém chất lượng, giữ lại tập dữ liệu tinh khiết (high fidelity) nhất để mô hình học được mối tương quan chuẩn xác giữa các biến.

### 2.2. Kỹ Thuật Đặc Trưng (Feature Engineering)

Dữ liệu thô từ SCADA cần được biến đổi để phù hợp với việc tính toán của mạng noron (LSTM). Nhóm đã thực hiện các xử lý sau:

#### a. Xử lý dữ liệu Góc (Cyclic Features)
Các cảm biến góc (như hướng gió, hướng Nacelle, góc Pitch) có giá trị từ 0° đến 360°. Về mặt số học, 0° và 360° rất xa nhau, nhưng về mặt vật lý chúng là như nhau.
*   **Vấn đề:** Nếu để nguyên, mô hình sẽ hiểu sai khoảng cách giữa 359° và 1°.
*   **Giải pháp:** Nhóm sử dụng phép biến đổi lượng giác để chuyển mỗi góc thành 2 thành phần tọa độ:
    $$x_{sin} = \sin(\theta)$$
    $$x_{cos} = \cos(\theta)$$
*   **Thực thi:** Áp dụng cho `sensor_1` (Wind direction), `sensor_2` (Nacelle position), `sensor_5` (Pitch angle), v.v.

#### b. Xử lý dữ liệu Bộ đếm (Cumulative Counters)
Các biến đếm tích lũy (như Tổng năng lượng phát - Wh, Tổng năng lượng phản kháng - VArh) tăng đơn điệu theo thời gian.
*   **Vấn đề:** Các giá trị này (Non-stationary) phụ thuộc vào thời điểm đo hơn là trạng thái máy, gây nhiễu cho việc học quy luật vận hành.
*   **Giải pháp:** Nhóm quyết định loại bỏ hoàn toàn các biến này khỏi tập dữ liệu đầu vào.


### 2.3. Lựa Chọn Đặc Trưng (Feature Selection)

Từ 86 trường dữ liệu ban đầu, nhóm đã chọn lọc ra 80 đặc trưng (sau khi feature engineering) có ý nghĩa vật lý cao nhất đối với việc phát hiện lỗi.

Các nhóm đặc trưng quan trọng được giữ lại bao gồm:
1.  **Nhóm Nhiệt độ (Temperature Sensors):** Quan trọng nhất để phát hiện quá nhiệt (overheating) - dấu hiệu sớm của ma sát, mòn vòng bi hoặc lỗi làm mát. (VD: Nhiệt độ dầu hộp số, nhiệt độ vòng bi máy phát).
2.  **Nhóm Rung động & Tốc độ (RPM & Vibration):** Phản ánh trạng thái cơ học của hệ thống truyền động (Drive train).
3.  **Nhóm Điện (Electrical):** Dòng điện, điện áp, tần số các pha, phản ánh sức khỏe của máy phát điện và bộ biến tần.
4.  **Nhóm Môi trường & Điều khiển (Wind & Control):** Tốc độ gió, công suất, góc pitch - đóng vai trò là các biến điều kiện (condition variables) để mô hình hiểu được bối cảnh vận hành.

---

## 3. Chiến Lược Chia Tập Dữ Liệu (Train/Val/Test Splitting Strategy)

Đây là bước cải tiến quan trọng nhất trong phương pháp luận của nhóm, tuân theo các nghiên cứu tiên tiến (dựa trên phương pháp của Care et al., 2021) để đảm bảo tính thực tế của việc đánh giá.

### 3.1. Tập Huấn Luyện (Training Set)
*   **Nguồn dữ liệu:** Lấy từ phần `train` của **TẤT CẢ 22 sự kiện** (bao gồm cả các sự kiện có lỗi và bình thường).
*   **Xử lý:** Áp dụng bộ lọc nghiêm ngặt (Status=0, Wind>4, Power>0) như đã mô tả ở mục 2.1.
*   **Ý nghĩa:** Việc gộp dữ liệu "bình thường" từ tất cả các tuabin giúp mô hình học được phân phối tổng quát (General Distribution) của chế độ vận hành lành mạnh, tránh bị hiện tượng quá khớp (Overfitting) vào một tuabin cụ thể.
*   **Kích thước:** Nhóm thu được khoảng 664,000 điểm dữ liệu, tạo ra ~7,800 chuỗi thời gian (training sequences).

### 3.2. Tập Xác Thực (Validation Set)
*   **Phương pháp:** Chia theo thời gian (Temporal Split).
*   **Thực thi:** Lấy 15% dữ liệu cuối cùng của mỗi chuỗi Training.
*   **Ý nghĩa:** Dùng để đánh giá chéo (Cross-validation) trong quá trình huấn luyện và quyết định thời điểm dừng sớm (Early Stopping) để tránh overfitting. Quan trọng hơn, tập này được nhóm dùng để thiết lập **Ngưỡng bất thường (Anomaly Threshold)**.

### 3.3. Tập Kiểm Tra (Test Set) - Quan trọng
*   **Nguồn dữ liệu:** Lấy từ phần `prediction` của các sự kiện.
*   **Xử lý:** **KHÔNG LỌC** bất kỳ trạng thái nào (Giữ nguyên status codes, wind speed thấp, v.v.).
*   **Ý nghĩa:** Đây là điểm mấu chốt mà nhóm muốn nhấn mạnh. Trong thực tế (Production), chúng ta không biết trước khi nào lỗi xảy ra. Do đó, tập Test phải phản ánh đúng dữ liệu thô mà hệ thống giám sát sẽ nhận được, bao gồm cả các giai đoạn bất ổn định và giai đoạn lỗi xảy ra.
*   **Mục tiêu:** Đánh giá khả năng của mô hình trong việc phân biệt giữa vận hành bình thường và giai đoạn sắp xảy ra lỗi (Pre-fault).

---

## 4. Chuẩn Hóa và Tạo Chuỗi (Normalization & Windowing)

### 4.1. Chuẩn Hóa (Normalization)
Nhóm sử dụng kỹ thuật `StandardScaler` (Z-score normalization):
$$z = \frac{x - \mu}{\sigma}$$

**Quy tắc vàng:** Scaler chỉ được `fit` (học tham số $\mu, \sigma$) trên tập **Training Data (Normal only)**.
*   Các tập Validation và Test sẽ được `transform` dựa trên tham số của tập Train.
*   Lý do: Đảm bảo rằng khi dữ liệu lỗi (Test) xuất hiện, chúng sẽ có giá trị sai lệch lớn so với phân phối chuẩn của tập Train, giúp khuếch đại tín hiệu lỗi cho mô hình dễ phát hiện.

### 4.2. Cửa Sổ Thời Gian (Sliding Window)
Để mô hình LSTM nắm bắt được phụ thuộc thời gian dài hạn (Long-term dependencies), dữ liệu được nhóm cắt thành các cửa sổ trượt:
*   **Window Size:** 14 ngày (tương đương 2,016 điểm dữ liệu 10-phút).
    *   *Cơ sở:* 14 ngày đủ dài để bao phủ các chu kỳ vận hành, thời tiết và chế độ nhiệt động lực học của tuabin.
*   **Stride (Bước trượt):** 12 giờ (72 điểm dữ liệu).
    *   *Mục đích:* Tăng cường số lượng mẫu huấn luyện và giảm thiểu sự trùng lặp dư thừa.

---

## 5. Kết Luận

Quy trình tiền xử lý này đã chuyển đổi dữ liệu SCADA thô sơ, nhiễu loạn thành một tập dữ liệu sạch, giàu thông tin và được cấu trúc khoa học. Chiến lược chia tập dữ liệu V2 (sử dụng tất cả 22 sự kiện cho training và test trên vùng prediction) đảm bảo tính khách quan và khả năng ứng dụng thực tế cao cho mô hình Normal Behavior Model. Đây là nền tảng vững chắc để nhóm xây dựng mô hình Deep Learning có độ tin cậy cao.
