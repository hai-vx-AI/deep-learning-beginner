# Football Distance

Ứng dụng phân tích và theo dõi cầu thủ, bóng và thời gian kiểm soát bóng trong video bóng đá.

## Yêu cầu

- Windows 10 hoặc Windows 11
- Python 3.12 64-bit
- Git

Kiểm tra Python 3.12:

```powershell
py -3.12 --version
```

Kết quả mong đợi:

```text
Python 3.12.x
```

## Tải trọng số mô hình

Trọng số mô hình không được lưu trực tiếp trên GitHub do kích thước file lớn.

Tải các file trọng số tại:

[Google Drive – Football Distance Weights](https://drive.google.com/drive/folders/1__dHKc2UMFts60foQxSASFtaZlRdTPwd?usp=sharing)

Sau khi tải, đặt các file trọng số vào thư mục:

```text
football_distance/
└── weights/
```

Giữ nguyên tên các file trọng số để chương trình có thể tìm đúng đường dẫn.

Cấu trúc dự án sau khi thêm trọng số:

```text
football_distance/
├── weights/
│   ├── ...
│   └── ...
├── football_tracking_station/
├── ui/
├── app.py
└── requirements.txt
```

## Cài đặt

Mở PowerShell hoặc Terminal tại thư mục muốn lưu dự án.

### 1. Clone mã nguồn

```powershell
git clone https://github.com/hai-vx-AI/football_distance.git
```

Sau khi clone, giữ nguyên Terminal tại thư mục hiện tại. Không di chuyển vào thư mục `football_distance`.

Cấu trúc thư mục:

```text
thu_muc_hien_tai/
└── football_distance/
```

### 2. Tạo môi trường ảo

Khuyến nghị sử dụng Python 3.12:

```powershell
py -3.12 -m venv .venv
```

Sau bước này, cấu trúc thư mục sẽ là:

```text
thu_muc_hien_tai/
├── .venv/
└── football_distance/
```

Nếu máy không có lệnh `py -3.12`, kiểm tra phiên bản Python mặc định:

```powershell
python --version
```

Sau đó tạo môi trường bằng:

```powershell
python -m venv .venv
```

Hoặc trên một số máy Windows:

```powershell
py -m venv .venv
```

> Dự án được thiết kế và kiểm thử với Python 3.12. Các phiên bản Python khác có thể gặp vấn đề tương thích thư viện.

### 3. Cài đặt thư viện

Không cần kích hoạt môi trường ảo. Chạy trực tiếp Python trong `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Cài đặt các thư viện của dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\football_distance\requirements.txt
```

Chờ đến khi Terminal xuất hiện thông báo cài đặt thành công và trả lại dòng lệnh.

### 4. Chạy ứng dụng

Đảm bảo các file trọng số đã được đặt đúng trong:

```text
football_distance\weights\
```

Sau đó chạy:

```powershell
.\.venv\Scripts\python.exe .\football_distance\app.py
```

## Chạy lại ứng dụng

Ở những lần sử dụng tiếp theo, mở Terminal tại thư mục chứa cả `.venv` và `football_distance`, sau đó chạy:

```powershell
.\.venv\Scripts\python.exe .\football_distance\app.py
```

Không cần cài lại thư viện hoặc tạo lại môi trường ảo.

## Mở dự án bằng Visual Studio Code

Mở thư mục cha chứa cả `.venv` và `football_distance`:

```powershell
code .
```

Cấu trúc thư mục trong Visual Studio Code:

```text
thu_muc_hien_tai/
├── .venv/
└── football_distance/
```

Visual Studio Code nên sử dụng interpreter:

```text
.venv\Scripts\python.exe
```

Nếu Visual Studio Code chưa tự nhận diện:

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ Enter interpreter path
→ .venv\Scripts\python.exe
```

Có thể kiểm tra interpreter đang sử dụng bằng lệnh:

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```
