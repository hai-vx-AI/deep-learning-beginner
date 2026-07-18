# Football Distance

Ứng dụng phân tích và theo dõi đối tượng trong video bóng đá.

## Yêu cầu

- Windows 10 hoặc Windows 11
- Python 3.12 64-bit
- Git

Kiểm tra Python 3.12:

```powershell
py -3.12 --version
```

## Cài đặt

Mở PowerShell hoặc Terminal tại thư mục muốn lưu dự án.

### 1. Clone mã nguồn

```powershell
git clone https://github.com/hai-vx-AI/football_distance.git
```

Sau khi clone, giữ nguyên Terminal tại thư mục hiện tại. Không di chuyển vào thư mục `football_distance`.

Cấu trúc thư mục sẽ có dạng:

```text
thu_muc_hien_tai/
└── football_distance/
```

### 2. Tạo môi trường ảo Python 3.12

```powershell
py -3.12 -m venv .venv
```

Sau bước này, cấu trúc thư mục sẽ là:

```text
thu_muc_hien_tai/
├── .venv/
└── football_distance/
```

### 3. Cài đặt thư viện

Không cần kích hoạt môi trường ảo. Chạy trực tiếp Python trong `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Cài đặt các thư viện của dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\football_distance\requirements.txt
```

### 4. Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe .\football_distance\app.py
```

## Chạy lại ứng dụng

Ở những lần sử dụng tiếp theo, mở Terminal tại thư mục chứa `.venv` và `football_distance`, sau đó chạy:

```powershell
.\.venv\Scripts\python.exe .\football_distance\app.py
```

## Mở dự án bằng Visual Studio Code

Mở thư mục cha chứa cả `.venv` và `football_distance`:

```powershell
code .
```

Visual Studio Code sẽ sử dụng môi trường:

```text
.venv\Scripts\python.exe
```

Nếu VS Code chưa tự nhận diện, chọn:

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ .venv\Scripts\python.exe
```
