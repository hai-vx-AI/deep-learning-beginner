# Football Distance

Ứng dụng phân tích và theo dõi đối tượng trong video bóng đá.

## Yêu cầu

- Windows 10 hoặc Windows 11
- Python 3.12 64-bit
- Git

Kiểm tra Python 3.12:

```bash
py -3.12 --version
```

## Cài đặt

### 1. Clone mã nguồn

```bash
git clone https://github.com/hai-vx-AI/football_distance.git
cd football_distance
```

### 2. Tạo môi trường ảo bằng Python 3.12

```bash
py -3.12 -m venv .venv
```

### 3. Kích hoạt môi trường ảo

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate
```

Nếu PowerShell chặn việc kích hoạt môi trường, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 4. Cài đặt thư viện

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 5. Chạy ứng dụng

```bash
python app.py
```

## Chạy lại ứng dụng

Tại thư mục dự án, kích hoạt môi trường:

```powershell
.venv\Scripts\Activate.ps1
```

Sau đó chạy:

```bash
python app.py
```
