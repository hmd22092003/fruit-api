# Sử dụng Python 3.8 làm môi trường
FROM python:3.8

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Sao chép toàn bộ code vào container
COPY . /app

# Cài đặt thư viện cần thiết từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose cổng 5000 để chạy API
EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "app_api.py"]
