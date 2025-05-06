import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import serial

# Khởi tạo scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Load mô hình từ file .h5
model = load_model("C:\\Users\\vttpa\\Downloads\\ankemne.h5")

# Kết nối với cổng serial
ser = serial.Serial('COM7', 9600)

# Kích thước cửa sổ dữ liệu
window_size = 10

# Đọc dữ liệu ban đầu từ cổng serial
input_data = []
print("Đang đọc dữ liệu từ cổng serial...")
valid_samples = 0

while valid_samples < window_size:
    line = ser.readline().decode().strip()
    if line:
        try:
            temperature, humidity = map(float, line.split())
            input_data.append([temperature, humidity])
            valid_samples += 1
        except ValueError:
            print(f"Dữ liệu không hợp lệ: {line}")

# Chuẩn hóa và reshape đầu vào
input_data = scaler.fit_transform(input_data)
input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1])

# Biến đếm mẫu
sample_count = 1

# Vòng lặp dự đoán liên tục
while True:
    # Dự đoán nhiệt độ và độ ẩm
    predictions = model.predict(input_data)

    # Inverse transform kết quả dự đoán
    predictions_inverse = scaler.inverse_transform(predictions.reshape(-1, 2))

    # Hiển thị kết quả dự đoán
    print(f"Kết quả dự đoán cho mẫu {sample_count}:")
    for i in range(predictions_inverse.shape[0]):
        print(f"Nhiệt độ={predictions_inverse[i, 0]}, Độ ẩm={predictions_inverse[i, 1]}")

    # Đọc mẫu mới từ cổng serial
    line = ser.readline().decode().strip()
    if line:
        try:
            new_sample = scaler.transform([[float(x) for x in line.split()]])
        except ValueError:
            print(f"Dữ liệu không hợp lệ: {line}")
            continue

        # Di chuyển cửa sổ dữ liệu
        input_data = np.concatenate((input_data[:, 1:, :], new_sample.reshape(1, 1, -1)), axis=1)

    # Tăng biến đếm mẫu
    sample_count += 1

# Đóng kết nối cổng serial
ser.close()