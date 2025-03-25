import torch
print(torch.cuda.is_available())  # True nếu GPU hoạt động
print(torch.cuda.device_count())  # Số GPU có sẵn
print(torch.cuda.get_device_name(0))  # Tên GPU đang dùng
