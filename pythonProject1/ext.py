import requests

# end_device_id đã được lấy từ API lấy unit của user
end_device_id = "123456789"

# Tạo yêu cầu GET đến API lấy widget dựa vào end device id
url = f"https://backend.eoh.io/api/property_manager/devices/{end_device_id}/display/"
headers = {"Authorization": "Token your_token_here"}
response = requests.get(url, headers=headers)

# Kiểm tra mã trạng thái của phản hồi
if response.status_code == 200:
    # Trích xuất dữ liệu JSON từ phản hồi
    data = response.json()

    # Trích xuất widget_id từ dữ liệu JSON
    widget_id = data["Items"][0]["id"]

    # In widget_id ra để kiểm tra
    print(widget_id)
else:
    print("Lỗi yêu cầu GET đến API")
