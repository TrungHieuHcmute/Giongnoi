import requests
import time

url = 'https://libraryrobot.online/data.txt'  # Điền URL của trang web PHP ở đây

while True:
    response = requests.get(url)

    if response.status_code == 200:
        data = response.text
        print(data)
        if data == 'A':
            print("ok")
            response = requests.post(url, data="")
            if response.status_code == 200:
                print("Đã gửi lại dữ liệu trống thành công.")
            else:
                print("Gửi lại dữ liệu thất bại. Mã trạng thái HTTP:", response.status_code)
    else:
        print('Yêu cầu không thành công. Mã trạng thái HTTP:', response.status_code)
    time.sleep(1)  # Ngủ 5 giây trước khi gửi yêu cầu tiếp theo
