import requests
import time

url = 'https://libraryrobot.online/data_drive.txt'  # Điền URL của trang web PHP ở đây

while True:
    response = requests.get(url)

    if response.status_code == 200:
        data = response.text
        print(data)
    else:
        print('Yêu cầu không thành công. Mã trạng thái HTTP:', response.status_code)
    time.sleep(1)  # Ngủ 5 giây trước khi gửi yêu cầu tiếp theo