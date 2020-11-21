import os

import socket
import threading


def tcplink(sock, addr):
    while True:
        data = sock.recv(0x100_000)
        os.system('clear')
        print(data.decode('utf-8'))


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 9999))
    s.listen(5)
    while True:
        # 接受一个新连接:
        g_sock, g_addr = s.accept()
        # 创建新线程来处理TCP连接:
        t = threading.Thread(target=tcplink, args=(g_sock, g_addr))
        t.start()
