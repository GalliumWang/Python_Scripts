import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('202.38.93.111', 1)
client_socket.connect(server_address)

request_header = 'GET / HTTP/1.0\r\n\r\n\r\n'
client_socket.send(request_header)

response = ''
while True:
    recv = client_socket.recv(1024)
    if not recv:
        break
    response += recv 

print(response)
client_socket.close()  