# 配合src/tools/zmq_remote_show.cpp中的zmq remote实现远程显示服务器画面的效果
# pip install zmq

import cv2
import zmq
import numpy as np

if __name__ == "__main__":
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.101.66:15556")

    while True:
        
        socket.send(b"a")
        message = socket.recv()

        if(len(message) == 1 and message == b'x'):
            break

        image = np.frombuffer(message, dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        print(f"width = {image.shape[1]}, height = {image.shape[0]}")

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        key = cv2.waitKey(1) & 0xFF

        if(key == ord('q')):
            break
    
    print("done.")
    cv2.destroyAllWindows()