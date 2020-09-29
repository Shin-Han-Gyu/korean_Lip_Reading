import socket
import signal
import os
import sys
import moviepy.editor as mp
import ffmpeg
from pydub import AudioSegment

HOST = '164.125.35.26'

PORT = 9999

def getFileFromServer():

   data_transferred = 0

   result = None

   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   #sock.settimeout(5)
   sock.bind((HOST,PORT))

   sock.listen()

   client_sock, addr = sock.accept()
   client_sock.settimeout(2)
   print('Connected by :', addr) 

   data = client_sock.recv(4096)

   if not data:

      print("NOT EXIST")

      return


   with open('./output.mp4', 'wb') as f:

      try:

         while data:

            f.write(data)

            data_transferred += len(data)

            data = client_sock.recv(4096)
         
      except Exception as e:
         
         print(e)


   os.system('python importTest.py')

   filename = '/home/vblab/team1/hoseok/output.txt'
   f = open(filename, mode='rb')
   f.read(3)
   client_sock.sendall(f.read())

   #client_sock.sendall('&&&'.encode())

   filename2 = '/home/vblab/team1/kwon/mergeOut.txt'
   f2 = open(filename2, mode='rb')
   client_sock.sendall(f2.read())


   client_sock.close()
   sock.close()


getFileFromServer()
