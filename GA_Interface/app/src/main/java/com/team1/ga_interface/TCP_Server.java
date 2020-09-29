package com.team1.ga_interface;

import android.os.AsyncTask;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;

public class TCP_Server extends AsyncTask {
//    protected static String SERV_IP = "192.168.0.119"; //서버의 ip주소를 작성하면 됩니다.
    protected static int PORT = 6677;

    private DataInputStream dis;
    private Socket sock;
    private ServerSocket serverSocket;

    @Override
    protected Object doInBackground(Object[] objects) {
        try{
            System.out.println("여기까진 가능");
            serverSocket = new ServerSocket();
            InetSocketAddress ipep = new InetSocketAddress(PORT);
            serverSocket.bind(ipep);
            System.out.println("여기까진 가능22");

            while(true){
                System.out.println("여기까진 가능333");
                sock = serverSocket.accept();
                System.out.println("여기까진 가능4444");
                System.out.println("클라이언트 연결");

                dis = new DataInputStream(sock.getInputStream());

                String readline = dis.readLine();

                System.out.println("---");
                System.out.println(readline);
                System.out.println("---");

                sock.close();
                serverSocket.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }
}