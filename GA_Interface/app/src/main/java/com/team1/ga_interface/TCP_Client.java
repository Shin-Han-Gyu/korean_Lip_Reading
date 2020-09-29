package com.team1.ga_interface;

import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;

import androidx.annotation.RequiresApi;

public class TCP_Client extends AsyncTask {
    protected static String SERV_IP = "164.125.35.26"; //서버의 ip주소를 작성하면 됩니다.
    protected static int PORT = 9999;

    private String filename;

    @RequiresApi(api = Build.VERSION_CODES.R)
    @Override
    protected Object doInBackground(Object[] objects) {
        try {
            Log.d("TCP", "server connecting");
            InetAddress serverAddr = InetAddress.getByName(SERV_IP);
            Socket sock = new Socket(serverAddr, PORT);

            try{
                System.out.println("데이터찾는중");

                File file = new File(Environment.getExternalStorageDirectory(), "/DCIM/Camera/"+filename);

                DataInputStream dis = new DataInputStream(new FileInputStream(file));
                DataOutputStream dos = new DataOutputStream(sock.getOutputStream());

                long fileSize = file.length();
                byte[] buf = new byte[4096];

                long totalReadBytes = 0;
                int readBytes;
                System.out.println("데이터찾기 끝");

                while ((readBytes = dis.read(buf)) > 0) { //길이 정해주고 서버로 보냅니다.
                    System.out.println("while");
                    dos.write(buf, 0, readBytes);
                    totalReadBytes += readBytes;
                }

                System.out.println("데이터보내기 끝 직전");
                dos.close();
                System.out.println("데이터끝");
                sock.close();
                System.out.println("소켓 닫음");

            } catch(IOException e){
                Log.d("TCP", "don't send message");
                e.printStackTrace();
            }

        } catch (UnknownHostException e) {
            e.printStackTrace();
        } catch(IOException    e){
            e.printStackTrace();
        }
        return null;
    }
    protected void setfileName(String name){
        this.filename = name;
    }
}
