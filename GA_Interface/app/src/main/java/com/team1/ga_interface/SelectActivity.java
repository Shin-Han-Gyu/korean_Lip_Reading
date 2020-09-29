package com.team1.ga_interface;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.w3c.dom.Text;

import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Iterator;
import java.util.Set;

public class SelectActivity extends AppCompatActivity {
    private Button mSound;
    private Button mLip;
    private TextView mSelect;
    private String filename;

    private TCP_Client tc;

    public InputStream inputStream;
    public OutputStream outputStream;
    private ServerSocket serverSocket;
    private Socket socket;
    private String ip = "164.125.35.26";
    private int port = 9996;

    Thread thread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_select);

        mSound = (Button) findViewById(R.id.sound_Btn);
        mLip = (Button) findViewById(R.id.lip_Btn);
        mSelect = (TextView) findViewById(R.id.selectfile);

        mSound.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(),ResultActivity.class);
                intent.putExtra("Title",mSound.getText());
                startActivityForResult(intent, 2001);
            }
        });

        mLip.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(),ResultActivity.class);
                intent.putExtra("Title",mLip.getText());
                startActivityForResult(intent, 2002);
            }
        });

        filename = getIntent().getStringExtra("filename");

        mSelect.setText("선택된 파일 : "+filename);
    }

    @Override
    protected void onStart() {
        super.onStart();
        tc = new TCP_Client();
        tc.setfileName(filename);
        tc.execute(this);

        thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try{
//                    socket = new Socket(ip,port);
                    serverSocket = new ServerSocket();
                    InetSocketAddress ipep = new InetSocketAddress(port);
                    serverSocket.bind(ipep);

                } catch (IOException e) {
                    e.printStackTrace();
                }

                byte[] buffer = new byte[1024];
                int bytes;

                while(true){
                    try{
                        System.out.println("수신 대기");
                        socket = serverSocket.accept();
                        System.out.println("수신 성공!!");
                        inputStream = socket.getInputStream();
                        outputStream = socket.getOutputStream();
                        System.out.println("데이터 전송 시작");
                        bytes = inputStream.read(buffer);
                        String tmp = new String(buffer, 0, bytes);
                        System.out.println(tmp);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }
        });
        thread.start();
    }
}