package com.team1.ga_interface;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.File;
import java.io.FilenameFilter;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CAMERA_PERMISSION = 1234;
    private static final int RECORD_REQUEST_CODE = 1001;
    private static final int SELECT_VIDEO_CODE = 1002;
    private static final int RECORGNITION_REQUEST_CODE = 1003;

    private Button mRecord;
    private Button mRecognition;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
            return;
        }

        mRecord = findViewById(R.id.record);
        mRecognition = findViewById(R.id.recognition);

        mRecord.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent recordVideo = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
                startActivityForResult(recordVideo, REQUEST_CAMERA_PERMISSION);
            }
        });

        mRecognition.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // 동영상선택
                Intent i = new Intent(Intent.ACTION_GET_CONTENT, MediaStore.Video.Media.INTERNAL_CONTENT_URI);
                //EXTERNAL_CONTENT_URI
                i.setType("video/*");
                i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                try
                {
                    startActivityForResult(i,SELECT_VIDEO_CODE);
                } catch (android.content.ActivityNotFoundException e)
                {
                    e.printStackTrace();
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == RECORD_REQUEST_CODE){
            File folder = new File(Environment.getExternalStorageDirectory(), "/DCIM/Camera");
            long folderModi = folder.lastModified();

            FilenameFilter filter = new FilenameFilter() {
                public boolean accept(File dir, String name) {
                    return (name.endsWith("mp4"));
                }
            };

            File[] folderList = folder.listFiles(filter);

            String recentName = "";

            for (int i = 0; i < folderList.length; i++) {
                long fileModi = folderList[i].lastModified();

                if (folderModi == fileModi) {
                    recentName = folderList[i].getName();
                }
            }
        }
        else if (requestCode == SELECT_VIDEO_CODE && resultCode == RESULT_OK){
            Uri uri = data.getData();
            String name = getName(uri);

            Intent intent = new Intent(getApplicationContext(),SelectActivity.class);
            intent.putExtra("filename", name);
            intent.putExtra("uri", uri.toString());
            Bundle bundle = new Bundle();
            bundle.putParcelable("videoUri", uri);
            intent.putExtras(bundle);

            startActivityForResult(intent, RECORGNITION_REQUEST_CODE);
        }
    }

    //파일명 찾기
    private String getName(Uri uri){
        String[] projection = {MediaStore.Video.VideoColumns.DISPLAY_NAME};
        Cursor cursor = managedQuery(uri, projection, null, null, null);
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Video.VideoColumns.DISPLAY_NAME);
        cursor.moveToFirst();
        return cursor.getString(column_index);
    }
}