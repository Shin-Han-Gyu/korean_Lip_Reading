package com.team1.ga_interface;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.EditText;
import android.widget.TextView;

public class ResultActivity extends AppCompatActivity {

    private TextView mTextview;
    private EditText mEdittext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        mTextview = (TextView) findViewById(R.id.titleView);
        mEdittext = (EditText) findViewById(R.id.editText);

        String title = getIntent().getStringExtra("Title");
        mTextview.setText(title);
    }
}