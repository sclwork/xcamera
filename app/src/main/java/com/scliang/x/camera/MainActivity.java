package com.scliang.x.camera;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import java.util.List;

public class MainActivity extends AppCompatActivity {
    private final String[] C01 = new String[] { "0","1" };
    private final String[] C10 = new String[] { "1","0" };
    private int camMerge = 0;
    private String[] cs = C01;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        CameraManager.init(this, findViewById(R.id.gl_view));
        CameraManager.preview(camMerge, cs);
        checkPermissions();
    }

    @Override
    protected void onStart() {
        super.onStart();
        CameraManager.start();
    }

    @Override
    protected void onStop() {
        super.onStop();
        CameraManager.stop();
    }

    @Override
    protected void onDestroy() {
        CameraManager.release();
        super.onDestroy();
    }

    private void checkPermissions() {
        if (PackageManager.PERMISSION_GRANTED != checkSelfPermission(Manifest.permission.CAMERA) ||
                PackageManager.PERMISSION_GRANTED != checkSelfPermission(Manifest.permission.RECORD_AUDIO)) {
            requestPermissions(new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO}, 1001);
            setupPermissionView(false);
        } else {
            setupPermissionView(true);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1001 &&
                PackageManager.PERMISSION_GRANTED == grantResults[0] &&
                PackageManager.PERMISSION_GRANTED == grantResults[1]) {
            setupPermissionView(true);
        }
    }


    /*
     *
     */
    private void setupPermissionView(boolean hasPermission) {
        View noPermission = findViewById(R.id.no_permissions);
        if (noPermission != null) {
            noPermission.setVisibility(hasPermission?View.GONE:View.VISIBLE);
        }
        Button camera = findViewById(R.id.camera);
        Button effect = findViewById(R.id.effect);
        Button merge = findViewById(R.id.merge);
        if (hasPermission) {
            if (camera != null) camera.setVisibility(View.VISIBLE);
            if (camera != null) camera.setOnClickListener(v -> {
                if (cs == C01) cs = C10;
                else if (cs == C10) cs = C01;
                CameraManager.preview(camMerge, cs);
            });
            if (effect != null) effect.setVisibility(View.VISIBLE);
            if (effect != null) effect.setOnClickListener(v -> {
                List<String> names = CameraManager.getSupportedEffectPaints();
                String[] items = new String[names.size()];
                for (int i = 0; i < names.size(); i++) {
                    String name = names.get(i);
                    items[i] = name;
                }
                AlertDialog.Builder listDialog = new AlertDialog.Builder(this);
                listDialog.setItems(items, (dialog, which) -> {
                    String name = items[which];
                    effect.setText(name);
                    CameraManager.updateEffectPaint(name);
                });
                listDialog.show();
            });
            if (merge != null) merge.setVisibility(View.VISIBLE);
            if (merge != null) merge.setOnClickListener(v -> {
                String[] items = new String[] { "SINGLE", "VERTICAL", "CHAT" };
                AlertDialog.Builder listDialog = new AlertDialog.Builder(this);
                listDialog.setItems(items, (dialog, which) -> {
                    String name = items[which];
                    merge.setText(name);
                    camMerge = which;
                    CameraManager.preview(camMerge, cs);
                });
                listDialog.show();
            });
        } else {
            if (camera != null) camera.setVisibility(View.GONE);
            if (effect != null) effect.setVisibility(View.GONE);
            if (merge != null) merge.setVisibility(View.GONE);
        }
    }
}
