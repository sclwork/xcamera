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

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        CameraManager.init(this, findViewById(R.id.gl_view));
        CameraManager.preview("0", "1");
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
        Button effect = findViewById(R.id.effect);
        if (hasPermission) {
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
        } else {
            if (effect != null) effect.setVisibility(View.GONE);
        }
    }
}
