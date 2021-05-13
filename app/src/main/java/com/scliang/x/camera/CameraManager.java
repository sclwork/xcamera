package com.scliang.x.camera;

import android.app.Activity;
import android.content.Context;
import android.opengl.GLSurfaceView;
import android.text.TextUtils;
import android.view.Window;
import android.view.WindowManager;

import androidx.annotation.NonNull;
import androidx.annotation.RawRes;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class CameraManager {
    /*
     *
     */
    public static void init(Context context, GLSurfaceView glView) {
        CameraManager m = SingletonHolder.INSTANCE;
        m.mCtx = new SoftReference<>(context);
        m.initGlslResource(context);
        if (glView != null) {
            glView.setEGLContextClientVersion(3);
            glView.setRenderer(new CameraRenderer());
            glView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
            glView.queueEvent(()->m.jniInit(m.getFileRootPath(context)));
            GlView = new SoftReference<>(glView);
        }
    }

    public static void start() {
        CameraManager m = SingletonHolder.INSTANCE;
        m.acquireScreen();
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(m::jniResume);
            GlView.get().onResume();
        }
    }

    public static void stop() {
        CameraManager m = SingletonHolder.INSTANCE;
        m.releaseScreen();
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(m::jniPause);
            GlView.get().onPause();
        }
    }

    public static void release() {
        CameraManager m = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(m::jniRelease);
        }
        GlView = null;
    }


    /*
     *
     */
    public static List<String> getSupportedEffectPaints() {
        return new ArrayList<>(effectNames);
    }

    public static void updateEffectPaint(String name) {
        CameraManager m = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(()->m.jniUpdatePaint(name));
        }
    }


    /*
     *
     */
    public static void preview(String... cameras) {
        if (cameras.length > 0) {
            StringBuilder sb = new StringBuilder();
            for (String c : cameras) sb.append(",").append(c);
            CameraManager m = SingletonHolder.INSTANCE;
            if (GlView != null && GlView.get() != null) {
                GlView.get().queueEvent(()->m.jniPreview(sb.substring(1)));
            }
        }
    }


    /*
     *
     */
    private void initGlslResource(Context context) {
        effectNames.clear();
        setupGlslFiles(context);
        setupErrorTipFile(context);
    }


    /*
     *
     */
    private String getFileRootPath(Context context) {
        try {
            File dir = context.getDir("files", Context.MODE_PRIVATE);
            return dir.getAbsolutePath();
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }

    private void setupErrorTipFile(Context context) {
        InputStream is = null;
        FileOutputStream os = null;
        try {
            is = context.getResources().openRawResource(R.raw.ic_vid_file_not_exists);
            File dir = context.getDir("files", Context.MODE_PRIVATE);
            File file = new File(dir, "ic_vid_file_not_exists.png");
            if (file.exists()) file.delete();
            os = new FileOutputStream(file);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1)
                os.write(buffer, 0, bytesRead);

            file.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try { if (is != null) is.close();
            } catch (IOException ignored) { }
            try { if (os != null) os.close();
            } catch (IOException ignored) { }
        }
    }

    private void getShaderFile(Context context, @RawRes int raw, String name) {
        InputStream is = null;
        FileOutputStream os = null;
        try {
            if (TextUtils.isEmpty(name)) {
                return;
            }

            if (name.contains("shader_frag_effect_")) {
                effectNames.add(
                        name.replace("shader_frag_effect_", "")
                                .replace(".glsl", "")
                                .toUpperCase());
            }

            is = context.getResources().openRawResource(raw);
            File dir = context.getDir("files", Context.MODE_PRIVATE);
            File file = new File(dir, name);
            if (file.exists()) file.delete();
            os = new FileOutputStream(file);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1)
                os.write(buffer, 0, bytesRead);

            file.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try { if (is != null) is.close();
            } catch (IOException ignored) { }
            try { if (os != null) os.close();
            } catch (IOException ignored) { }
        }
    }

    private void setupGlslFiles(Context context) {
        getShaderFile(context,
                R.raw.shader_frag_none,
                "shader_frag_none.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_none,
                "shader_frag_effect_none.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_face,
                "shader_frag_effect_face.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_ripple,
                "shader_frag_effect_ripple.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_distortedtv,
                "shader_frag_effect_distortedtv.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_distortedtv_box,
                "shader_frag_effect_distortedtv_box.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_distortedtv_glitch,
                "shader_frag_effect_distortedtv_glitch.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_distortedtv_crt,
//                "shader_frag_effect_distortedtv_crt.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_floyd,
                "shader_frag_effect_floyd.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_3basic,
//                "shader_frag_effect_3basic.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_3floyd,
//                "shader_frag_effect_3floyd.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_pagecurl,
//                "shader_frag_effect_pagecurl.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_old_video,
                "shader_frag_effect_old_video.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_crosshatch,
                "shader_frag_effect_crosshatch.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_cmyk,
                "shader_frag_effect_cmyk.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_drawing,
                "shader_frag_effect_drawing.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_neon,
                "shader_frag_effect_neon.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_fisheye,
                "shader_frag_effect_fisheye.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_barrelblur,
                "shader_frag_effect_barrelblur.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_fastblur,
                "shader_frag_effect_fastblur.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_illustration,
                "shader_frag_effect_illustration.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_hexagon,
                "shader_frag_effect_hexagon.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_sobel,
                "shader_frag_effect_sobel.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_lens,
//                "shader_frag_effect_lens.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_float_camera,
                "shader_frag_effect_float_camera.glsl");
        getShaderFile(context,
                R.raw.shader_vert_none,
                "shader_vert_none.glsl");
        getShaderFile(context,
                R.raw.shader_vert_effect_none,
                "shader_vert_effect_none.glsl");
    }


    /*
     *
     */
    private static class CameraRenderer implements GLSurfaceView.Renderer {
        private final CameraManager cm = SingletonHolder.INSTANCE;
        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config){cm.jniSurfaceCreated();}
        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height){cm.jniSurfaceChanged(width, height);}
        @Override
        public void onDrawFrame(GL10 gl){cm.jniDrawFrame();}
    }
    /*
     *
     */
    private void acquireScreen() {
        Context ctx = mCtx == null ? null : mCtx.get();
        if (GlView != null && GlView.get() != null && ctx instanceof Activity) {
            GlView.get().post(() -> {
                Window window = ((Activity)ctx).getWindow();
                if (window != null) {
                    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                }
            });
        }
    }
    private void releaseScreen() {
        Context ctx = mCtx == null ? null : mCtx.get();
        if (GlView != null && GlView.get() != null && ctx instanceof Activity) {
            GlView.get().post(() -> {
                Window window = ((Activity)ctx).getWindow();
                if (window != null) {
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                }
            });
        }
    }
    /*
     *
     */
    private SoftReference<Context> mCtx;
    private static SoftReference<GLSurfaceView> GlView;
    /*
     *
     */
    private static void requestRender(int code) {
        if(GlView !=null&& GlView.get()!=null) GlView.get().requestRender(); }
    /*
     *
     */
    static { System.loadLibrary("xcamera-lib"); }
    /*
     *
     */
    private native int jniInit(@NonNull String fileRoot);
    private native int jniResume();
    private native int jniPause();
    private native int jniRelease();
    private native int jniSurfaceCreated();
    private native int jniSurfaceChanged(int width, int height);
    private native int jniUpdatePaint(@NonNull String name);
    private native int jniDrawFrame();
    private native int jniPreview(@NonNull String cameras);
    /*
     *
     */
    private final static List<String> effectNames = new ArrayList<>();
    /*
     *
     */
    private static class SingletonHolder { private static final CameraManager INSTANCE = new CameraManager(); }
    private CameraManager() {}
}
