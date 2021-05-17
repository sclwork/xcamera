#include <jni.h>
#include <regex>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <cstring>
#include <ctime>
#include <glm/glm.hpp>
#include <android/log.h>
#include <libyuv/libyuv.h>
#include <concurrent_queue.h>
#include <glm/detail/type_mat.hpp>
#include <glm/detail/type_mat4x4.hpp>
#include <glm/ext.hpp>
#include <GLES3/gl3.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>

#define log_d(...)  __android_log_print(ANDROID_LOG_DEBUG, "xcamera", __VA_ARGS__)
#define log_e(...)  __android_log_print(ANDROID_LOG_ERROR, "xcamera", __VA_ARGS__)

namespace x {
    /*
     *
     */
    static std::string *FileRoot = nullptr;
    static std::string *EffectName = nullptr;
    static std::atomic_bool CollectRunnable(false);


    /*
     *
     */
    typedef struct image_args {
        image_args(uint32_t w, uint32_t h, uint32_t c, uint32_t f)
                :width(w), height(h), channels(c), fps(f) { frame_size = width*height*channels; }
        uint32_t width, height, channels, fps, frame_size;
    } image_args;


    /*
     *
     */
    typedef struct audio_args {
        audio_args(uint32_t c, uint32_t s, uint32_t f)
                :channels(c), sample_rate(s), frame_size(f) {}
        uint32_t channels, sample_rate, frame_size;
    } audio_args;


    /*
     *
     */
    typedef struct yuv_args {
        int32_t i, j, x, y, wof, hof, frame_w, frame_h, format, plane_count,
                y_stride, u_stride, v_stride, vu_pixel_stride, y_len, u_len, v_len,
                ori, src_w, src_h, img_width, img_height;
        uint8_t *y_pixel, *u_pixel, *v_pixel, *argb_pixel, *dst_argb_pixel;
        uint32_t *frame_cache, argb;
        AImageCropRect src_rect;
        AImage *image;
    } yuv_args;


    /*
     *
     */
    class GlUtils {
    public:
        static void setBool(GLuint programId, const std::string &name, bool value) {
            glUniform1i(glGetUniformLocation(programId, name.c_str()), (int) value);
        }

        static void setInt(GLuint programId, const std::string &name, int value) {
            glUniform1i(glGetUniformLocation(programId, name.c_str()), value);
        }

        static void setFloat(GLuint programId, const std::string &name, float value) {
            glUniform1f(glGetUniformLocation(programId, name.c_str()), value);
        }

        static void setVec2(GLuint programId, const std::string &name, const glm::vec2 &value) {
            glUniform2fv(glGetUniformLocation(programId, name.c_str()), 1, &value[0]);
        }

        static void setVec2(GLuint programId, const std::string &name, float x, float y) {
            glUniform2f(glGetUniformLocation(programId, name.c_str()), x, y);
        }

        static void setVec3(GLuint programId, const std::string &name, const glm::vec3 &value) {
            glUniform3fv(glGetUniformLocation(programId, name.c_str()), 1, &value[0]);
        }

        static void setVec3(GLuint programId, const std::string &name, float x, float y, float z) {
            glUniform3f(glGetUniformLocation(programId, name.c_str()), x, y, z);
        }

        static void setVec4(GLuint programId, const std::string &name, const glm::vec4 &value) {
            glUniform4fv(glGetUniformLocation(programId, name.c_str()), 1, &value[0]);
        }

        static void setVec4(GLuint programId, const std::string &name, float x, float y, float z, float w) {
            glUniform4f(glGetUniformLocation(programId, name.c_str()), x, y, z, w);
        }

        static void setMat2(GLuint programId, const std::string &name, const glm::mat2 &mat) {
            glUniformMatrix2fv(glGetUniformLocation(programId, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        }

        static void setMat3(GLuint programId, const std::string &name, const glm::mat3 &mat) {
            glUniformMatrix3fv(glGetUniformLocation(programId, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        }

        static void setMat4(GLuint programId, const std::string &name, const glm::mat4 &mat) {
            glUniformMatrix4fv(glGetUniformLocation(programId, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        }

        static glm::vec3 texCoordToVertexCoord(const glm::vec2& texCoord) {
            return glm::vec3(2 * texCoord.x - 1, 1 - 2 * texCoord.y, 0);
        }
    };


    /*
     *
     */
    class ImageFrame {
    public:
        ImageFrame(): ori(0), width(0), height(0), cache(nullptr) {}

        ImageFrame(int32_t w, int32_t h, bool perch = false): ori(0), width(w), height(h),
                                                              cache((uint32_t*)malloc(sizeof(uint32_t)*width*height)) {
            if (cache == nullptr) {
                log_e("ImageFrame malloc image cache fail.");
            } else if (FileRoot != nullptr) {
                if (perch) {
                    cv::Mat img = cv::imread(*FileRoot + "/ic_vid_file_not_exists.png");
                    cv::cvtColor(img, img, cv::COLOR_BGRA2RGBA);
                    int32_t wof = (w - img.cols) / 2;
                    int32_t hof = (h - img.rows) / 2;
                    for (int32_t i = 0; i < img.rows; i++) {
                        for (int32_t j = 0; j < img.cols; j++) {
                            cache[(i + hof) * w + j + wof] =
                                    (((int32_t) img.data[(i * img.cols + j) * 4 + 3]) << 24) +
                                    (((int32_t) img.data[(i * img.cols + j) * 4 + 2]) << 16) +
                                    (((int32_t) img.data[(i * img.cols + j) * 4 + 1]) << 8) +
                                    (img.data[(i * img.cols + j) * 4]);
                        }
                    }
                } else {
                    memset(cache, 0, sizeof(uint32_t) * width * height);
                }
            }
        }

        ImageFrame(ImageFrame &&frame) noexcept: ori(frame.ori),
                                                 width(frame.width), height(frame.height),
                                                 cache((uint32_t*)malloc(sizeof(uint32_t)*width*height)) {
            if (cache) {
                memcpy(cache, frame.cache, sizeof(uint32_t) * width * height);
            }
        }

        ImageFrame(const ImageFrame &frame) noexcept: ori(frame.ori),
                                                      width(frame.width), height(frame.height),
                                                      cache((uint32_t*)malloc(sizeof(uint32_t)*width*height)) {
            if (cache) {
                memcpy(cache, frame.cache, sizeof(uint32_t) * width * height);
            }
        }

        ImageFrame& operator=(ImageFrame &&frame) noexcept {
            if (ori != frame.ori || width != frame.width || height != frame.height) {
                ori = frame.ori;
                width = frame.width;
                height = frame.height;
                if (cache) free(cache);
                cache = (uint32_t *) malloc(sizeof(uint32_t) * width * height);
            }
            if (cache) { memcpy(cache, frame.cache, sizeof(uint32_t) * width * height); }
            return *this;
        }

        ImageFrame& operator=(const ImageFrame &frame) noexcept {
            if (ori != frame.ori || width != frame.width || height != frame.height) {
                ori = frame.ori;
                width = frame.width;
                height = frame.height;
                if (cache) free(cache);
                cache = (uint32_t *) malloc(sizeof(uint32_t) * width * height);
            }
            if (cache) { memcpy(cache, frame.cache, sizeof(uint32_t) * width * height); }
            return *this;
        }

        ~ImageFrame() {
            if (cache) free(cache);
            cache = nullptr;
        }

    public:
        /**
         * check frame size
         * @param w frame width
         * @param h frame height
         * @return true: same size
         */
        bool sameSize(int32_t w, int32_t h) const
            { return w == width && h == height; }
        /**
         * check image frame available
         */
        bool available() const
            { return cache != nullptr; }

    public:
        /**
         * update frame size
         * @param w frame width
         * @param h frame height
         * if w/h changed, remalloc data cache.
         */
        void updateSize(int32_t w, int32_t h) {
            if (w <= 0 || h <= 0) {
                return;
            }

            if (sameSize(w, h)) {
                if (cache) memset(cache, 0, sizeof(uint32_t) * width * height);
            } else {
                if (cache) free(cache);
                width = w; height = h;
                cache = (uint32_t*)malloc(sizeof(uint32_t) * width * height);
                if (cache) memset(cache, 0, sizeof(uint32_t) * width * height);
            }
        }

        /**
         * setup camera/image orientation
         * @param o orientation:[0|90|180|270]
         */
        void setOrientation(int32_t o) { ori = o; }
        int32_t getOrientation() const { return ori; }

        /**
         * @return true: if camera/image orientation is 270
         */
        bool useMirror() const
            { return ori == 270; }

        /**
         * get image frame args/data pointer
         * @param out_w [out] frame width
         * @param out_h [out] frame height
         * @param out_cache [out] frame data pointer
         */
        void get(int32_t *out_w, int32_t *out_h, uint32_t **out_cache = nullptr) const {
            if (out_w) *out_w = width;
            if (out_h) *out_h = height;
            if (out_cache) *out_cache = cache;
        }

        uint32_t *getData() const {
            return cache;
        }

    private:
        int32_t ori;
        int32_t width;
        int32_t height;
        uint32_t *cache;
    };


    /*
     *
     */
    class Audio;
    class AudioFrame {
    public:
        AudioFrame(): offset(0), size(0), cache(nullptr) {
        }

        explicit AudioFrame(int32_t sz): offset(0), size(sz),
                                         cache((uint8_t*)malloc(sizeof(uint8_t)*size)) {
        }

        AudioFrame(AudioFrame&& frame) noexcept: offset(0), size(frame.size),
                                                 cache((uint8_t*)malloc(sizeof(uint8_t)*size)) {
            if (cache) { memcpy(cache, frame.cache, sizeof(int8_t)*size); }
        }

        AudioFrame(const AudioFrame &frame): offset(0), size(frame.size),
                                             cache((uint8_t*)malloc(sizeof(uint8_t)*size)) {
            if (cache) { memcpy(cache, frame.cache, sizeof(int8_t)*size); }
        }

        AudioFrame& operator=(AudioFrame&& frame) noexcept {
            if (size != frame.size) {
                size = frame.size;
                if (cache) free(cache);
                cache = (uint8_t *) malloc(sizeof(uint8_t) * size);
            }
            if (cache) { memcpy(cache, frame.cache, sizeof(int8_t)*size); }
            return *this;
        }

        AudioFrame& operator=(const AudioFrame &frame) noexcept {
            if (size != frame.size) {
                size = frame.size;
                if (cache) free(cache);
                cache = (uint8_t *) malloc(sizeof(uint8_t) * size);
            }
            if (cache) { memcpy(cache, frame.cache, sizeof(int8_t)*size); }
            return *this;
        }

        ~AudioFrame() {
            if (cache) free(cache);
        }

    public:
        /**
         * check audio available
         */
        bool available() const {
            return cache != nullptr;
        }
        /**
         * get audio frame args/pcm data
         * @param out_size [out] audio pcm data size
         * @param out_cache [out] audio pcm data pointer
         */
        void get(int32_t *out_size, uint8_t **out_cache = nullptr) const {
            if (out_size) *out_size = size;
            if (out_cache) *out_cache = cache;
        }
        /**
         * get audio frame pcm short data
         * @param out_size [out] audio pcm short data size
         * @return audio frame pcm short data pointer
         */
        std::shared_ptr<uint16_t> get(int32_t *out_size) const {
            if (out_size) *out_size = size / 2;
            auto sa = new uint16_t[size / 2];
            std::shared_ptr<uint16_t> sht(sa,[](const uint16_t*p){delete[]p;});
            for (int32_t i = 0; i < size / 2; i++) {
                sa[i] = ((uint16_t)(cache[i * 2])     & 0xff) +
                        (((uint16_t)(cache[i * 2 + 1]) & 0xff) << 8);
            }
            return sht;
        }
        /**
         * set short data to audio frame pcm
         * @param sht audio pcm short data
         * @param length audio pcm short data size
         */
        void set(const uint16_t *sht, int32_t length) {
            if (sht != nullptr && length > 0 && length * 2 == size && cache != nullptr) {
                for (int32_t i = 0; i < length; i++) {
                    cache[i * 2]     = (uint8_t) (sht[i]        & 0xff);
                    cache[i * 2 + 1] = (uint8_t)((sht[i] >> 8)  & 0xff);
                }
            }
        }

        void set(const std::shared_ptr<uint16_t> &sht, int32_t length) {
            if (sht != nullptr && length > 0 && length * 2 == size && cache != nullptr) {
                uint16_t *sd = sht.get();
                for (int32_t i = 0; i < length; i++) {
                    cache[i * 2]     = (uint8_t) (sd[i]        & 0xff);
                    cache[i * 2 + 1] = (uint8_t)((sd[i] >> 8)  & 0xff);
                }
            }
        }

    private:
        friend class Audio;

    private:
        int32_t  offset;
        int32_t  size;
        uint8_t *cache;
    };


    /*
     *
     */
    typedef moodycamel::ConcurrentQueue<ImageFrame> ImageQueue;


    /*
     *
     */
    typedef moodycamel::ConcurrentQueue<AudioFrame> AudioQueue;


    /*
     *
     */
    static void postImageFrame(ImageFrame &frame);


    /*
     *
     */
    class ImagePaint {
    public:
        explicit ImagePaint(const std::string &&e_name): effect(e_name), rf(),
                                                         cvs_width(0), cvs_height(0),
                                                         frm_index(0),
                                                         program(GL_NONE), effect_program(GL_NONE),
                                                         texture(GL_NONE),
                                                         src_fbo(GL_NONE), src_fbo_texture(GL_NONE),
                                                         dst_fbo(GL_NONE), dst_fbo_texture(GL_NONE) {
            srandom((unsigned)time(nullptr));
            log_d("ImagePaint[%s] created.", effect.c_str());
        }

        ~ImagePaint() {
            release();
            log_d("ImagePaint[%s] release.", effect.c_str());
        }

    public:
        /**
         * setup canvas size
         * @param width canvas width
         * @param height canvas height
         */
        void updateSize(int32_t width, int32_t height) {
            release();
            cvs_width = width;
            cvs_height = height;
            glViewport(0, 0, cvs_width, cvs_height);
            glClearColor(0.0, 0.0, 0.0, 1.0);

            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, GL_NONE);

            glGenTextures(1, &src_fbo_texture);
            glBindTexture(GL_TEXTURE_2D, src_fbo_texture);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, GL_NONE);

            glGenTextures(1, &dst_fbo_texture);
            glBindTexture(GL_TEXTURE_2D, dst_fbo_texture);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, GL_NONE);

            glGenFramebuffers(1, &src_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, src_fbo);
            glBindTexture(GL_TEXTURE_2D, src_fbo_texture);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src_fbo_texture, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                log_e("ImagePaint[%s] create src_fbo fail.", effect.c_str());
                if(src_fbo_texture != GL_NONE) {
                    glDeleteTextures(1, &src_fbo_texture);
                    src_fbo_texture = GL_NONE;
                }
                if(src_fbo != GL_NONE) {
                    glDeleteFramebuffers(1, &src_fbo);
                    src_fbo = GL_NONE;
                }
                glBindTexture(GL_TEXTURE_2D, GL_NONE);
                glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
            }

            glGenFramebuffers(1, &dst_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
            glBindTexture(GL_TEXTURE_2D, dst_fbo_texture);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst_fbo_texture, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                log_e("ImagePaint[%s] create dst_fbo fail.", effect.c_str());
                if(dst_fbo_texture != GL_NONE) {
                    glDeleteTextures(1, &dst_fbo_texture);
                    dst_fbo_texture = GL_NONE;
                }
                if(dst_fbo != GL_NONE) {
                    glDeleteFramebuffers(1, &dst_fbo);
                    dst_fbo = GL_NONE;
                }
                glBindTexture(GL_TEXTURE_2D, GL_NONE);
                glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
            }

            program = createProgram(vertShaderStr().c_str(), fragShaderStr().c_str());
            effect_program = createProgram(effectVertShaderStr().c_str(), effectFragShaderStr(effect).c_str());
        }

        /**
         * draw image frame
         * @param frame image frame
         */
        void draw(const ImageFrame &frame, ImageFrame *of= nullptr) {
            if (!frame.available()) {
                return;
            }

            bool mirror = frame.useMirror();
            int32_t width = 0, height = 0;
            frame.get(&width, &height);
            uint32_t *data = frame.getData();

            glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            if(effect_program == GL_NONE || src_fbo == GL_NONE || data == nullptr) {
                return;
            }

            glBindTexture(GL_TEXTURE_2D, GL_NONE);
            glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);

            glBindFramebuffer(GL_FRAMEBUFFER, src_fbo);
            glViewport(0, 0, width, height);
            glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glUseProgram(effect_program);
            glBindTexture(GL_TEXTURE_2D, src_fbo_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, dst_fbo_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            glBindTexture(GL_TEXTURE_2D, GL_NONE);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof (GLfloat), vcs);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof (GLfloat), mirror ? mirror_tcs : tcs);
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);

            glm::mat4 matrix;
            updateMatrix(matrix);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            GlUtils::setInt(effect_program, "s_Texture", 0);
            GlUtils::setMat4(effect_program, "u_MVPMatrix", matrix);
            setupProgramArgs(effect_program, width, height, mirror);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);
            glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);

            glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
            glViewport(0, 0, width, height);
            glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glUseProgram (program);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof (GLfloat), vcs);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof (GLfloat), mirror ? mirror_tcs : tcs);
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);

            glBindTexture(GL_TEXTURE_2D, src_fbo_texture);
            GlUtils::setInt(program, "s_Texture", 0);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);
            if (of != nullptr) pixels2Frame(of, width, height);
            glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);

            glViewport(0, 0, cvs_width, cvs_height);
            glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glUseProgram(program);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, dst_fbo_texture);
            GlUtils::setInt(program, "s_Texture", 0);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);
        }

    private:
        void release() {
            if (program != GL_NONE) glDeleteProgram(program);
            program = GL_NONE;
            if (effect_program != GL_NONE) glDeleteProgram(effect_program);
            effect_program = GL_NONE;
            if (texture != GL_NONE) glDeleteTextures(1, &texture);
            texture = GL_NONE;
            if (src_fbo_texture != GL_NONE) glDeleteTextures(1, &src_fbo_texture);
            src_fbo_texture = GL_NONE;
            if (src_fbo != GL_NONE) glDeleteFramebuffers(1, &src_fbo);
            src_fbo = GL_NONE;
            if (dst_fbo_texture != GL_NONE) glDeleteTextures(1, &dst_fbo_texture);
            dst_fbo_texture = GL_NONE;
            if (dst_fbo != GL_NONE) glDeleteFramebuffers(1, &dst_fbo);
            dst_fbo = GL_NONE;
        }

        void setupProgramArgs(GLuint prog, int32_t width, int32_t height, bool mirror) {
            GlUtils::setVec2(prog, "u_TexSize", glm::vec2(width, height));
            GlUtils::setBool(prog, "u_Mirror", mirror);

            if (effect == "FACE") {
                GlUtils::setFloat(prog, "u_Time", frm_index);
                GlUtils::setInt(prog, "u_FaceCount", 0);
                GlUtils::setVec4(prog, "u_FaceRect", glm::vec4(0, 0, 0, 0));
            } else if (effect == "RIPPLE") {
                auto time = (float)(fmod(frm_index, 200) / 160);
                if (time == 0.0) {
                    rf[0][0] = random() % width; rf[1][0] = random() % height;
                    rf[0][1] = random() % width; rf[1][1] = random() % height;
                    rf[0][2] = random() % width; rf[1][2] = random() % height;
                }
                GlUtils::setFloat(prog, "u_Time", time * 2.5f);
                GlUtils::setInt(prog, "u_FaceCount", 0);
                GlUtils::setVec4(prog, "u_FaceRect", glm::vec4(rf[0][0], rf[1][0], rf[0][0] + 80, rf[1][0] + 80));
                GlUtils::setVec4(prog, "u_RPoint", glm::vec4(rf[0][1] + 40, rf[1][1] + 40, rf[0][2] + 40, rf[1][2] + 40));
                GlUtils::setVec2(prog, "u_ROffset", glm::vec2(10 + random() % 10, 10 + random() % 10));
                GlUtils::setFloat(prog, "u_Boundary", 0.12);
            } else {
                GlUtils::setFloat(prog, "u_Time", frm_index);
            }

            frm_index++;
            if (frm_index == INT32_MAX) {
                frm_index = 0;
            }
        }

        static std::string readShaderStr(const std::string &name) {
            std::ostringstream buf;
            std::ifstream file(name);
            char ch;
            while(buf&&file.get(ch)) buf.put(ch);
            return buf.str();
        }

        static GLuint loadShader(GLenum shaderType, const char *pSource) {
            GLuint shader = glCreateShader(shaderType);
            if (shader) {
                glShaderSource(shader, 1, &pSource, nullptr);
                glCompileShader(shader);
                GLint compiled = 0;
                glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
                if (!compiled) {
                    GLint infoLen = 0;
                    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
                    if (infoLen) {
                        char* buf = (char*) malloc((size_t)infoLen);
                        if (buf) {
                            glGetShaderInfoLog(shader, infoLen, nullptr, buf);
                            log_e("LoadShader Could not compile shader %d: %s", shaderType, buf);
                            free(buf);
                        }
                        glDeleteShader(shader);
                        shader = 0;
                    }
                }
            }

            return shader;
        }

        static GLuint createProgram(const char *pVertexShaderSource, const char *pFragShaderSource) {
            GLuint prog = 0;
            GLuint vertexShaderHandle = loadShader(GL_VERTEX_SHADER, pVertexShaderSource);
            if (!vertexShaderHandle) {
                return prog;
            }

            GLuint fragShaderHandle = loadShader(GL_FRAGMENT_SHADER, pFragShaderSource);
            if (!fragShaderHandle) {
                return prog;
            }

            prog = glCreateProgram();
            if (prog) {
                glAttachShader(prog, vertexShaderHandle);
                glAttachShader(prog, fragShaderHandle);
                glLinkProgram(prog);
                GLint linkStatus = GL_FALSE;
                glGetProgramiv(prog, GL_LINK_STATUS, &linkStatus);

                glDetachShader(prog, vertexShaderHandle);
                glDeleteShader(vertexShaderHandle);
                glDetachShader(prog, fragShaderHandle);
                glDeleteShader(fragShaderHandle);
                if (linkStatus != GL_TRUE) {
                    GLint bufLength = 0;
                    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &bufLength);
                    if (bufLength) {
                        char* buf = (char*)malloc((size_t)bufLength);
                        if (buf) {
                            glGetProgramInfoLog(prog, bufLength, nullptr, buf);
                            log_e("GLUtils::CreateProgram Could not link program: %s", buf);
                            free(buf);
                        }
                    }
                    glDeleteProgram(prog);
                    prog = 0;
                }
            }

            return prog;
        }

        static std::string vertShaderStr() {
            if (FileRoot == nullptr) return "";
            return readShaderStr(*FileRoot + "/shader_vert_none.glsl");
        }

        static std::string fragShaderStr() {
            if (FileRoot == nullptr) return "";
            return readShaderStr(*FileRoot + "/shader_frag_none.glsl");
        }

        static std::string effectVertShaderStr() {
            if (FileRoot == nullptr) return "";
            return readShaderStr(*FileRoot + "/shader_vert_effect_none.glsl");
        }

        static std::string effectFragShaderStr(const std::string &effect) {
            if (FileRoot == nullptr) return "";
            std::string name;
            if (effect == "FACE") {
                name = "/shader_frag_effect_face.glsl";
            } else if (effect == "RIPPLE") {
                name = "/shader_frag_effect_ripple.glsl";
            } else if (effect == "DISTORTEDTV") {
                name = "/shader_frag_effect_distortedtv.glsl";
            } else if (effect == "DISTORTEDTV_BOX") {
                name = "/shader_frag_effect_distortedtv_box.glsl";
            } else if (effect == "DISTORTEDTV_GLITCH") {
                name = "/shader_frag_effect_distortedtv_glitch.glsl";
            } else if (effect == "FLOYD") {
                name = "/shader_frag_effect_floyd.glsl";
            } else if (effect == "OLD_VIDEO") {
                name = "/shader_frag_effect_old_video.glsl";
            } else if (effect == "CROSSHATCH") {
                name = "/shader_frag_effect_crosshatch.glsl";
            } else if (effect == "CMYK") {
                name = "/shader_frag_effect_cmyk.glsl";
            } else if (effect == "DRAWING") {
                name = "/shader_frag_effect_drawing.glsl";
            } else if (effect == "NEON") {
                name = "/shader_frag_effect_neon.glsl";
            } else if (effect == "FISHEYE") {
                name = "/shader_frag_effect_fisheye.glsl";
            } else if (effect == "FASTBLUR") {
                name = "/shader_frag_effect_fastblur.glsl";
            } else if (effect == "BARRELBLUR") {
                name = "/shader_frag_effect_barrelblur.glsl";
            } else if (effect == "ILLUSTRATION") {
                name = "/shader_frag_effect_illustration.glsl";
            } else if (effect == "HEXAGON") {
                name = "/shader_frag_effect_hexagon.glsl";
            } else if (effect == "SOBEL") {
                name = "/shader_frag_effect_sobel.glsl";
            } else if (effect == "LENS") {
                name = "/shader_frag_effect_lens.glsl";
            } else if (effect == "FLOAT_CAMERA") {
                name = "/shader_frag_effect_float_camera.glsl";
            } else {
                name = "/shader_frag_effect_none.glsl";
            }
            return readShaderStr(*FileRoot + name);
        }

        #define MATH_PI 3.1415926535897932384626433832802f
        static void updateMatrix(glm::mat4 &matrix,
                                 int32_t angleX = 0, int32_t angleY = 0,
                                 float scaleX = 1.0f, float scaleY = 1.0f) {
            angleX = angleX % 360;
            angleY = angleY % 360;

            auto radiansX = MATH_PI / 180.0f * angleX;
            auto radiansY = MATH_PI / 180.0f * angleY;

            // Projection matrix
//            glm::mat4 Projection = glm::ortho(-cvs_ratio, cvs_ratio, -1.0f, 1.0f, 0.0f, 100.0f);
            glm::mat4 Projection = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
//            glm::mat4 Projection = glm::frustum(-ratio, ratio, -1.0f, 1.0f, 4.0f, 100.0f);
//            glm::mat4 Projection = glm::perspective(45.0f,cvs_ratio, 0.1f,100.f);

            // View matrix
            glm::mat4 View = glm::lookAt(
                    glm::vec3(0, 0, 4), // Camera is at (0,0,1), in World Space
                    glm::vec3(0, 0, 0), // and looks at the origin
                    glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
            );

            // Model matrix
            glm::mat4 Model = glm::mat4(1.0f);
            Model = glm::scale(Model, glm::vec3(scaleX, scaleY, 1.0f));
            Model = glm::rotate(Model, radiansX, glm::vec3(1.0f, 0.0f, 0.0f));
            Model = glm::rotate(Model, radiansY, glm::vec3(0.0f, 1.0f, 0.0f));
            Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));

            matrix = Projection * View * Model;
        }

        static void pixels2Frame(ImageFrame *of, int32_t width, int32_t height) {
            uint32_t *of_data = nullptr;
            if (of->available()) {
                if (of->sameSize(width, height)) {
                    of->get(nullptr, nullptr, &of_data);
                } else {
                    of->updateSize(width, height);
                    of->get(nullptr, nullptr, &of_data);
                }
            } else {
                of->updateSize(width, height);
                of->get(nullptr, nullptr, &of_data);
            }
            if (of_data != nullptr) {
                glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, of_data);
            }
        }

    private:
        ImagePaint(ImagePaint&&) = delete;
        ImagePaint(const ImagePaint&) = delete;
        ImagePaint& operator=(ImagePaint&&) = delete;
        ImagePaint& operator=(const ImagePaint&) = delete;

    private:
        std::string  effect;
        long         rf[2][3];
        int32_t      cvs_width;
        int32_t      cvs_height;
        int32_t      frm_index;

    private:
        GLfloat      vcs[12]       { -1.0f,  1.0f, 0.0f,
                                     -1.0f, -1.0f, 0.0f,
                                      1.0f, -1.0f, 0.0f,
                                      1.0f,  1.0f, 0.0f, };
        GLfloat      tcs[8]        {  0.0f,  0.0f,
                                      0.0f,  1.0f,
                                      1.0f,  1.0f,
                                      1.0f,  0.0f, };
        GLfloat      mirror_tcs[8] {  1.0f,  0.0f,
                                      1.0f,  1.0f,
                                      0.0f,  1.0f,
                                      0.0f,  0.0f, };
        GLushort     indices[6]    {  0, 1, 2,
                                      0, 2, 3, };

    private:
        GLuint       program;
        GLuint       effect_program;
        GLuint       texture;
        GLuint       src_fbo;
        GLuint       src_fbo_texture;
        GLuint       dst_fbo;
        GLuint       dst_fbo_texture;
    };


    /*
     *
     */
    class ImageRenderer {
    public:
        ImageRenderer(std::string &e_name): width(0), height(0), drawQ() {
            log_d("ImageRenderer created.");
            paint = new ImagePaint(std::forward<std::string>(e_name));
        }

        ~ImageRenderer() {
            delete paint;
            log_d("ImageRenderer release.");
        }

    public:
        /**
         * run in renderer thread.
         */
        void surfaceCreated() {
        }

        /**
         * run in renderer thread.
         */
        void surfaceDestroyed() {
            delete paint;
            paint = nullptr;
        }

        /**
         * run in renderer thread.
         */
        void surfaceChanged(int32_t w, int32_t h) {
            width = w; height = h;
            if (paint != nullptr) {
                paint->updateSize(w, h);
            }
        }

        /**
         * update effect paint
         */
        void updatePaint(std::string &e_name) {
            clearFrame();
            delete paint;
            paint = new ImagePaint(std::forward<std::string>(e_name));
            paint->updateSize(width, height);
        }

        /**
         * run in caller thread.
         * append frm to frameQ.
         */
        void appendFrame(ImageFrame &&frm) {
            if (frm.available()) {
                drawQ.enqueue(std::forward<ImageFrame>(frm));
            }
        }

        /**
         * run in renderer thread.
         * read frm from frameQ and draw.
         */
        void drawFrame() {
            ImageFrame nf;
            drawQ.try_dequeue(nf);
            if (paint != nullptr) {
                paint->draw(nf);
            }
        }


    public:
        /*
         *
         */
        int32_t getWidth() const { return width; }
        int32_t getHeight() const { return height; }

    private:
        void clearFrame() {
            ImagePaint *t = paint;
            paint = nullptr;
            ImageFrame f;
            while (drawQ.try_dequeue(f));
            paint = t;
        }

    private:
        ImageRenderer(ImageRenderer&&) = delete;
        ImageRenderer(const ImageRenderer&) = delete;
        ImageRenderer& operator=(ImageRenderer&&) = delete;
        ImageRenderer& operator=(const ImageRenderer&) = delete;

    private:
        int32_t     width;
        int32_t     height;
        ImagePaint *paint;
        ImageQueue  drawQ;
    };


    /*
     *
     */
    enum class CameraState {
        None,
        Previewing,
    };


    /*
     *
     */
    enum class CameraMerge {
        Single,
        Vertical,
        Chat,
    };


    /*
     *
     */
    class Camera {
    public:
        Camera(std::string &&_id, int32_t fps): yuv_args(), id(_id), state(CameraState::None), dev(nullptr),
                                                fps_req(fps), fps_range(), ori(0), af_mode(ACAMERA_CONTROL_AF_MODE_OFF),
                                                reader(nullptr), window(nullptr), cap_request(nullptr), out_container(nullptr),
                                                out_session(nullptr), cap_session(nullptr), out_target(nullptr),
                                                ds_callbacks({nullptr, onDisconnected, onError}),
                                                css_callbacks({nullptr, onClosed, onReady, onActive}) {
            log_d("Camera[%s] created.", id.c_str());
        }

        ~Camera() {
            close();
            log_d("Camera[%s] release.", id.c_str());
        }

    public:
        /**
         * check lc.id == rc.id
         * @param lc left camera
         * @param rc right camera
         * @return true: lc.id == rc.id
         */
        static bool equal(const Camera &lc, const Camera &rc) {
            return lc.id == rc.id;
        }

        /**
         * enumerate all cameras
         * @param cams all cameras
         */
        static void enumerate(std::vector<std::shared_ptr<Camera>> &cams, std::vector<std::string> &ids) {
            ACameraManager *manager = ACameraManager_create();
            if (manager == nullptr) {
                return;
            }

            camera_status_t status;
            ACameraIdList *cameraIdList = nullptr;
            status = ACameraManager_getCameraIdList(manager, &cameraIdList);
            if (status != ACAMERA_OK) {
                log_e("Failed to get camera id list (reason: %d).", status);
                ACameraManager_delete(manager);
                return;
            }

            if (cameraIdList == nullptr || cameraIdList->numCameras < 1) {
                log_e("No camera device detected.");
                if (cameraIdList)
                    ACameraManager_deleteCameraIdList(cameraIdList);
                ACameraManager_delete(manager);
                return;
            }

            cams.clear();
            if (ids.empty()) {
                for (int32_t i = 0; i < cameraIdList->numCameras; i++) {
                    cams.push_back(std::make_shared<Camera>(cameraIdList->cameraIds[i], 30));
                }
            } else {
                for (const auto& d : ids) {
                    bool has = false;
                    for (int32_t i = 0; i < cameraIdList->numCameras; i++) {
                        std::string id(cameraIdList->cameraIds[i]);
                        if (d == id) { has = true;break; }
                    }
                    if (has) cams.push_back(std::make_shared<Camera>(std::string(d), 30));
                }
            }

            ACameraManager_delete(manager);
        }

    public:
        /**
         * check _id == camera.id
         * @param _id
         * @return true: _id == camera.id
         */
        bool equal(const std::string &&_id) {
            return _id == id;
        }

        /**
         * get latest image from camera
         * call after {@link preview}
         * @param frame [out] latest image frame
         */
        bool getLatestImage(ImageFrame &frame) {
            frame.setOrientation(ori);
            media_status_t status = AImageReader_acquireLatestImage(reader, &yuv_args.image);
            if (status != AMEDIA_OK) {
                return false;
            }

            status = AImage_getFormat(yuv_args.image, &yuv_args.format);
            if (status != AMEDIA_OK || yuv_args.format != AIMAGE_FORMAT_YUV_420_888) {
                AImage_delete(yuv_args.image);
                return false;
            }

            status = AImage_getNumberOfPlanes(yuv_args.image, &yuv_args.plane_count);
            if (status != AMEDIA_OK || yuv_args.plane_count != 3) {
                AImage_delete(yuv_args.image);
                return false;
            }

            AImage_getPlaneRowStride(yuv_args.image, 0, &yuv_args.y_stride);
            AImage_getPlaneRowStride(yuv_args.image, 1, &yuv_args.v_stride);
            AImage_getPlaneRowStride(yuv_args.image, 1, &yuv_args.u_stride);
            AImage_getPlaneData(yuv_args.image, 0, &yuv_args.y_pixel, &yuv_args.y_len);
            AImage_getPlaneData(yuv_args.image, 1, &yuv_args.v_pixel, &yuv_args.v_len);
            AImage_getPlaneData(yuv_args.image, 2, &yuv_args.u_pixel, &yuv_args.u_len);
            AImage_getPlanePixelStride(yuv_args.image, 1, &yuv_args.vu_pixel_stride);

            AImage_getCropRect(yuv_args.image, &yuv_args.src_rect);
            yuv_args.src_w = yuv_args.src_rect.right - yuv_args.src_rect.left;
            yuv_args.src_h = yuv_args.src_rect.bottom - yuv_args.src_rect.top;

            yuv_args.argb_pixel = (uint8_t *)malloc(sizeof(uint8_t) * yuv_args.src_w * yuv_args.src_h * 4);
            if (yuv_args.argb_pixel == nullptr) {
                AImage_delete(yuv_args.image);
                return false;
            }

            yuv_args.dst_argb_pixel = (uint8_t *)malloc(sizeof(uint8_t) * yuv_args.src_w * yuv_args.src_h * 4);
            if (yuv_args.dst_argb_pixel == nullptr) {
                free(yuv_args.argb_pixel);
                AImage_delete(yuv_args.image);
                return false;
            }

            if (yuv_args.ori == 90 || yuv_args.ori == 270) {
                yuv_args.img_width = yuv_args.src_h;
                yuv_args.img_height = yuv_args.src_w;
            } else {
                yuv_args.img_width = yuv_args.src_w;
                yuv_args.img_height = yuv_args.src_h;
            }

            frame.get(&yuv_args.frame_w, &yuv_args.frame_h, &yuv_args.frame_cache);
            yuv_args.wof = (yuv_args.frame_w - yuv_args.img_width) / 2;
            yuv_args.hof = (yuv_args.frame_h - yuv_args.img_height) / 2;
            yuv2argb(yuv_args);
            AImage_delete(yuv_args.image);
            free(yuv_args.argb_pixel);
            free(yuv_args.dst_argb_pixel);
            yuv_args.image = nullptr;
            return true;
        }

    public:
        /**
         * start camera preview
         * @param req_w requested image width
         * @param req_h requested image height
         * @param out_fps [out] camera fps
         * @return true: start preview success
         */
        bool preview(int32_t req_w, int32_t req_h, int32_t *out_fps) {
            if (dev || state == CameraState::Previewing) {
                log_e("Camera[%s] device is running.", id.c_str());
                return false;
            }

            if (id.empty()) {
                release();
                return false;
            }

            ACameraManager *mgr = ACameraManager_create();
            if (mgr == nullptr) {
                release();
                return false;
            }

            camera_status_t s;
            media_status_t ms;
            ACameraMetadata *metadata = nullptr;
            s = ACameraManager_getCameraCharacteristics(mgr, id.c_str(), &metadata);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to get camera meta data.", id.c_str());
                ACameraManager_delete(mgr);
                release();
                return false;
            }

            getFps(metadata);
//            log_d("Camera[%s] preview fps: %d,%d.", id.c_str(), fps_range[0], fps_range[1]);
            getOrientation(metadata);
//            log_d("Camera[%s] preview sensor orientation: %d.", id.c_str(), ori);
            getAfMode(metadata);
//            log_d("Camera[%s] select af mode: %d.", id.c_str(), af_mode);
            int32_t width = 0, height = 0;
            getSize(metadata, req_w, req_h, &width, &height);
//            log_d("Camera[%s] preview size: %d,%d.", id.c_str(), width, height);

            if (width <= 0 || height <= 0) {
                ACameraMetadata_free(metadata);
                ACameraManager_delete(mgr);
                release();
                return false;
            }

            if (reader) {
                AImageReader_setImageListener(reader, nullptr);
                AImageReader_delete(reader);
                ACameraMetadata_free(metadata);
                ACameraManager_delete(mgr);
                release();
                return false;
            }

            ms = AImageReader_new(width, height,
                                  AIMAGE_FORMAT_YUV_420_888, 2, &reader);
            if (ms != AMEDIA_OK) {
                log_e("Camera[%s] Failed to new image reader.", id.c_str());
                ACameraMetadata_free(metadata);
                ACameraManager_delete(mgr);
                release();
                return false;
            }

            if (window) {
                ACameraMetadata_free(metadata);
                ACameraManager_delete(mgr);
                release();
                return false;
            }

            ms = AImageReader_getWindow(reader, &window);
            if (ms != AMEDIA_OK) {
                log_e("Camera[%s] Failed to get native window.", id.c_str());
                ACameraMetadata_free(metadata);
                ACameraManager_delete(mgr);
                release();
                return false;
            }

            ANativeWindow_acquire(window);
            s = ACameraManager_openCamera(mgr, id.c_str(), &ds_callbacks, &dev);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed[%d] to open camera device.", id.c_str(), s);
                ACameraMetadata_free(metadata);
                ACameraManager_delete(mgr);
                release();
                return false;
            }

            s = ACameraDevice_createCaptureRequest(dev, TEMPLATE_RECORD, &cap_request);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to create capture request.", id.c_str());
                ACameraMetadata_free(metadata);
                ACameraManager_delete(mgr);
                release();
                return false;
            }

//            log_d("Camera[%s] Success to create capture request.", id.c_str());
            s = ACaptureSessionOutputContainer_create(&out_container);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to create session output container.", id.c_str());
                ACameraMetadata_free(metadata);
                ACameraManager_delete(mgr);
                release();
                return false;
            }

            // delete / release
            ACameraMetadata_free(metadata);
            ACameraManager_delete(mgr);
//            log_d("Camera[%s] Success to open camera device.", id.c_str());

            s = ACameraOutputTarget_create(window, &out_target);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to create CameraOutputTarget.", id.c_str());
                release();
                return false;
            }

            s = ACaptureRequest_addTarget(cap_request, out_target);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to add CameraOutputTarget.", id.c_str());
                release();
                return false;
            }

            s = ACaptureSessionOutput_create(window, &out_session);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to create CaptureSessionOutput.", id.c_str());
                release();
                return false;
            }

            s = ACaptureSessionOutputContainer_add(out_container, out_session);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to add CaptureSessionOutput.", id.c_str());
                release();
                return false;
            }

            s = ACameraDevice_createCaptureSession(dev, out_container, &css_callbacks, &cap_session);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed[%d] to create CaptureSession.", id.c_str(), s);
                release();
                return false;
            }

            s = ACaptureRequest_setEntry_u8(cap_request, ACAMERA_CONTROL_AF_MODE, 1, &af_mode);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to set af mode.", id.c_str());
            }
            s = ACaptureRequest_setEntry_i32(cap_request, ACAMERA_CONTROL_AE_TARGET_FPS_RANGE, 2, fps_range);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to set fps.", id.c_str());
            }
            if (ori == 270) {
                uint8_t scene = ACAMERA_CONTROL_SCENE_MODE_FACE_PRIORITY;
                s = ACaptureRequest_setEntry_u8(cap_request, ACAMERA_CONTROL_SCENE_MODE, 1, &scene);
                if (s != ACAMERA_OK) {
                    log_e("Camera[%s] Failed to set scene mode.", id.c_str());
                }
            } else {
                uint8_t scene = ACAMERA_CONTROL_SCENE_MODE_DISABLED;
                s = ACaptureRequest_setEntry_u8(cap_request, ACAMERA_CONTROL_SCENE_MODE, 1, &scene);
                if (s != ACAMERA_OK) {
                    log_e("Camera[%s] Failed to set scene mode.", id.c_str());
                }
            }

            s = ACameraCaptureSession_setRepeatingRequest(cap_session, nullptr, 1, &cap_request, nullptr);
            if (s != ACAMERA_OK) {
                log_e("Camera[%s] Failed to set RepeatingRequest.", id.c_str());
                release();
                return false;
            }

            if (out_fps) *out_fps = fps_range[0];
            state = CameraState::Previewing;
            log_d("Camera[%s] Success to start preview: %d-%d,%d,%d r(%d,%d) p(%d,%d).",
                  id.c_str(), fps_range[0], fps_range[1], ori, af_mode, req_w, req_h, width, height);
            return true;
        }

        /*
         *
         */
        bool previewing() const {
            return state == CameraState::Previewing;
        }

        /**
         * close camera preview
         */
        void close() {
            release();
            if (state != CameraState::None) {
                state = CameraState::None;
                log_d("Camera[%s] Success to close CameraDevice.", id.c_str());
            }
        }

    private:
        Camera(Camera&&) = delete;
        Camera(const Camera&) = delete;
        Camera& operator=(Camera&&) = delete;
        Camera& operator=(const Camera&) = delete;

    private:
        void release() {
            if (cap_request) {
                ACaptureRequest_free(cap_request);
                cap_request = nullptr;
            }

            if (dev) {
                ACameraDevice_close(dev);
                dev = nullptr;
            }

            if (out_session) {
                ACaptureSessionOutput_free(out_session);
                out_session = nullptr;
            }

            if (out_container) {
                ACaptureSessionOutputContainer_free(out_container);
                out_container = nullptr;
            }

            if (reader) {
                AImageReader_setImageListener(reader, nullptr);
                AImageReader_delete(reader);
                reader = nullptr;
            }

            if (window) {
                ANativeWindow_release(window);
                window = nullptr;
            }
        }

        void getFps(ACameraMetadata *metadata) {
            if (metadata == nullptr) {
                return;
            }

            camera_status_t status;
            ACameraMetadata_const_entry entry;
            status = ACameraMetadata_getConstEntry(metadata, ACAMERA_CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES, &entry);
            if (status != ACAMERA_OK) {
                return;
            }

            bool found = false;
            int32_t current_best_match = -1;
            for (int32_t i = 0; i < entry.count; i++) {
                int32_t min = entry.data.i32[i * 2 + 0];
                int32_t max = entry.data.i32[i * 2 + 1];
                if (fps_req == max) {
                    if (min == max) {
                        fps_range[0] = min;
                        fps_range[1] = max;
                        found = true;
                    } else if (current_best_match >= 0) {
                        int32_t current_best_match_min = entry.data.i32[current_best_match * 2 + 0];
                        if (min > current_best_match_min) {
                            current_best_match = i;
                        }
                    } else {
                        current_best_match = i;
                    }
                }
            }

            if (!found) {
                if (current_best_match >= 0) {
                    fps_range[0] = entry.data.i32[current_best_match * 2 + 0];
                    fps_range[1] = entry.data.i32[current_best_match * 2 + 1];
                } else {
                    fps_range[0] = entry.data.i32[0];
                    fps_range[1] = entry.data.i32[1];
                }
            }
        }

        void getOrientation(ACameraMetadata *metadata) {
            if (metadata == nullptr) {
                return;
            }

            camera_status_t status;
            ACameraMetadata_const_entry entry;
            status = ACameraMetadata_getConstEntry(metadata, ACAMERA_SENSOR_ORIENTATION, &entry);
            if (status != ACAMERA_OK) {
                return;
            }

            ori = entry.data.i32[0];
            yuv_args.ori = ori;
        }

        void getAfMode(ACameraMetadata *metadata) {
            if (metadata == nullptr) {
                return;
            }

            camera_status_t status;
            ACameraMetadata_const_entry entry;
            status = ACameraMetadata_getConstEntry(metadata, ACAMERA_CONTROL_AF_AVAILABLE_MODES, &entry);
            if (status != ACAMERA_OK) {
                return;
            }

            if (entry.count <= 0) {
                af_mode = ACAMERA_CONTROL_AF_MODE_OFF;
            } else if (entry.count == 1) {
                af_mode = entry.data.u8[0];
            } else {
                uint8_t af_a = 0, af_b = 0;
                for (int32_t i = 0; i < entry.count; i++) {
                    if (entry.data.u8[i] == ACAMERA_CONTROL_AF_MODE_CONTINUOUS_VIDEO) {
                        af_a = ACAMERA_CONTROL_AF_MODE_CONTINUOUS_VIDEO;
                    } else if (entry.data.u8[i] == ACAMERA_CONTROL_AF_MODE_AUTO) {
                        af_b = ACAMERA_CONTROL_AF_MODE_AUTO;
                    }
                }
                if (af_a != 0) {
                    af_mode = af_a;
                } else if (af_b != 0) {
                    af_mode = af_b;
                } else {
                    af_mode = ACAMERA_CONTROL_AF_MODE_OFF;
                }
            }
        }

    private:
        static void getSize(ACameraMetadata *metadata,
                            int32_t req_w, int32_t req_h,
                            int32_t *out_w, int32_t *out_h) {
            if (metadata == nullptr) {
                return;
            }

            camera_status_t status;
            ACameraMetadata_const_entry entry;
            status = ACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry);
            if (status != ACAMERA_OK) {
                return;
            }

            int32_t w, h, sub, min = 6000;
            for (int32_t i = 0; i < entry.count; i += 4) {
                int32_t input = entry.data.i32[i + 3];
                int32_t format = entry.data.i32[i + 0];
                if (input) {
                    continue;
                }

                if (format == AIMAGE_FORMAT_YUV_420_888 || format == AIMAGE_FORMAT_JPEG) {
                    w = entry.data.i32[i * 4 + 1];
                    h = entry.data.i32[i * 4 + 2];
                    if (w == 0 || h == 0 || w > 6000 || h > 6000 || w < 200 || h < 200 ||
                        w*h > 6000000 || w*h < 1000000) {
                        continue;
                    }
                    sub = w - req_h;
                    if (sub >= 0 && sub < min) {
                        min = sub;
                        *out_w = w;
                        *out_h = h;
                    }
                }
            }

            if (*out_w == 0 || *out_h == 0) {
                *out_w = req_h;
                *out_h = req_w;
            }
        }

        static bool yuv2argb(yuv_args &yuv_args) {
            int32_t res = libyuv::Android420ToARGB(yuv_args.y_pixel, yuv_args.y_stride,
                                                   yuv_args.u_pixel, yuv_args.u_stride,
                                                   yuv_args.v_pixel, yuv_args.v_stride,
                                                   yuv_args.vu_pixel_stride, yuv_args.argb_pixel,
                                                   yuv_args.src_w * 4,
                                                   yuv_args.src_w, yuv_args.src_h);
            if (res != 0) {
                return false;
            }

            libyuv::RotationModeEnum r;
            if (yuv_args.ori == 90) {
                r = libyuv::RotationModeEnum ::kRotate90;
            } else if (yuv_args.ori == 180) {
                r = libyuv::RotationModeEnum ::kRotate180;
            } else if (yuv_args.ori == 270) {
                r = libyuv::RotationModeEnum ::kRotate270;
            } else {
                r = libyuv::RotationModeEnum ::kRotate0;
            }

            res = libyuv::ARGBRotate(yuv_args.argb_pixel, yuv_args.src_w * 4,
                                     yuv_args.dst_argb_pixel, yuv_args.img_width * 4,
                                     yuv_args.src_w, yuv_args.src_h, r);
            if (res != 0) {
                return false;
            }

            if (yuv_args.wof >= 0 && yuv_args.hof >= 0) {
                for (int32_t i = 0; i < yuv_args.img_height; i++) {
                    memcpy(yuv_args.frame_cache + ((i + yuv_args.hof) * yuv_args.frame_w + yuv_args.wof),
                           yuv_args.dst_argb_pixel + (i * yuv_args.img_width) * 4,
                           sizeof(uint8_t) * yuv_args.img_width * 4);
                }
            } else if (yuv_args.wof < 0 && yuv_args.hof >= 0) {
                for (int32_t i = 0; i < yuv_args.img_height; i++) {
                    memcpy(yuv_args.frame_cache + ((i + yuv_args.hof) * yuv_args.frame_w),
                           yuv_args.dst_argb_pixel + (i * yuv_args.img_width - yuv_args.wof) * 4,
                           sizeof(uint8_t) * yuv_args.frame_w * 4);
                }
            } else if (yuv_args.wof >= 0 && yuv_args.hof < 0) {
                for (int32_t i = 0; i < yuv_args.frame_h; i++) {
                    memcpy(yuv_args.frame_cache + (i * yuv_args.frame_w + yuv_args.wof),
                           yuv_args.dst_argb_pixel + ((i - yuv_args.hof) * yuv_args.img_width) * 4,
                           sizeof(uint8_t) * yuv_args.img_width * 4);
                }
            } else if (yuv_args.wof < 0 && yuv_args.hof < 0) {
                for (int32_t i = 0; i < yuv_args.frame_h; i++) {
                    memcpy(yuv_args.frame_cache + (i * yuv_args.frame_w),
                           yuv_args.dst_argb_pixel + ((i - yuv_args.hof) * yuv_args.img_width - yuv_args.wof) * 4,
                           sizeof(uint8_t) * yuv_args.frame_w * 4);
                }
            }

            return true;
        }

        static void onDisconnected(void *context, ACameraDevice *device) {
        }

        static void onError(void *context, ACameraDevice *device, int error) {
        }

        static void onActive(void *context, ACameraCaptureSession *session) {
        }

        static void onReady(void *context, ACameraCaptureSession *session) {
        }

        static void onClosed(void *context, ACameraCaptureSession *session) {
        }

    private:
        yuv_args yuv_args;
        std::string id;
        std::atomic<CameraState> state;
        ACameraDevice *dev;

    private:
        int32_t fps_req;
        int32_t fps_range[2];
        int32_t ori;
        uint8_t af_mode;

    private:
        AImageReader *reader;
        ANativeWindow *window;
        ACaptureRequest *cap_request;
        ACaptureSessionOutputContainer *out_container;
        ACaptureSessionOutput *out_session;
        ACameraCaptureSession *cap_session;
        ACameraOutputTarget *out_target;
        ACameraDevice_StateCallbacks ds_callbacks;
        ACameraCaptureSession_stateCallbacks css_callbacks;
    };


    /*
     *
     */
    class Audio {
    public:
        explicit Audio(uint32_t cls = 2,
                       uint32_t spr = 44100): eng_obj(nullptr), eng_eng(nullptr),
                                              rec_obj(nullptr), rec_eng(nullptr), rec_queue(nullptr),
                                              channels(cls<=1?1:2), sampling_rate(spr==44100?SL_SAMPLINGRATE_44_1:SL_SAMPLINGRATE_16),
                                              sample_rate(sampling_rate / 1000), pcm_data((uint8_t*)malloc(sizeof(uint8_t)*(PCM_BUF_SIZE))),
                                              frm_size(1024*2*channels), frm_changed(false), cache(frm_size), frame(frm_size),
                                              frame_callback(nullptr), frame_ctx(nullptr) {
            log_d("Audio[%d,%d] created.", channels, sample_rate);
            initObjects();
        }

        ~Audio() {
            if (rec_obj) {
                (*rec_obj)->Destroy(rec_obj);
                rec_obj = nullptr;
            }
            if (eng_obj) {
                (*eng_obj)->Destroy(eng_obj);
                eng_obj = nullptr;
            }
            if (pcm_data) {
                free(pcm_data);
                pcm_data = nullptr;
            }
            log_d("Audio[%d,%d] release.", channels, sample_rate);
        }

    public:
        /**
         * @return true: audio recorder recording
         */
        bool recording() const {
            if (!recordable()) {
                return false;
            }
            SLuint32 state;
            SLresult res = (*rec_eng)->GetRecordState(rec_eng, &state);
            return res == SL_RESULT_SUCCESS && state == SL_RECORDSTATE_RECORDING;
        }
        /**
         * start audio record
         * @return true: start success
         */
        bool startRecord(void (*frm_callback)(void *) = nullptr, void *ctx = nullptr) {
            if (!recordable()) {
                return false;
            }
            if (!enqueue(false)) {
                return false;
            }
            SLresult res = (*rec_eng)->SetRecordState(rec_eng, SL_RECORDSTATE_RECORDING);
            if (res != SL_RESULT_SUCCESS) {
                return false;
            }
            frame_callback = frm_callback;
            frame_ctx = ctx;
            log_d("Audio[%d,%d] start record.", channels, sample_rate);
            return true;
        }
        /**
         * stop audio record
         */
        void stopRecord() {
            if (!recording()) {
                return;
            }
            (*rec_eng)->SetRecordState(rec_eng, SL_RECORDSTATE_STOPPED);
            log_d("Audio[%d,%d] stop record.", channels, sample_rate);
        }
        /**
         * collect an audio frame
         * @param changed true: audio frame data cache changed
         * @return collect success audio_frame, all return audio_frame is same address
         */
        bool collectFrame(AudioFrame &frm, bool *changed = nullptr) {
            bool chg = frm_changed;
            if (chg) {
                frm = frame;
            }
            if (changed != nullptr) {
                *changed = frm_changed;
            }
            frm_changed = false;
            return chg;
        }

    public:
        /**
         * @return pcm channels num
         */
        uint32_t getChannels() const { return channels; }
        /**
         * @return pcm sample rate
         */
        uint32_t getSampleRate() const { return sample_rate; }
        /**
         * @return audio frame data size
         */
        uint32_t getFrameSize() const { return frm_size; }

    private:
        void initObjects() {
            SLresult res;
            SLmillisecond period;
            res = slCreateEngine(&eng_obj, 0, nullptr, 0, nullptr, nullptr);
            if (res != SL_RESULT_SUCCESS) {
                log_e("Audio[%d,%d] create eng obj fail. %d.", channels, sample_rate, res);
                return;
            }
            res = (*eng_obj)->Realize(eng_obj, SL_BOOLEAN_FALSE);
            if (res != SL_RESULT_SUCCESS) {
                (*eng_obj)->Destroy(eng_obj);
                eng_obj = nullptr;
                log_e("Audio[%d,%d] realize eng obj fail. %d.", channels, sample_rate, res);
                return;
            }
            res = (*eng_obj)->GetInterface(eng_obj, SL_IID_ENGINE, &eng_eng);
            if (res != SL_RESULT_SUCCESS) {
                (*eng_obj)->Destroy(eng_obj);
                eng_obj = nullptr;
                log_e("Audio[%d,%d] get eng eng fail. %d.", channels, sample_rate, res);
                return;
            }
            SLDataLocator_IODevice ioDevice = {
                    SL_DATALOCATOR_IODEVICE,
                    SL_IODEVICE_AUDIOINPUT,
                    SL_DEFAULTDEVICEID_AUDIOINPUT,
                    nullptr
            };
            SLDataSource dataSrc = { &ioDevice, nullptr };
            SLDataLocator_AndroidSimpleBufferQueue bufferQueue = {
                    SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE, 20 };
            SLDataFormat_PCM formatPcm = {
                    SL_DATAFORMAT_PCM, channels, sampling_rate,
                    SL_PCMSAMPLEFORMAT_FIXED_16, SL_PCMSAMPLEFORMAT_FIXED_16,
                    channels==1?SL_SPEAKER_FRONT_CENTER:SL_SPEAKER_FRONT_LEFT|SL_SPEAKER_FRONT_RIGHT,
                    SL_BYTEORDER_LITTLEENDIAN
            };
            SLDataSink audioSink = { &bufferQueue, &formatPcm };
            const SLInterfaceID iid[] = { SL_IID_ANDROIDSIMPLEBUFFERQUEUE, SL_IID_ANDROIDCONFIGURATION };
            const SLboolean req[] = { SL_BOOLEAN_TRUE, SL_BOOLEAN_TRUE };
            res = (*eng_eng)->CreateAudioRecorder(eng_eng, &rec_obj, &dataSrc, &audioSink, 2, iid, req);
            if (res != SL_RESULT_SUCCESS) {
                (*eng_obj)->Destroy(eng_obj);
                eng_obj = nullptr;
                eng_eng = nullptr;
                log_e("Audio[%d,%d] create audio recorder fail. %d.", channels, sample_rate, res);
                return;
            }
            res = (*rec_obj)->Realize(rec_obj, SL_BOOLEAN_FALSE);
            if (res != SL_RESULT_SUCCESS) {
                (*eng_obj)->Destroy(eng_obj);
                eng_obj = nullptr;
                eng_eng = nullptr;
                (*rec_obj)->Destroy(rec_obj);
                rec_obj = nullptr;
                log_e("Audio[%d,%d] realize audio recorder fail. %d.", channels, sample_rate, res);
                return;
            }
            res = (*rec_obj)->GetInterface(rec_obj, SL_IID_RECORD, &rec_eng);
            if (res != SL_RESULT_SUCCESS) {
                (*eng_obj)->Destroy(eng_obj);
                eng_obj = nullptr;
                eng_eng = nullptr;
                (*rec_obj)->Destroy(rec_obj);
                rec_obj = nullptr;
                log_e("Audio[%d,%d] get audio recorder fail. %d.", channels, sample_rate, res);
                return;
            }
            (*rec_eng)->SetPositionUpdatePeriod(rec_eng, PERIOD_TIME);
            (*rec_eng)->GetPositionUpdatePeriod(rec_eng, &period);
//            log_d("Audio[%d,%d] period millisecond: %dms", channels, sample_rate, period);
            res = (*rec_obj)->GetInterface(rec_obj, SL_IID_ANDROIDSIMPLEBUFFERQUEUE, &rec_queue);
            if (res != SL_RESULT_SUCCESS) {
                (*eng_obj)->Destroy(eng_obj);
                eng_obj = nullptr;
                eng_eng = nullptr;
                (*rec_obj)->Destroy(rec_obj);
                rec_obj = nullptr;
                log_e("Audio[%d,%d] get audio recorder queue fail. %d.", channels, sample_rate, res);
                return;
            }
            res = (*rec_queue)->RegisterCallback(rec_queue, queueCallback, this);
            if (res != SL_RESULT_SUCCESS) {
                (*eng_obj)->Destroy(eng_obj);
                eng_obj = nullptr;
                eng_eng = nullptr;
                (*rec_obj)->Destroy(rec_obj);
                rec_obj = nullptr;
                log_e("Audio[%d,%d] queue register callback fail. %d.", channels, sample_rate, res);
                return;
            }
//            log_d("Audio[%d,%d] init success.", channels, sample_rate);
        }

        bool recordable() const {
            return rec_obj   != nullptr &&
                   rec_eng   != nullptr &&
                   rec_queue != nullptr;
        }

        bool enqueue(bool chk_recording) {
            if (chk_recording && !recording()) {
                return false;
            }
            SLresult res = (*rec_queue)->Enqueue(rec_queue, pcm_data, PCM_BUF_SIZE);
            return res == SL_RESULT_SUCCESS;
        }

    private:
        void handleFrame() {
            if (cache.offset + PCM_BUF_SIZE > cache.size) {
                memcpy(frame.cache, cache.cache, sizeof(uint8_t) * cache.offset);
                int32_t c = cache.size - cache.offset;
                memcpy(frame.cache + cache.offset, pcm_data, sizeof(uint8_t) * c);
                frm_changed = true;
                cache.offset = PCM_BUF_SIZE - c;
                memcpy(cache.cache, pcm_data + c, sizeof(uint8_t) * cache.offset);
                if (frame_callback != nullptr) frame_callback(frame_ctx);
            } else {
                memcpy(cache.cache + cache.offset, pcm_data, sizeof(uint8_t) * PCM_BUF_SIZE);
                cache.offset += PCM_BUF_SIZE;
            }
        }

    private:
        static void queueCallback(SLAndroidSimpleBufferQueueItf queue, void *ctx) {
            auto *rec = (Audio*)ctx;
            rec->handleFrame();
            rec->enqueue(true);
        }

    private:
        Audio(Audio&&) = delete;
        Audio(const Audio&) = delete;
        Audio& operator=(Audio&&) = delete;
        Audio& operator=(const Audio&) = delete;

    private:
        // PCM Size=采样率*采样时间*采样位深/8*通道数（Bytes）
        const int32_t PERIOD_TIME  = 10;  // 10ms
        const int32_t PCM_BUF_SIZE = 320; // 320bytes

    private:
        SLObjectItf eng_obj;
        SLEngineItf eng_eng;
        SLObjectItf rec_obj;
        SLRecordItf rec_eng;
        SLAndroidSimpleBufferQueueItf rec_queue;

    private:
        uint32_t channels;
        SLuint32 sampling_rate;
        uint32_t sample_rate;
        uint8_t *pcm_data;

    private:
        uint32_t         frm_size;
        std::atomic_bool frm_changed;
        AudioFrame       cache;
        AudioFrame       frame;

    private:
        void (*frame_callback)(void *);
        void *frame_ctx;
    };


    /*
     *
     */
    class Recorder {
    public:
        explicit Recorder(const std::string &&cms): cams(cms), cameras(), camFrames(),
                                                    camWidth(0), camHeight(0) {
            log_d("Recorder[%s] created.", cams.c_str());
            CollectRunnable = false;
            std::regex re{ "," };
            std::vector<std::string> ids {
                std::sregex_token_iterator(cams.begin(), cams.end(), re, -1),
                std::sregex_token_iterator()};
            Camera::enumerate(cameras, ids);
            audio = std::make_shared<Audio>();
        }

        ~Recorder() {
            CollectRunnable = false;
            for (auto& camera : cameras) { camera.reset(); }
            for (auto& camFrame : camFrames) { camFrame.reset(); }
            std::vector<std::shared_ptr<Camera>> ec; cameras.swap(ec);
            std::vector<std::shared_ptr<ImageFrame>> ef; camFrames.swap(ef);
            audio.reset();
            log_d("Recorder[%s] release.", cams.c_str());
        }

    public:
        void startPreview(int32_t width, int32_t height, CameraMerge merge = CameraMerge::Single) {
            if (CollectRunnable) return;
            camWidth = width / 2; camHeight = height / 2;
            for (auto& camFrame : camFrames) { camFrame.reset(); }
            std::vector<std::shared_ptr<ImageFrame>> ef; camFrames.swap(ef);
            if (camFrames.empty()) {
                for (int32_t i = 0; i < cameras.size(); i++) {
                    camFrames.push_back(std::make_shared<ImageFrame>(camWidth, camHeight));
                }
            }
            int32_t fps = 0;
            if (merge == CameraMerge::Single) {
                const auto &camera = cameras.front();
                if (!camera->previewing()) {
                    camera->preview(camWidth, camHeight, &fps);
                }
            } else {
                for (const auto &camera : cameras) {
                    if (!camera->previewing()) {
                        camera->preview(camWidth, camHeight, &fps);
                    }
                }
            }
            fps = fps <= 0 ? 30 : fps;
            auto fps_ms = (int32_t)(1000.0f / fps);
            if (audio != nullptr) audio->startRecord(collectAudio, this);
            CollectRunnable = true;
            std::thread ct(Recorder::collectRunnable, this, fps_ms, merge);
            ct.detach();
        }

        void stopPreview() {
            CollectRunnable = false;
            for (const auto& camera : cameras) {
                if (camera->previewing()) camera->close();
            }
            if (audio != nullptr) audio->stopRecord();
        }

    private:
        static void collectRunnable(Recorder *recorder, int32_t fps_ms, CameraMerge merge) {
            log_d("Recorder[%s] collect thread start.", recorder->cams.c_str());
            long ms;
            struct timeval tv{};
            while (CollectRunnable) {
                gettimeofday(&tv, nullptr);
                ms = tv.tv_sec * 1000 + tv.tv_usec / 1000;
                collectCameras(recorder, merge);
                gettimeofday(&tv, nullptr);
                ms = tv.tv_sec * 1000 + tv.tv_usec / 1000 - ms;
                ms = fps_ms - ms;
                if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            }
            log_d("Recorder[%s] collect thread exit.", recorder->cams.c_str());
        }

        static void collectCameras(Recorder *recorder, CameraMerge merge) {
            if (merge == CameraMerge::Single) {
                const auto &camera = recorder->cameras.front();
                if (camera->previewing()) {
                    ImageFrame frame(recorder->camWidth, recorder->camHeight);
                    if (camera->getLatestImage(frame)) {
                        postImageFrame(frame);
                    }
                }
            } else {
                int32_t n = recorder->cameras.size();
                for (const auto &camera : recorder->cameras) {
                    if (!camera->previewing()) n--;
                }
                if (n == 1) {
                    const auto &camera = recorder->cameras.front();
                    if (camera->previewing()) {
                        ImageFrame frame(recorder->camWidth, recorder->camHeight);
                        if (camera->getLatestImage(frame)) {
                            postImageFrame(frame);
                        }
                    }
                } else if (recorder->cameras.size() > 1) {
                    collectMerge(recorder, merge);
                }
            }
        }

        static void collectMerge(Recorder *recorder, CameraMerge merge) {
            int32_t i = 0, n = recorder->cameras.size();
            for (const auto &camera : recorder->cameras) {
                if (!camera->previewing()) n--;
            }
            ImageFrame frame(recorder->camWidth, recorder->camHeight);
            auto *fData = frame.getData();
            for (const auto &camera : recorder->cameras) {
                if (camera->previewing()) {
                    if (camera->getLatestImage(*recorder->camFrames[i])) {
                        if (recorder->camFrames[i]->useMirror()) {
                            auto *data = recorder->camFrames[i]->getData();
                            cv::Mat ot(recorder->camHeight, recorder->camWidth, CV_8UC4, data);
                            cv::flip(ot, ot, 1);
                        }
                    }
                    i++;
                }
            }
            switch (merge) {
                default:
                case CameraMerge::Single:
                case CameraMerge::Vertical: {
                    auto iw = recorder->camWidth, ih = recorder->camHeight / n;
                    auto ic = (recorder->camHeight - ih) / 2;
                    for (i = 0; i < n; i++) {
                        auto *data = recorder->camFrames[i]->getData();
                        memcpy(fData + i * iw * ih, data + iw * ic, sizeof(uint32_t) * iw * ih);
                    }
                } break;
                case CameraMerge::Chat: {
                    auto iw = recorder->camWidth, ih = recorder->camHeight / (n - 1);
                    auto ic = (recorder->camHeight - ih) / 2;
                    for (i = 0; i < (n - 1); i++) {
                        auto *data = recorder->camFrames[i]->getData();
                        memcpy(fData + i * iw * ih, data + iw * ic, sizeof(uint32_t) * iw * ih);
                    }
                    auto *data = recorder->camFrames[n - 1]->getData();
                    cv::Mat rt(recorder->camHeight, recorder->camWidth, CV_8UC4, data);
                    auto dw = (int32_t)(recorder->camWidth  * 0.25f);
                    auto dh = (int32_t)(recorder->camHeight * 0.25f);
                    cv::Mat dt(dh, dw, CV_8UC4);
                    cv::resize(rt, dt, cv::Size(dw, dh));
                    cv::Mat ft(recorder->camHeight, recorder->camWidth, CV_8UC4, fData);
                    cv::Mat roi = ft(cv::Rect(ft.cols - dt.cols - 30, 50, dt.cols, dt.rows));
                    dt.copyTo(roi);
                } break;
            }
            postImageFrame(frame);
        }

        static void collectAudio(void *ctx) {
            auto *recorder = (Recorder*)ctx;
            AudioFrame frame;
            recorder->audio->collectFrame(frame);
//            log_d("Recorder[%s] collect audio %d.", recorder->cams.c_str(), frame.available());
        }

    private:
        Recorder(Recorder&&) = delete;
        Recorder(const Recorder&) = delete;
        Recorder& operator=(Recorder&&) = delete;
        Recorder& operator=(const Recorder&) = delete;

    private:
        std::string                              cams;
        std::vector<std::shared_ptr<Camera>>     cameras;
        std::vector<std::shared_ptr<ImageFrame>> camFrames;
        int32_t                                  camWidth;
        int32_t                                  camHeight;

    private:
        std::shared_ptr<Audio> audio;
    };
} // namespace x


/*
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

static jobject           g_MainClass = nullptr;
static JavaVM           *g_JavaVM    = nullptr;
static x::ImageRenderer *g_Renderer  = nullptr;
static x::Recorder      *g_Recorder  = nullptr;

static void requestGlRender(void *ctx = nullptr) {
    if (g_JavaVM == nullptr || g_MainClass == nullptr) {
        return;
    }

    JNIEnv *p_env = nullptr;
    if (g_JavaVM != nullptr) {
        g_JavaVM->AttachCurrentThread(&p_env, nullptr);
    }

    if (p_env != nullptr && g_MainClass != nullptr) {
        auto mediaClass = (jclass) g_MainClass;
        jmethodID mediaRRID = p_env->GetStaticMethodID(mediaClass, "requestRender", "(I)V");
        if (mediaRRID != nullptr) {
            p_env->CallStaticVoidMethod(mediaClass, mediaRRID, 0);
        }
    }

    if (g_JavaVM != nullptr) {
        g_JavaVM->DetachCurrentThread();
    }
}

static void postGlRender(long delayMillis = 0, void* ctx= nullptr) {
    if (delayMillis <= 0) {
        std::thread t(requestGlRender, ctx);
        t.detach();
    } else {
        std::thread t([](long delayMillis) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMillis));
            requestGlRender();
        }, delayMillis);
        t.detach();
    }
}

static void x::postImageFrame(x::ImageFrame &frame) {
    if (g_Renderer != nullptr) {
        g_Renderer->appendFrame(std::forward<x::ImageFrame>(frame));
        requestGlRender();
    }
}

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_JavaVM = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    g_JavaVM = nullptr;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniInit(
JNIEnv *env, jobject thiz,
jstring fileRootPath) {
    log_d("JNI init.");
    jclass mainClass = env->FindClass("com/scliang/x/camera/CameraManager");
    if (mainClass != nullptr) {
        g_MainClass = env->NewGlobalRef(mainClass);
    }
    const char *file = env->GetStringUTFChars(fileRootPath, nullptr);
    x::FileRoot = new std::string(file);
    env->ReleaseStringUTFChars(fileRootPath, file);
    x::EffectName = new std::string("NONE");
    g_Renderer = new x::ImageRenderer(*x::EffectName);
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniResume(
JNIEnv *env, jobject thiz) {
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniPause(
JNIEnv *env, jobject thiz) {
    if (g_Recorder != nullptr) {
        g_Recorder->stopPreview();
    }
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniRelease(
JNIEnv *env, jobject thiz) {
    if (g_MainClass != nullptr) {
        env->DeleteGlobalRef(g_MainClass);
    }
    g_MainClass = nullptr;
    delete x::FileRoot;
    x::FileRoot = nullptr;
    delete x::EffectName;
    x::EffectName = nullptr;
    delete g_Recorder;
    g_Recorder = nullptr;
    delete g_Renderer;
    g_Renderer = nullptr;
    log_d("JNI release.");
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniSurfaceCreated(
JNIEnv *env, jobject thiz) {
    if (g_Renderer != nullptr) {
        g_Renderer->surfaceCreated();
    }
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniSurfaceChanged(
JNIEnv *env, jobject thiz,
jint width, jint height) {
    if (g_Renderer != nullptr) {
        g_Renderer->surfaceChanged(width, height);
    }
    if (g_Recorder != nullptr && g_Renderer != nullptr &&
        g_Renderer->getWidth() > 0 && g_Renderer->getHeight() > 0) {
        g_Recorder->startPreview(g_Renderer->getWidth(), g_Renderer->getHeight());
    }
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniUpdatePaint(
JNIEnv *env, jobject thiz,
jstring name) {
    const char *en = env->GetStringUTFChars(name, nullptr);
    delete x::EffectName;
    x::EffectName = new std::string(en);
    if (g_Renderer != nullptr) {
        g_Renderer->updatePaint(*x::EffectName);
    }
    env->ReleaseStringUTFChars(name, en);
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniDrawFrame(
JNIEnv *env, jobject thiz) {
    if (g_Renderer != nullptr) {
        g_Renderer->drawFrame();
    }
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_scliang_x_camera_CameraManager_jniPreview(
JNIEnv *env, jobject thiz,
jstring cameras) {
    delete g_Recorder;
    const char *cams = env->GetStringUTFChars(cameras, nullptr);
    g_Recorder = new x::Recorder(std::string(cams));
    if (g_Renderer != nullptr && g_Renderer->getWidth() > 0 && g_Renderer->getHeight() > 0) {
        g_Recorder->startPreview(g_Renderer->getWidth(), g_Renderer->getHeight());
    }
    env->ReleaseStringUTFChars(cameras, cams);
    return 0;
}

#ifdef __cplusplus
}
#endif
