// libdmfront.so, in the launcher app: the game's screen. It listens on the local socket the host (dmhost, which the app starts) connects to, receives the host's hardware
// buffers once, and shows each frame the host announces on the GameActivity's SurfaceView with OpenGL ES (the picture kept at its aspect, centred on black). Controller
// input from the activity goes back to the host, and the game's keyboard requests (DMF_TEXT) go to GameActivity.onTextInput, whose text box answers through
// nativeTextDone. The link is dm/android/host/dm_front_proto.h.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <android/native_window_jni.h>
#include <errno.h>
#include <jni.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include "../host/dm_front_proto.h"

#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "dmfront", __VA_ARGS__)

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static int g_listen = -1, g_fd = -1, g_stop;
static ANativeWindow *g_win; static int g_win_changed;
static AHardwareBuffer *g_ahb[DMF_MAX_BUFFERS]; static int g_nb; static uint32_t g_w, g_h;
static int g_pending = -1;   // the newest frame not shown yet
static _Atomic long long g_frames;   // the host's finished frames (DMF_FRAME), for the frame-rate counter (GameActivity polls nativeFrames)
static int g_gen;            // counts DMF_BUFFERS: a restarted host (the pause menu's Restart) sends new buffers, and the renderer drops the old ones' textures
static pthread_t g_srv, g_ren;
static JavaVM *g_vm; static jobject g_act; static jmethodID g_ontext;   // GameActivity.onTextInput, cached at nativeStart (the server thread calls it for DMF_TEXT)

static void send_msg(const dmf_msg *m) { int fd = g_fd; if (fd >= 0) send(fd, m, sizeof *m, MSG_NOSIGNAL); }
static void send_raw(const void *p, size_t n) { int fd = g_fd; if (fd >= 0) send(fd, p, n, MSG_NOSIGNAL); }
static void release(int i) { if (i >= 0) { dmf_msg m = {DMF_RELEASE, (uint32_t)i}; send_msg(&m); } }

// DMF_TEXT (the server thread): hand the game's keyboard request to GameActivity.onTextInput, which posts its text box to the UI thread (the host's guest thread waits)
static void text_ask(const dmf_text_msg *t) {
    JavaVM *vm = g_vm; if (!vm || !g_act || !g_ontext) return;
    JNIEnv *env = NULL;
    if ((*vm)->AttachCurrentThread(vm, &env, NULL) != JNI_OK) return;
    jstring js[4] = { (*env)->NewStringUTF(env, t->title), (*env)->NewStringUTF(env, t->sub), (*env)->NewStringUTF(env, t->guide), (*env)->NewStringUTF(env, t->initial) };
    (*env)->CallVoidMethod(env, g_act, g_ontext, js[0], js[1], js[2], js[3], (jint)t->max_chars, (jint)t->min_chars);
    if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    for (int i = 0; i < 4; i++) if (js[i]) (*env)->DeleteLocalRef(env, js[i]);
    (*vm)->DetachCurrentThread(vm);
}

static void *server(void *unused) {
    (void)unused;
    for (;;) {
        int fd = accept(g_listen, NULL, NULL);
        if (fd < 0) { if (g_stop) return NULL; continue; }
        LOG("the host connected");
        pthread_mutex_lock(&g_mu); g_fd = fd; pthread_mutex_unlock(&g_mu);
        union { dmf_msg m; dmf_text_msg t; } rb;   // the text messages are much bigger than dmf_msg
        for (;;) {
            ssize_t n = recv(fd, &rb, sizeof rb, 0);
            if (n <= 0) break;
            if (rb.m.type == DMF_TEXT) {   // the game's keyboard: GameActivity shows its text box with the system keyboard (nativeTextDone answers)
                if (n >= (ssize_t)sizeof(rb.t)) text_ask(&rb.t);
                continue;
            }
            if (n != (ssize_t)sizeof(rb.m)) continue;
            dmf_msg m = rb.m;
            if (m.type == DMF_BUFFERS) {   // a new set (the first, or a restarted host's): the old buffers are let go; the renderer's images of them keep their own references until it drops them
                pthread_mutex_lock(&g_mu);
                for (int i = 0; i < DMF_MAX_BUFFERS; i++) if (g_ahb[i]) { AHardwareBuffer_release(g_ahb[i]); g_ahb[i] = NULL; }
                g_nb = 0; g_w = m.width; g_h = m.height; g_pending = -1; g_gen++;
                pthread_mutex_unlock(&g_mu);
            }
            else if (m.type == DMF_BUFFER && m.index < DMF_MAX_BUFFERS) {
                AHardwareBuffer *b = NULL;
                if (AHardwareBuffer_recvHandleFromUnixSocket(fd, &b)) { LOG("buffer %u could not be received", m.index); continue; }
                pthread_mutex_lock(&g_mu); g_ahb[m.index] = b; if ((int)m.index + 1 > g_nb) g_nb = (int)m.index + 1; pthread_mutex_unlock(&g_mu);
            } else if (m.type == DMF_FRAME && m.index < DMF_MAX_BUFFERS) {
                g_frames++;
                pthread_mutex_lock(&g_mu);
                int old = g_pending; g_pending = (int)m.index; pthread_cond_signal(&g_cv);
                pthread_mutex_unlock(&g_mu);
                if (old >= 0 && old != (int)m.index) release(old);   // superseded before it was drawn
            }
        }
        LOG("the host closed the link");
        pthread_mutex_lock(&g_mu); g_fd = -1; close(fd); pthread_mutex_unlock(&g_mu);
    }
}

static const char *k_vs = "attribute vec2 p; varying vec2 uv; void main() { uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); gl_Position = vec4(p, 0.0, 1.0); }";
static const char *k_fs = "precision mediump float; varying vec2 uv; uniform sampler2D t; void main() { gl_FragColor = vec4(texture2D(t, uv).rgb, 1.0); }";
static GLuint shader(GLenum type, const char *src) { GLuint s = glCreateShader(type); glShaderSource(s, 1, &src, NULL); glCompileShader(s); return s; }

static void *renderer(void *unused) {
    (void)unused;
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY); eglInitialize(dpy, NULL, NULL);
    const EGLint ca[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
    EGLConfig cfg; EGLint nc = 0; eglChooseConfig(dpy, ca, &cfg, 1, &nc);
    const EGLint xa[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, xa);
    PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC get_cb = (PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC)eglGetProcAddress("eglGetNativeClientBufferANDROID");
    PFNEGLCREATEIMAGEKHRPROC create_img = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC target_tex = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    EGLSurface surf = EGL_NO_SURFACE; GLuint prog = 0, tex[DMF_MAX_BUFFERS] = {0}; EGLImageKHR img[DMF_MAX_BUFFERS] = {0}; int ntex = 0, shown = -1, gen = 0;
    PFNEGLDESTROYIMAGEKHRPROC destroy_img = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_stop && g_pending < 0 && !g_win_changed) pthread_cond_wait(&g_cv, &g_mu);
        if (g_stop) { pthread_mutex_unlock(&g_mu); break; }
        int frame = g_pending; g_pending = -1;
        int changed = g_win_changed; g_win_changed = 0; ANativeWindow *win = g_win;
        int nb = g_nb, cur = g_gen; AHardwareBuffer *ahb[DMF_MAX_BUFFERS]; memcpy(ahb, g_ahb, sizeof ahb); uint32_t w = g_w, h = g_h;
        pthread_mutex_unlock(&g_mu);
        if (changed) {
            if (surf != EGL_NO_SURFACE) { eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); eglDestroySurface(dpy, surf); surf = EGL_NO_SURFACE; }
            if (win) { surf = eglCreateWindowSurface(dpy, cfg, win, NULL); eglMakeCurrent(dpy, surf, surf, ctx);
                if (!prog) { prog = glCreateProgram(); glAttachShader(prog, shader(GL_VERTEX_SHADER, k_vs)); glAttachShader(prog, shader(GL_FRAGMENT_SHADER, k_fs));
                             glBindAttribLocation(prog, 0, "p"); glLinkProgram(prog); } }
        }
        if (surf == EGL_NO_SURFACE || frame < 0) { if (frame >= 0) release(frame); continue; }
        if (cur != gen && ntex) {   // a new host: its buffers replace the old ones (whose host is gone: nothing to hand back)
            for (int i = 0; i < ntex; i++) { if (tex[i]) glDeleteTextures(1, &tex[i]); if (img[i]) destroy_img(dpy, img[i]); tex[i] = 0; img[i] = NULL; }
            ntex = 0; shown = -1;
        }
        gen = cur;
        if (ntex != nb) {   // the host's buffers as textures (once per host)
            for (int i = 0; i < nb; i++) {
                if (!ahb[i]) continue;
                EGLint ia[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
                img[i] = create_img(dpy, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID, get_cb(ahb[i]), ia);
                glGenTextures(1, &tex[i]); glBindTexture(GL_TEXTURE_2D, tex[i]); target_tex(GL_TEXTURE_2D, (GLeglImageOES)img[i]);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            ntex = nb; LOG("%d buffers of %ux%u as textures", nb, w, h);
        }
        EGLint sw = 0, sh = 0; eglQuerySurface(dpy, surf, EGL_WIDTH, &sw); eglQuerySurface(dpy, surf, EGL_HEIGHT, &sh);
        glViewport(0, 0, sw, sh); glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
        // the picture at its aspect, centred
        float sa = (float)sw / (float)(sh ? sh : 1), pa = (float)w / (float)(h ? h : 1);
        int vw = sw, vh = sh; if (sa > pa) vw = (int)(sh * pa); else vh = (int)(sw / pa);
        glViewport((sw - vw) / 2, (sh - vh) / 2, vw, vh);
        static const float quad[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
        glUseProgram(prog); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex[frame]);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, quad); glEnableVertexAttribArray(0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        eglSwapBuffers(dpy, surf);
        if (shown >= 0 && shown != frame) release(shown);   // the one before is off the screen now
        shown = frame;
    }
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (surf != EGL_NO_SURFACE) eglDestroySurface(dpy, surf);
    eglDestroyContext(dpy, ctx); eglTerminate(dpy);
    return NULL;
}

JNIEXPORT jboolean JNICALL Java_dev_nxnative_fortunesweave_GameActivity_nativeStart(JNIEnv *env, jobject thiz, jstring name) {
    if (g_listen >= 0) return JNI_TRUE;
    (*env)->GetJavaVM(env, &g_vm);
    jclass cls = (*env)->GetObjectClass(env, thiz);
    if (cls) {
        g_act = (*env)->NewGlobalRef(env, thiz);
        g_ontext = (*env)->GetMethodID(env, cls, "onTextInput", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;II)V");
        if (!g_ontext) (*env)->ExceptionClear(env);
        (*env)->DeleteLocalRef(env, cls);
    }
    const char *n = (*env)->GetStringUTFChars(env, name, NULL);
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    struct sockaddr_un a; memset(&a, 0, sizeof a); a.sun_family = AF_UNIX; size_t len = strlen(n); if (len > sizeof a.sun_path - 2) len = sizeof a.sun_path - 2;
    memcpy(a.sun_path + 1, n, len);
    int ok = fd >= 0 && !bind(fd, (struct sockaddr *)&a, (socklen_t)(offsetof(struct sockaddr_un, sun_path) + 1 + len)) && !listen(fd, 1);
    LOG("listening on %s: %s", n, ok ? "yes" : strerror(errno));
    (*env)->ReleaseStringUTFChars(env, name, n);
    if (!ok) { if (fd >= 0) close(fd); return JNI_FALSE; }
    g_listen = fd; g_stop = 0;
    pthread_create(&g_srv, NULL, server, NULL); pthread_create(&g_ren, NULL, renderer, NULL);
    return JNI_TRUE;
}
JNIEXPORT void JNICALL Java_dev_nxnative_fortunesweave_GameActivity_nativeSurface(JNIEnv *env, jclass cls, jobject surface) {
    (void)cls;
    ANativeWindow *w = surface ? ANativeWindow_fromSurface(env, surface) : NULL;
    pthread_mutex_lock(&g_mu);
    ANativeWindow *old = g_win; g_win = w; g_win_changed = 1; pthread_cond_signal(&g_cv);
    pthread_mutex_unlock(&g_mu);
    if (old) ANativeWindow_release(old);   // the renderer has its own surface: it drops it when it sees the change (EGL keeps the window alive meanwhile)
}
JNIEXPORT void JNICALL Java_dev_nxnative_fortunesweave_GameActivity_nativeInput(JNIEnv *env, jclass cls, jlong buttons, jint lx, jint ly, jint rx, jint ry) {
    (void)env; (void)cls;
    dmf_msg m = {DMF_INPUT, 0, 0, 0, (uint64_t)buttons, lx, ly, rx, ry};
    pthread_mutex_lock(&g_mu); send_msg(&m); pthread_mutex_unlock(&g_mu);
}
// The pause menu (GameActivity): DMF_PAUSE 1 when it opens, 0 when it closes; DMF_OPMODE when the console mode setting changes during play
JNIEXPORT void JNICALL Java_dev_nxnative_fortunesweave_GameActivity_nativePause(JNIEnv *env, jclass cls, jboolean on) {
    (void)env; (void)cls;
    dmf_msg m = {DMF_PAUSE, on ? 1u : 0u};
    pthread_mutex_lock(&g_mu); send_msg(&m); pthread_mutex_unlock(&g_mu);
}
JNIEXPORT void JNICALL Java_dev_nxnative_fortunesweave_GameActivity_nativeOpmode(JNIEnv *env, jclass cls, jint mode) {
    (void)env; (void)cls;
    dmf_msg m = {DMF_OPMODE, mode ? 1u : 0u};
    pthread_mutex_lock(&g_mu); send_msg(&m); pthread_mutex_unlock(&g_mu);
}
JNIEXPORT jlong JNICALL Java_dev_nxnative_fortunesweave_GameActivity_nativeFrames(JNIEnv *env, jclass cls) { (void)env; (void)cls; return (jlong)g_frames; }
// GameActivity's text box: the user's answer to the game's keyboard request goes back to the host (its guest thread is waiting in vkd_front_text)
JNIEXPORT void JNICALL Java_dev_nxnative_fortunesweave_GameActivity_nativeTextDone(JNIEnv *env, jclass cls, jboolean ok, jstring text) {
    (void)cls;
    dmf_text_msg t; memset(&t, 0, sizeof t);
    t.type = DMF_TEXT_DONE; t.ok = ok ? 1 : 0;
    if (text) {
        const char *s = (*env)->GetStringUTFChars(env, text, NULL);
        if (s) { strncpy(t.text, s, sizeof t.text - 1); (*env)->ReleaseStringUTFChars(env, text, s); }
    }
    pthread_mutex_lock(&g_mu); send_raw(&t, sizeof t); pthread_mutex_unlock(&g_mu);
}
