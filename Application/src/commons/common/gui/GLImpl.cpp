#if TREX_HAS_OPENGL
#include <types.h>
#include <cstdio>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include <imgui/backends/imgui_impl_opengl2.h>
using ImTextureID_t = ImGui_OpenGL2_TextureID;

#include <imgui/backends/imgui_impl_opengl3.h>
//using ImTextureID_t = ImGui_OpenGL3_TextureID;

#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>    // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>    // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

#ifndef GL_VERSION_3_2
#define OPENGL3_CONDITION (false)
#else
#define OPENGL3_CONDITION (!CMN_USE_OPENGL2 && ((GLVersion.major == 3 && GLVersion.minor >= 2) || (GLVersion.major > 3)))
#endif

#ifndef GL_PIXEL_PACK_BUFFER
#define GL_PIXEL_PACK_BUFFER 0
#endif
#ifndef GL_RG
#define GL_RG 0
#endif
#ifndef GL_RG8
#define GL_RG8 0
#endif
#ifndef GL_TEXTURE_SWIZZLE_RGBA
#define GL_TEXTURE_SWIZZLE_RGBA 0
#endif

//#define GLFW_INCLUDE_GL3  /* don't drag in legacy GL headers. */
#define GLFW_NO_GLU       /* don't drag in the old GLU lib - unless you must. */

#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include "GLImpl.h"
#include <misc/Timer.h>
#include <misc/checked_casts.h>

#ifdef WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw/glfw3native.h>
#undef GLFW_EXPOSE_NATIVE_WIN32

#include <gui/darkmode.h>
#endif

#define GLIMPL_CHECK_THREAD_ID() check_thread_id( __LINE__ , __FILE__ )

//#include "misc/freetype/imgui_freetype.h"
//#include "misc/freetype/imgui_freetype.cpp"

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

namespace gui {

GLImpl::GLImpl(std::function<void()> draw, std::function<bool()> new_frame_fn) : draw_function(draw), new_frame_fn(new_frame_fn)
{
}

void GLImpl::init() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        U_EXCEPTION("[GL] Cannot initialize GLFW.");
    
    draw_calls = 0;
    _update_thread = std::this_thread::get_id();
}

void GLImpl::post_init() {
    if OPENGL3_CONDITION {
        ImGui_ImplOpenGL3_NewFrame();
    } else
        ImGui_ImplOpenGL2_NewFrame();
}

void GLImpl::set_icons(const std::vector<file::Path>& icons) {
    std::vector<GLFWimage> images;
    std::vector<Image::UPtr> data;

    for (auto& path : icons) {
        if (!path.exists()) {
            Except("Cant find application icon '%S'.", &path.str());
            continue;
        }

        cv::Mat image = cv::imread(path.str(), cv::IMREAD_UNCHANGED);
        if (image.empty())
            continue;

        assert(image.channels() <= 4 && image.channels() != 2);

        cv::cvtColor(image, image, image.channels() == 3 ? cv::COLOR_BGR2RGBA : (image.channels() == 4 ? cv::COLOR_BGRA2RGBA : cv::COLOR_GRAY2RGBA));

        auto ptr = Image::Make(image);
        data.emplace_back(std::move(ptr));
        images.push_back(GLFWimage());
        images.back().pixels = data.back()->data();
        images.back().width = sign_cast<int>(data.back()->cols);
        images.back().height = sign_cast<int>(data.back()->rows);
    }

    glfwSetWindowIcon(window, images.size(), images.data());
}

void GLImpl::create_window(const char* title, int width, int height) {
    glfwSetErrorCallback([](int code, const char* str) {
        Except("[GLFW] Error %d: '%s'", code, str);
    });

#if __APPLE__
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    
    #if !CMN_USE_OPENGL2
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
    #else
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    #endif
    
#else
    #if !CMN_USE_OPENGL2
        // GL 3.0 + GLSL 130
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
    #else
        // GL 2.1 + GLSL 120
        const char* glsl_version = "#version 110";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    #endif
#endif
    
    // Create window with graphics context
    window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (window == NULL)
        U_EXCEPTION("[GL] Cannot create GLFW window.");
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#else
    bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
    if (err)
    {
        U_EXCEPTION("Failed to initialize OpenGL loader!");
    }
    
    if OPENGL3_CONDITION
        Debug("Using OpenGL3.2 (seems supported, %s).", glGetString(GL_VERSION));
    else
        Debug("Using OpenGL2.1 (%s)", glGetString(GL_VERSION));
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);

    if OPENGL3_CONDITION
        ImGui_ImplOpenGL3_Init(glsl_version);
    else
        ImGui_ImplOpenGL2_Init();

#ifdef WIN32
    auto native = glfwGetWin32Window(window);
    std::once_flag once;
    std::call_once(once, []() {
        InitDarkMode();
        g_darkModeEnabled = true;
        AllowDarkModeForApp(true);
    });
    
    AllowDarkModeForWindow(native, true);
    RefreshTitleBarThemeColor(native);
#endif

    Debug("Init complete.");
}

GLFWwindow* GLImpl::window_handle() {
    return window;
}

LoopStatus GLImpl::update_loop(const CrossPlatform::custom_function_t& custom_loop) {
    LoopStatus status = LoopStatus::IDLE;
    glfwPollEvents();
    
    if(glfwWindowShouldClose(window))
        return LoopStatus::END;
    
    if(!custom_loop())
        return LoopStatus::END;
    
    if(new_frame_fn())
    {
        {
            std::lock_guard guard(texture_mutex);
            for(auto & fn : _texture_updates)
                fn();
            _texture_updates.clear();
        }
        
        if OPENGL3_CONDITION
            ImGui_ImplOpenGL3_NewFrame();
        else
            ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        
        ImGui::NewFrame();
        
        draw_function();
        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        
        if(_frame_capture_enabled)
            init_pbo((uint)display_w, (uint)display_h);
        
        glClearColor(_clear_color.r / 255.f, _clear_color.g / 255.f, _clear_color.b / 255.f, _clear_color.a / 255.f);
        glClear(GL_COLOR_BUFFER_BIT);
        if OPENGL3_CONDITION {
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        } else {
            ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        }
        
        if(_frame_capture_enabled)
            update_pbo();
        else if(pboImage) {
            pboImage = nullptr;
            pboOutput = nullptr;
        }
        
        if(!OPENGL3_CONDITION)
            glfwMakeContextCurrent(window);
        glfwSwapBuffers(window);
        
        ++draw_calls;
        status = LoopStatus::UPDATED;
        
    }
    
    /*if(draw_timer.elapsed() >= 1) {
        Debug("%f draw_calls / s", draw_calls);
        draw_calls = 0;
        draw_timer.reset();
    }*/
    
    return status;
}

void GLImpl::init_pbo(uint width, uint height) {
    if(!pboImage || pboImage->cols != width || pboImage->rows != height) {
        if OPENGL3_CONDITION {
            if(pboImage) {
                glDeleteBuffers(2, pboIds);
            }
            
            pboImage = Image::Make(height, width, 4);
            pboOutput = Image::Make(height, width, 4);
            
            glGenBuffers(2, pboIds);
            auto nbytes = width * height * 4;
            for(int i=0; i<2; ++i) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[i]);
                glBufferData(GL_PIXEL_PACK_BUFFER, nbytes, NULL, GL_STREAM_READ);
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        }
    }
}

void GLImpl::update_pbo() {
    if OPENGL3_CONDITION {
        // "index" is used to read pixels from framebuffer to a PBO
        // "nextIndex" is used to update pixels in the other PBO
        index = (index + 1) % 2;
        nextIndex = (index + 1) % 2;

        // set the target framebuffer to read
        glReadBuffer(GL_BACK);

        // read pixels from framebuffer to PBO
        // glReadPixels() should return immediately.
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[index]);
        glReadPixels(0, 0, (GLsizei)pboImage->cols, (GLsizei)pboImage->rows, GL_BGRA, GL_UNSIGNED_BYTE, 0);

        // map the PBO to process its data by CPU
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[nextIndex]);
        GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
        if(ptr)
        {
            memcpy(pboImage->data(), ptr, pboImage->size());
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            
            // flip vertically
            cv::flip(pboImage->get(), pboOutput->get(), 0);
        }

        // back to conventional pixel operation
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }
}

GLImpl::~GLImpl() {
    glDeleteBuffers(2, pboIds);
    
    // Cleanup
    if OPENGL3_CONDITION
        ImGui_ImplOpenGL3_Shutdown();
    else
        ImGui_ImplOpenGL2_Shutdown();
    
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    
    glfwTerminate();
}

void GLImpl::enable_readback() {
    
}

void GLImpl::disable_readback() {
    
}

const Image::UPtr& GLImpl::current_frame_buffer() {
    return pboOutput;
}

void GLImpl::check_thread_id(int line, const char* file) const {
#ifndef NDEBUG
    if(std::this_thread::get_id() != _update_thread)
        U_EXCEPTION("Wrong thread in '%s' line %d.", file, line);
#endif
}

static std::vector<uchar> empty;

void GLImpl::toggle_full_screen() {
    GLFWmonitor *_monitor = nullptr;

    static int count;
    static auto monitors = glfwGetMonitors(&count);

    int wx, wy, wh, ww;

    // backup window position and window size
    glfwGetWindowPos(window_handle(), &wx, &wy);
    glfwGetWindowSize(window_handle(), &ww, &wh);

    Vec2 center = Vec2(wx, wy) + Size2(ww, wh) * 0.5;

    for (int i = 0; i < count; ++i) {
        int x, y, w, h;
        glfwGetMonitorWorkarea(monitors[i], &x, &y, &w, &h);
        if (Bounds(x, y, w, h).contains(center)) {
            _monitor = monitors[i];
        }
    }

    // get resolution of monitor
    const GLFWvidmode * mode = glfwGetVideoMode(_monitor);
    if (_monitor == nullptr)
        _monitor = glfwGetPrimaryMonitor();
    
    if ( fullscreen )
    {
        // backup window position and window size
        glfwGetWindowPos( window_handle(), &_wndPos[0], &_wndPos[1] );
        glfwGetWindowSize( window_handle(), &_wndSize[0], &_wndSize[1] );

        // switch to full screen
        glfwSetWindowSize(window_handle(), mode->width, mode->height);
        glfwSetWindowPos(window_handle(), 0, 0);
        glfwSetWindowMonitor(window_handle(), _monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        //glfwSetWindowMonitor( window_handle(), _monitor, 0, 0, mode->width, mode->height, 0 );
        fullscreen = false;
    }
    else
    {
        // restore last window size and position
        glfwSetWindowMonitor( window_handle(), nullptr,  _wndPos[0], _wndPos[1], _wndSize[0], _wndSize[1], mode->refreshRate );
        fullscreen = true;
        
    }
}

TexturePtr GLImpl::texture(const Image * ptr) {
    GLIMPL_CHECK_THREAD_ID();
    
    // Turn the RGBA pixel data into an OpenGL texture:
    GLuint my_opengl_texture;
    glGenTextures(1, &my_opengl_texture);
    if(my_opengl_texture == 0)
        U_EXCEPTION("Cannot create texture of dimensions %dx%d", ptr->cols, ptr->rows);
    glBindTexture(GL_TEXTURE_2D, my_opengl_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, ptr->dims != 4 ? (GLint)ptr->dims : 0);
    
#if !CMN_USE_OPENGL2
#define GL_LUMINANCE 0x1909
#define GL_LUMINANCE_ALPHA 0x190A
#endif
    
    GLint output_type = GL_RGBA8;
    GLenum input_type = GL_RGBA;
    
    if OPENGL3_CONDITION {
        if(ptr->dims == 1) {
            output_type = GL_RED;
            input_type = GL_RED;
        }
        if(ptr->dims == 2) {
            output_type = GL_RG8;
            input_type = GL_RG;
            
            GLint swizzleMask[] = {GL_RED, GL_ZERO, GL_ZERO, GL_GREEN};
            glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
        }
        
    } else {
        output_type = GL_RGBA;
        
        if(ptr->dims == 1) {
            //output_type = GL_LUMINANCE8;
            input_type = GL_LUMINANCE;
        }
        if(ptr->dims == 2) {
            //output_type = GL_LUMINANCE_ALPHA;
            input_type = GL_LUMINANCE_ALPHA;
        }
    }
    
    auto width = next_pow2(sign_cast<uint64_t>(ptr->cols)), height = next_pow2(sign_cast<uint64_t>(ptr->rows));
    auto capacity = size_t(ptr->dims) * size_t(width) * size_t(height);
    if (empty.size() < capacity)
        empty.resize(capacity, 0);

    glTexImage2D(GL_TEXTURE_2D, 0, output_type, width, height, 0, input_type, GL_UNSIGNED_BYTE, empty.data());

    glTexSubImage2D(GL_TEXTURE_2D,0,0,0, (GLsizei)ptr->cols, (GLsizei)ptr->rows, input_type, GL_UNSIGNED_BYTE, ptr->data());
    glBindTexture(GL_TEXTURE_2D, 0);
    
    return std::make_unique<PlatformTexture>(
        new ImTextureID_t{ (uint64_t)my_opengl_texture, ptr->dims != 4 },
        [this](void ** ptr) {
            std::lock_guard guard(texture_mutex);
            _texture_updates.emplace_back([this, object = (ImTextureID_t*)*ptr](){
                GLIMPL_CHECK_THREAD_ID();
                
                GLuint _id = (GLuint) object->texture_id;
                
                glBindTexture(GL_TEXTURE_2D, _id);
                glDeleteTextures(1, &_id);
                glBindTexture(GL_TEXTURE_2D, 0);
                
                delete object;
            });
            
            *ptr = nullptr;
        },
        static_cast<uint>(width), static_cast<uint>(height),
        ptr->cols, ptr->rows
    );
}

void GLImpl::clear_texture(TexturePtr&&) {
    /*GLIMPL_CHECK_THREAD_ID();
    
    auto object = (ImTextureID_t*)id_->ptr;
    GLuint _id = (GLuint) object->texture_id;
    
    glBindTexture(GL_TEXTURE_2D, _id);
    glDeleteTextures(1, &_id);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    delete object;*/
}

void GLImpl::bind_texture(const PlatformTexture& id_) {
    GLIMPL_CHECK_THREAD_ID();
    
    auto object = (ImTextureID_t*)id_.ptr;
    GLuint _id = (GLuint) object->texture_id;
    
    glBindTexture(GL_TEXTURE_2D, _id);
}

void GLImpl::update_texture(PlatformTexture& id_, const Image *ptr) {
    GLIMPL_CHECK_THREAD_ID();
    
    auto object = (ImTextureID_t*)id_.ptr;
    GLuint _id = (GLuint) object->texture_id;
    
    if(object->greyscale != (ptr->dims != 4))
        U_EXCEPTION("Texture has not been allocated for number of color channels in Image (%d) != texture (%d)", ptr->dims, object->greyscale ? 1 : 4);
    
    glBindTexture(GL_TEXTURE_2D, _id);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, ptr->dims != 4 ? (GLint)ptr->dims : 0);
    
    GLenum input_type = GL_RGBA;
    if OPENGL3_CONDITION {
        if(ptr->dims == 1) {
            input_type = GL_RED;
        }
        if(ptr->dims == 2) {
            input_type = GL_RG;
        }
        
    } else {
        if(ptr->dims == 1) {
            input_type = GL_LUMINANCE;
        }
        if(ptr->dims == 2) {
            input_type = GL_LUMINANCE_ALPHA;
        }
    }

    auto capacity = size_t(ptr->dims) * size_t(id_.width) * size_t(id_.height);
    if (empty.size() < capacity)
        empty.resize(capacity, 0);

    if (ptr->cols != (uint)id_.width || ptr->rows != (uint)id_.height) {
        glTexSubImage2D(GL_TEXTURE_2D, 0, (GLint)ptr->cols, 0, GLint(id_.width) - GLint(ptr->cols), id_.height, input_type, GL_UNSIGNED_BYTE, empty.data());
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, (GLint)ptr->rows, (GLint)ptr->cols, GLint(id_.height) - GLint(ptr->rows), input_type, GL_UNSIGNED_BYTE, empty.data());
        //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, id_.width, id_.height, input_type, GL_UNSIGNED_BYTE, empty.data());
    }
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0, (GLint)ptr->cols, (GLint)ptr->rows, input_type, GL_UNSIGNED_BYTE, ptr->data());
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ptr->cols, ptr->rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, ptr->data());
    glBindTexture(GL_TEXTURE_2D, 0);
    
    id_.image_width = int(ptr->cols);
    id_.image_height = int(ptr->rows);
}

void GLImpl::set_title(std::string title) {
    glfwSetWindowTitle(window, title.c_str());
}

}
#endif

