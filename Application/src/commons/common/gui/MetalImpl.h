#pragma once
#if __APPLE__ && __has_include(<Metal/Metal.h>)
#define TREX_METAL_AVAILABLE true

#include <gui/CrossPlatform.h>
#include <misc/Timer.h>
struct GLFWwindow;

namespace gui {
    struct MetalData;

    class MetalImpl : public CrossPlatform {
        GLFWwindow *window = nullptr;
        std::function<void()> draw_function;
        std::function<bool()> new_frame_fn;
        MetalData* _data;
        
        double _draw_calls = 0;
        Timer _draw_timer;
        
        Image::UPtr _current_framebuffer;
        std::mutex _texture_mutex;
        std::vector<void*> _delete_textures;
        
        std::atomic<size_t> frame_index;
        std::thread::id _update_thread;
        
    public:
        MetalImpl(std::function<void()> draw, std::function<bool()> new_frame_fn);
        float center[2] = {0.5f, 0.5f};
        
        void init() override;
        void post_init() override;
        void create_window(const char* title, int width, int height) override;
        LoopStatus update_loop(const custom_function_t& = nullptr) override;
        TexturePtr texture(const Image*) override;
        void clear_texture(TexturePtr&&) override;
        void bind_texture(const PlatformTexture&) override;
        void update_texture(PlatformTexture&, const Image*) override;
        void set_title(std::string) override;
        const Image::UPtr& current_frame_buffer() override;
        void toggle_full_screen() override;
        void message(const std::string&) const;
        
        virtual ~MetalImpl();
        GLFWwindow* window_handle() override;
    public:
        bool open_files(const std::vector<file::Path>&);
        void check_thread_id(int line, const char* file) const;
    };
}

#else
#define TREX_METAL_AVAILABLE false
#endif
