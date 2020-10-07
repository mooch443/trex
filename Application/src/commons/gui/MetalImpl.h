#pragma once
#if __APPLE__ && __has_include(<Metal/Metal.h>)
#define TREX_METAL_AVAILABLE true

#include <gui/CrossPlatform.h>
#include <misc/Timer.h>
struct GLFWwindow;

namespace gui {
    struct MetalData;

    class MetalImpl : public CrossPlatform {
        GLFWwindow *window;
        std::function<void()> draw_function;
        std::function<bool()> new_frame_fn;
        MetalData* _data;
        
        double _draw_calls = 0;
        Timer _draw_timer;
        
        Image::Ptr _current_framebuffer;
        std::mutex _texture_mutex;
        
    public:
        MetalImpl(std::function<void()> draw, std::function<bool()> new_frame_fn);
        
        void init() override;
        void post_init() override;
        void create_window(int width, int height) override;
        void loop(custom_function_t) override;
        LoopStatus update_loop() override;
        TexturePtr texture(const Image*) override;
        void clear_texture(TexturePtr&&) override;
        void bind_texture(const PlatformTexture&) override;
        void update_texture(PlatformTexture&, const Image*) override;
        void set_title(std::string) override;
        Image::Ptr current_frame_buffer() override;
        
        virtual ~MetalImpl();
        GLFWwindow* window_handle() override;
    public:
        bool open_files(const std::vector<file::Path>&);
    };
}

#else
#define TREX_METAL_AVAILABLE false
#endif
