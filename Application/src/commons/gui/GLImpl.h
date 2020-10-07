#pragma once

#include <gui/CrossPlatform.h>
#include <misc/Timer.h>
#include <gui/colors.h>

struct GLFWwindow;

namespace gui {
    class GLImpl : public CrossPlatform {
    protected:
        GLFWwindow *window;
        std::function<void()> draw_function;
        std::function<bool()> new_frame_fn;
        double draw_calls;
        Timer draw_timer;
        int index, nextIndex;
        uint32_t pboIds[2];
        Image::Ptr pboImage;
        Image::Ptr pboOutput;
        std::thread::id _update_thread;
        
    public:
        GLImpl(std::function<void()> draw, std::function<bool()> new_frame_fn);
        
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
        void enable_readback();
        void disable_readback();
        Image::Ptr current_frame_buffer() override;
        void update_pbo();
        void init_pbo(uint dwidth, uint dheight);
        void set_icons(const std::vector<file::Path>& icons) override;
        
        GLFWwindow* window_handle() override;
        virtual ~GLImpl();
    private:
        void check_thread_id(int, const char*) const;
    };
}
