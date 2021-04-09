#pragma once

#include <commons/common/misc/defines.h>
#include <misc/Image.h>
#include <gui/types/Basic.h>
#include <gui/DrawBase.h>

struct GLFWwindow;

namespace gui {
    struct PlatformTexture {
        void* ptr = nullptr ;
        std::function<void(void**)> deleter;
        uint width = 0;
        uint height = 0;
        uint image_width = 0;
        uint image_height = 0;
        
        PlatformTexture() = default;
        ~PlatformTexture() {
            if(deleter)
                deleter(&ptr);
        }
    };
    using TexturePtr = std::unique_ptr<PlatformTexture>;

    class CrossPlatform {
    protected:
        std::function<bool(const std::vector<file::Path>&)> _fn_open_files;
    public:
        typedef std::function<bool()> custom_function_t;
        
        CrossPlatform() : _clear_color(0, 0, 0, 0), _frame_capture_enabled(false) {}
        virtual ~CrossPlatform() {}
        virtual void init() = 0;
        virtual void post_init() = 0;
        virtual void create_window(const char* title, int width, int height) = 0;
        virtual LoopStatus update_loop(const custom_function_t& = nullptr) = 0;
        //virtual void* texture(uint width, uint height) = 0;
        virtual TexturePtr texture(const Image*) = 0;
        virtual void clear_texture(TexturePtr&&) = 0;
        virtual void bind_texture(const PlatformTexture&) = 0;
        virtual void update_texture(PlatformTexture&, const Image*) = 0;
        virtual void set_title(std::string) = 0;
        virtual GLFWwindow* window_handle() = 0;
        virtual void set_icons(const std::vector<file::Path>& ) {}
        virtual void set_open_files_fn(std::function<bool(const std::vector<file::Path>&)> fn) { _fn_open_files = fn; }
        virtual void toggle_full_screen() {}
        
        virtual Image::Ptr current_frame_buffer() {
            return nullptr;
        }
        
        GETTER_SETTER(Color, clear_color)
        GETTER_SETTER(bool, frame_capture_enabled)
    };
}
