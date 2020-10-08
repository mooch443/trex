

#include <types.h>
#include <misc/metastring.h>
#include "MetalImpl.h"

#include <imgui/imgui.h>
#include <imgui/examples/imgui_impl_glfw.h>
#include <imgui/examples/imgui_impl_metal.h>

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <stdio.h>
#include <misc/Timer.h>

#include <Availability.h>

#ifdef __MAC_OS_X_VERSION_MAX_ALLOWED
#if __MAC_OS_X_VERSION_MAX_ALLOWED < 101500
#define NO_SWIZZLE_DIZZLE
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED < 101300
#define NO_ALLOWS_NEXT_DRAWABLE
#endif
#endif

#import <objc/runtime.h>

#define GLIMPL_CHECK_THREAD_ID() check_thread_id( __LINE__ , __FILE__ )

namespace gui {
struct MetalData {
    id <MTLDevice> device;
    id <MTLCommandQueue> commandQueue;
    CAMetalLayer *layer;
    MTLRenderPassDescriptor *renderPassDescriptor;
};
namespace metal {
gui::MetalImpl * current_instance = nullptr;
}
}

@interface GLFWCustomDelegate : NSObject
+ (void)load; // load is called before even main() is run (as part of objc class registration)
@end

// part of your application

bool startup_kind_of_done = false;
std::string startup_file_to_load = "";

extern "C"{
    bool forward_load_message(const std::vector<file::Path>& paths){
        auto str = cmn::Meta::toStr(paths);
        Debug("Open file: %S", &str);
        
        if(gui::metal::current_instance) {
            return gui::metal::current_instance->open_files(paths);
        }
        return false;
    }
}

@implementation GLFWCustomDelegate

+ (void)load{


    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        {
            Class target_c = [GLFWCustomDelegate class];
            Method originalMethod = class_getInstanceMethod(objc_getClass("GLFWApplicationDelegate"), @selector(application:openFiles:));
            Method swizzledMethod = class_getInstanceMethod(target_c, @selector(swz_application:openFiles:));

            BOOL didAddMethod =
            class_addMethod(objc_getClass("GLFWApplicationDelegate"),
                            @selector(application:openFiles:),
                            method_getImplementation(swizzledMethod),
                            method_getTypeEncoding(swizzledMethod));

            if (didAddMethod) {
                class_replaceMethod(objc_getClass("GLFWApplicationDelegate"),
                                    @selector(swz_application:openFiles:),
                                    method_getImplementation(originalMethod),
                                    method_getTypeEncoding(originalMethod));
            } else {
                method_exchangeImplementations(originalMethod, swizzledMethod);
            }
        }
        
        Class target_c = [GLFWCustomDelegate class];
        Method originalMethod = class_getInstanceMethod(objc_getClass("GLFWApplicationDelegate"), @selector(application:openFile:));
        Method swizzledMethod = class_getInstanceMethod(target_c, @selector(swz_application:openFile:));

        BOOL didAddMethod =
        class_addMethod(objc_getClass("GLFWApplicationDelegate"),
                        @selector(application:openFile:),
                        method_getImplementation(swizzledMethod),
                        method_getTypeEncoding(swizzledMethod));

        if (didAddMethod) {
            class_replaceMethod(objc_getClass("GLFWApplicationDelegate"),
                                @selector(swz_application:openFile:),
                                method_getImplementation(originalMethod),
                                method_getTypeEncoding(originalMethod));
        } else {
            method_exchangeImplementations(originalMethod, swizzledMethod);
        }
    });
    
}

- (BOOL)swz_application:(NSApplication *)sender openFile:(NSString *)filename{
    return forward_load_message({filename.UTF8String});
}

- (void)swz_application:(NSApplication *)sender openFiles:(NSArray<NSString *> *)filenames{
    std::vector<file::Path> paths;
    for(size_t i = 0; i < filenames.count; ++i)
        paths.push_back(filenames[i].UTF8String);
    forward_load_message(paths);
}

@end

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

namespace gui {
void MetalImpl::check_thread_id(int line, const char* file) const {
#ifndef NDEBUG
    if(std::this_thread::get_id() != _update_thread)
        U_EXCEPTION("Wrong thread in '%s' line %d.", file, line);
#endif
}

    MetalImpl::MetalImpl(std::function<void()> draw, std::function<bool()> new_frame_fn)
        : draw_function(draw), new_frame_fn(new_frame_fn), _data(new MetalData)
    {
        gui::metal::current_instance = this;
    }

    MetalImpl::~MetalImpl() {
        GLIMPL_CHECK_THREAD_ID();
        
        if(gui::metal::current_instance == this)
            gui::metal::current_instance = nullptr;
        
        // Cleanup
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        glfwDestroyWindow(window);
        glfwTerminate();
    }

bool MetalImpl::open_files(const std::vector<file::Path> &paths) {
    if(_fn_open_files)
        return _fn_open_files(paths);
    return false;
}

    void MetalImpl::init()
    {
        _update_thread = std::this_thread::get_id();
        
        // Setup Dear ImGui binding
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
        
        // Setup style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();
        
        // Setup window
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            U_EXCEPTION("[METAL] Cannot init GLFW.");
    }

    void MetalImpl::post_init() {
        ImGui_ImplMetal_Init(_data->device);
        
        _data->commandQueue = [_data->device newCommandQueue];
    }
    
    void MetalImpl::create_window(int width, int height) {
        // Create window with graphics context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(width, height, "Dear ImGui GLFW+Metal example", NULL, NULL);
        if (window == NULL)
            U_EXCEPTION("[METAL] Cannot create GLFW window.");
        
        _data->device = MTLCreateSystemDefaultDevice();
        //_data->commandQueue = [_data->device newCommandQueue];
        
        NSWindow *nswin = glfwGetCocoaWindow(window);
        _data->layer = [CAMetalLayer layer];
        _data->layer.device = _data->device;
        _data->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
#ifndef NO_ALLOWS_NEXT_DRAWABLE
        _data->layer.allowsNextDrawableTimeout = YES;
        _data->layer.displaySyncEnabled = NO;
#endif
        _data->layer.framebufferOnly = NO;
        nswin.contentView.layer = _data->layer;
        nswin.contentView.wantsLayer = YES;
        
        _data->renderPassDescriptor = [MTLRenderPassDescriptor new];
        
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        
    }

    LoopStatus MetalImpl::update_loop() {
        GLIMPL_CHECK_THREAD_ID();
        
        LoopStatus status = LoopStatus::IDLE;
        static dispatch_semaphore_t _frameBoundarySemaphore = dispatch_semaphore_create(1);
        static std::mutex mutex;
        //dispatch_semaphore_wait(_frameBoundarySemaphore, DISPATCH_TIME_FOREVER);
        
        ++frame_index;
        
        
        glfwPollEvents();
        if(glfwWindowShouldClose(window))
            return LoopStatus::END;
        
        if(new_frame_fn()) {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            if(_frame_capture_enabled) {
                if(!_current_framebuffer)
                    _current_framebuffer = std::make_shared<Image>(height, width, 4);
            } else if(_current_framebuffer)
                _current_framebuffer = nullptr;
            
            
            @autoreleasepool {
                
                _data->layer.drawableSize = CGSizeMake(width, height);
                id<CAMetalDrawable> drawable = [_data->layer nextDrawable];
                
                id<MTLCommandBuffer> commandBuffer = [_data->commandQueue commandBuffer];
                _data->renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(_clear_color[0] / 255.f, _clear_color[1] / 255.f, _clear_color[2] / 255.f, _clear_color[3] / 255.f);
                _data->renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
                _data->renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
                _data->renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
                id <MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:_data->renderPassDescriptor];
                [renderEncoder pushDebugGroup:@"TRex"];
                
                // Start the Dear ImGui frame
                ImGui_ImplMetal_NewFrame(_data->renderPassDescriptor);
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                
                auto lock = new std::lock_guard<std::mutex>(mutex);
                
                draw_function();
                
                // Rendering
                ImGui::Render();
                {
                    std::lock_guard<std::mutex> guard(_texture_mutex);
                    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);
                }
                
                [renderEncoder popDebugGroup];
                [renderEncoder endEncoding];
                
                [commandBuffer presentDrawable:drawable];
                
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> commandBuffer) {
                    {
                    }
                    delete lock;
                    {
                        std::lock_guard<std::mutex> guard(_texture_mutex);
                        for(auto ptr : _delete_textures) {
                            id<MTLTexture> texture = (__bridge id<MTLTexture>)ptr;
                            [texture release];
                        }
                        _delete_textures.clear();
                    }
                    //dispatch_semaphore_signal(_frameBoundarySemaphore);
                }];
                
                [commandBuffer commit];
                
                if(_frame_capture_enabled) {
                    [commandBuffer waitUntilCompleted];
                    [drawable.texture getBytes:_current_framebuffer->data() bytesPerRow:_current_framebuffer->dims * _current_framebuffer->cols fromRegion:MTLRegionMake2D(0, 0, _current_framebuffer->cols, _current_framebuffer->rows) mipmapLevel:0];
                }
            }
            
            /*CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
            CGBitmapInfo bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaFirst;

            CGDataProviderRef provider = CGDataProviderCreateWithData(nil, p, selfturesize, nil);
            CGImageRef cgImageRef = CGImageCreate(width, height, 8, 32, rowBytes, colorSpace, bitmapInfo, provider, nil, true, (CGColorRenderingIntent)kCGRenderingIntentDefault);

            UIImage *getImage = [UIImage imageWithCGImage:cgImageRef];
            CFRelease(cgImageRef);*/
            
            ++_draw_calls;
            status = LoopStatus::UPDATED;
            
        } //else
            //delete lock;
        
        /*if(_draw_timer.elapsed() >= 1) {
            Debug("%f draw_calls / s", _draw_calls);
            _draw_calls = 0;
            _draw_timer.reset();
        }*/
        
        return status;
    }

    Image::Ptr MetalImpl::current_frame_buffer() {
        return _frame_capture_enabled ? _current_framebuffer : nullptr;
    }
    
    void MetalImpl::loop(CrossPlatform::custom_function_t custom_loop) {
        LoopStatus status = LoopStatus::IDLE;
        
        // Main loop
        while (status != LoopStatus::END)
        {
            if(!custom_loop())
                break;
            
            status = update_loop();
            if(status != gui::LoopStatus::UPDATED)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    TexturePtr MetalImpl::texture(const Image * ptr) {
        GLIMPL_CHECK_THREAD_ID();
        
        int width = next_pow2(ptr->cols);
        int height = next_pow2(ptr->rows);
        
        auto input_format = MTLPixelFormatRGBA8Unorm;
        if(ptr->dims == 1) {
            input_format = MTLPixelFormatR8Unorm;
        } else if(ptr->dims == 2) {
            input_format = MTLPixelFormatRG8Unorm;
        }
        
        MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:input_format width:width height:height mipmapped:NO];
        textureDescriptor.usage = MTLTextureUsageShaderRead;
    #if TARGET_OS_OSX
        textureDescriptor.storageMode = MTLStorageModeManaged;
    #else
        textureDescriptor.storageMode = MTLStorageModeShared;
    #endif
#ifndef NO_SWIZZLE_DIZZLE
        if (@available(macOS 10.15, *)) {
            if(ptr->dims != 4) {
                MTLTextureSwizzleChannels swizzle;
                
                if(ptr->dims == 1) {
                    swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne);
                } else if(ptr->dims == 2) {
                    swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleGreen);
                } else
                    U_EXCEPTION("Unknown texture format with %d channels.", ptr->dims);
                textureDescriptor.swizzle = swizzle;
                //texture = [texture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm textureType:MTLTextureType2D levels:NSMakeRange(0, 0) slices:NSMakeRange(0, 0) swizzle:swizzle];
            }
        }
#endif
        
        id <MTLTexture> texture = [_data->device newTextureWithDescriptor:textureDescriptor];
        [texture replaceRegion:MTLRegionMake2D(0, 0, ptr->cols, ptr->rows) mipmapLevel:0 withBytes:ptr->data() bytesPerRow:ptr->cols * ptr->dims];
        
        return std::unique_ptr<PlatformTexture>(new PlatformTexture{(__bridge void*)texture, [this](void** ptr){
            std::lock_guard<std::mutex> guard(_texture_mutex);
            _delete_textures.emplace_back(*ptr);
            
            //Debug("Deleting %X", *ptr);
            *ptr = nullptr;
            //id<MTLTexture> texture = (__bridge id<MTLTexture>)ptr;
            //[texture release];
        }, width, height, static_cast<int>(ptr->cols), static_cast<int>(ptr->rows)});
    }

    void MetalImpl::clear_texture(TexturePtr&& tex) {
        /*std::lock_guard<std::mutex> guard(_texture_mutex);
        id<MTLTexture> texture = (__bridge id<MTLTexture>)tex->ptr;
        [texture release];*/
    }

    void MetalImpl::bind_texture(const PlatformTexture&) {
        
    }

    void MetalImpl::update_texture(PlatformTexture& tex, const Image * ptr) {
        GLIMPL_CHECK_THREAD_ID();
        
        id<MTLTexture> texture = (__bridge id<MTLTexture>)tex.ptr;
        
        MTLRegion region = {
            { 0, 0, 0 },                   // MTLOrigin
            {ptr->cols, ptr->rows, 1} // MTLSize
        };
        NSUInteger bytesPerRow = ptr->dims * ptr->cols;
        [texture replaceRegion:region
            mipmapLevel:0
              withBytes:ptr->data()
            bytesPerRow:bytesPerRow];
        
        tex.image_height = ptr->rows;
        tex.image_width = ptr->cols;
    }

    void MetalImpl::set_title(std::string title) {
        glfwSetWindowTitle(window, title.c_str());
    }

    GLFWwindow* MetalImpl::window_handle() {
        return window;
    }
}
