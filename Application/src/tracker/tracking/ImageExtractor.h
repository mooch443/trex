#pragma once

#include <misc/idx_t.h>
#include <misc/frame_t.h>
#include <misc/PVBlob.h>
#include <tracking/Stuffs.h>
#include <misc/vec2.h>
#include <tracker/misc/default_config.h>
#include <pv.h>
#include <misc/PackLambda.h>

using namespace cmn;
using namespace track;

namespace extract {

struct Task {
    Idx_t fdx;
    pv::bid bdx;
    Range<Frame_t> segment;
    
    std::string toStr() const {
        return "task<"+Meta::toStr(fdx)+","+Meta::toStr(bdx)+">";
    }
};

struct Query {
    const BasicStuff* basic;
    const PostureStuff* posture;
};

struct Result {
    Frame_t frame;
    Idx_t fdx;
    pv::bid bdx;
    Image::UPtr image;
};

enum class Flag {
    None = 0,
    RemoveSmallFrames = 1 << 0
};

inline uint32_t operator|(const Flag& A, const Flag& B) {
    return (uint32_t)A | (uint32_t)B;
}

inline uint32_t operator|(uint32_t A, const Flag& B) {
    return A | (uint32_t)B;
}

inline uint32_t operator|(const Flag& A, uint32_t B) {
    return (uint32_t)A | B;
}

struct Settings {
    uint32_t flags{0u};
    uint64_t max_size_bytes{1000u * 1000u * 1000u};
    Size2 image_size{Float2_t(80), Float2_t(80)};
    uint8_t num_threads{5u};
    default_config::recognition_normalization_t::Class normalization{default_config::recognition_normalization_t::none};
    
    std::string toStr() const {
        return "settings<flags:"+Meta::toStr(flags)
            +" max:"+FileSize{max_size_bytes}.toStr()
            +" res:"+Meta::toStr(image_size)
            +" threads:"+Meta::toStr(num_threads)+">";
    }
};

class ImageExtractor {
public:
    static bool is(uint32_t flags, Flag flag);
    
protected:
    GETTER_I(uint64_t, pushed_items, 0)
    GETTER_I(uint64_t, collected_items, 0)
    
private:
    Settings _settings{};
    ska::bytell_hash_map<Frame_t, std::vector<Task>> _tasks;
    std::promise<void> _promise{};
    std::future<void> _future{_promise.get_future()};
    pv::File& _video;
    
    std::thread _thread;
    
    using selector_sig = bool(const Query&);
    using partial_apply_sig = void(std::vector<Result>&&);
    using callback_sig = void(ImageExtractor*, double, bool);
    
    using selector_t = package::F<selector_sig>;
    using partial_apply_t = package::F<partial_apply_sig>;
    using callback_t = package::F<callback_sig>;
    
public:
    template<typename F>
    ImageExtractor(pv::File & video,
                   F && selector,
                   auto && partial_apply,
                   auto && callback,
                   Settings&& settings = {})
        requires requires (Query q, F && selector) { { selector(q) } -> std::convertible_to<bool>; }
                      && similar_args<decltype(partial_apply), partial_apply_sig>
                      && similar_args<decltype(callback), callback_sig>
                      && similar_args<decltype(selector), selector_sig>
    :   _settings(std::move(settings)),
        _video(video),
        _thread([this,
                 selector = pack<selector_sig>(std::move(selector)),
                 callback = pack<callback_sig>(std::move(callback)),
                 partial_apply = pack<partial_apply_sig>(std::move(partial_apply))]()
                mutable
            {
                // starting the update thread, moving generic lambdas
                // into packaged tasks:
                update_thread(std::move(selector),
                              std::move(partial_apply),
                              std::move(callback));
            })
    { }
    
    ~ImageExtractor();
    
    std::future<void>& future();
    
    //! Filter tasks based on properties set in _settings::flags
    //! (e.g. remove frames with less than 25% of average number of items/frame)
    void filter_tasks();
    
    //! This figures out the kind of tasks that needs to be
    //! looked at.
    void collect(selector_t&& selector);
    
    //! The main thread of the async extractor.
    //! Here, it'll do:
    //!     1. collect tasks to be done
    //!     2. filter tasks (optional)
    //!     3. retrieve image data
    //!     4. callback() and fulfill promise
    void update_thread(selector_t&& selector, partial_apply_t&& partial_apply, callback_t&& callback);
    
    //! Retrieve image data from the video file, associated with the
    //! filtered tasks from all previous steps.
    uint64_t retrieve_image_data(partial_apply_t&& apply, callback_t& callback);
};

}
