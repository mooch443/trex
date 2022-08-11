#include <types.h>
#include <misc/Image.h>
#include <python/GPURecognition.h>
#include <tracking/Individual.h>
#include <tracking/Tracker.h>
#include <misc/default_config.h>
#include <misc/Output.h>
#include <tracking/Recognition.h>
#include <misc/CommandLine.h>

using namespace cmn;
using namespace track;
using py = PythonIntegration;

namespace Python {

auto schedule(Network* net, auto && fn) {
    return py::async_python_function(PackagedTask{
        ._network = net,
        ._task = std::packaged_task<void()>{std::move(fn)},
        ._can_run_before_init = true
    }, py::Flag::FORCE_ASYNC);
}

}

class VINetwork {
    Network _network;
    
public:
    VINetwork()
        : _network("visual_identification")
    {
        _network.setup = [](){
            /*py::set_function("receive", [](int value) {
                print("value = ", value);
            });*/
        };
        _network.unsetup = [](){
            //py::unset_function("receive");
            //print("unsetup");
        };
        
        /*py::async_python_function(PackagedTask{
            ._can_run_before_init = true,
            ._network = &_network,
            ._task = std::packaged_task<void()>([](){
                //py::import_module("learn_static");
            })
        }, py::Flag::DEFAULT);*/
        
        print("Current: ",file::Path(".").find_files());
        Recognition::check_learning_module(true);
        
        py::async_python_function(&_network, [](){
            py::run("learn_static", "reinitialize_network");
            
            try {
                auto program = "import numpy as np\nimport learn_static\n"
                "with np.load(learn_static.output_path+'.npz', allow_pickle=True) as npz:\n"
                "   m = npz['weights'].item()\n"
                "   for i, layer in zip(range(len(learn_static.model.layers)), learn_static.model.layers):\n"
                "       if i in m:\n"
                "           layer.set_weights(m[i])\n";
                py::execute(program);
                print("\tReloaded weights.");
                
            } catch(...) {
                FormatExcept("[py] Failed to load weights.");
                throw;
            }
            
        }).get();
        
    }
    
    auto apply(std::vector<Image::UPtr>&& images, auto&& callback) {
        return Python::schedule(&_network, [
            callback = std::move(callback),
            images = std::move(images)]()
          mutable
        {
            if(images.size() == 0) {
                print("Empty images array.");
                return;
            }
            
            try {
                py::set_variable("images", images, "learn_static");
                py::set_function("receive", std::packaged_task<void(std::vector<std::vector<float>>,std::vector<float>)>([callback = std::move(callback)](std::vector<std::vector<float>>, std::vector<float> r) mutable {
                    callback(std::move(r));
                }), "learn_static");
                py::run("learn_static", "predict");
                py::unset_function("receive", "learn_static");
                py::unset_function("images", "learn_static");
            } catch(const SoftExceptionImpl& e) {
                FormatWarning("Runtime exception: ", e.what());
                throw;
            }
        });
    }
};

/*void Python::update() {
    py::scoped_interpreter scope{};
    
    std::unique_lock guard(_mutex);
    while(!Python::_terminate || !_queue.empty()) {
        //! in the python queue
        while (!_queue.empty()) {
            auto item = std::move(_queue.front());
            _queue.pop();
            
            guard.unlock();
            try {
                if(item._network)
                    item._network->activate();
                else {
                    // deactivate active item?
                }
                
                item._item();
                
            } catch(...) {
                guard.lock();
                throw;
            }
            guard.lock();
        }
        
        if(!_terminate)
            update_variable.wait(guard);
    }
    
    print("[py] ended.");
}*/

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
    const Individual::BasicStuff* basic;
    const Individual::PostureStuff* posture;
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

struct Settings {
    uint32_t flags{0u};
    uint64_t max_size_bytes{1000u * 1000u * 1000u};
    Size2 image_size{Float2_t(80), Float2_t(80)};
    uint8_t num_threads{5u};
    default_config::recognition_normalization_t::Class normalization{default_config::recognition_normalization_t::none};
    
    /*Settings() {}
    Settings(Size2 image_size, uint64_t max_size_bytes, auto flags)
        : flags((uint32_t)flags),
          max_size_bytes(max_size_bytes),
          image_size(image_size)
    { }*/
    
    static constexpr Settings default_init() {
        return Settings{};
    }
    
    std::string toStr() const {
        return "settings<flags:"+Meta::toStr(flags)
            +" max:"+FileSize{max_size_bytes}.toStr()
            +" res:",Meta::toStr(image_size)
            +" threads:"+Meta::toStr(num_threads)+">";
    }
};


namespace constraints {
inline static std::mutex _filter_mutex;
inline static std::map<Idx_t, std::map<Range<Frame_t>, std::shared_ptr<TrainingFilterConstraints>>> _filter_cache_std, _filter_cache_no_std;

inline float standard_deviation(const std::set<float> & v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    
    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    
    return (float)std::sqrt(sq_sum / v.size());
}

std::tuple<Image::UPtr, Vec2> diff_image(const default_config::recognition_normalization_t::Class &normalize, const pv::BlobPtr& blob, const gui::Transform& midline_transform, float median_midline_length_px, const Size2& output_shape) {
    if(normalize == default_config::recognition_normalization_t::posture)
        return Individual::calculate_normalized_diff_image(midline_transform, blob, median_midline_length_px, output_shape, false);
    else if(normalize == default_config::recognition_normalization_t::legacy)
        return Individual::calculate_normalized_diff_image(midline_transform, blob, median_midline_length_px, output_shape, true);
    else if (normalize == default_config::recognition_normalization_t::moments)
    {
        blob->calculate_moments();
        
        gui::Transform tr;
        float angle = narrow_cast<float>(-blob->orientation() + M_PI * 0.25);
        
        tr.rotate(DEGREE(angle));
        tr.translate( -blob->bounds().size() * 0.5);
        //tr.translate(-offset());
        
        return Individual::calculate_normalized_diff_image(tr, blob, 0, output_shape, false);
    }
    else {
        auto && [img, pos] = Individual::calculate_diff_image(blob, output_shape);
        return std::make_tuple(std::move(img), pos);
    }
}

bool cached_filter(Idx_t fdx, const Range<Frame_t>& segment, TrainingFilterConstraints & constraints, const bool with_std) {
    std::lock_guard<std::mutex> guard(_filter_mutex);
    const auto &cache = with_std ? _filter_cache_std : _filter_cache_no_std;
    auto fit = cache.find(fdx);
    if(fit != cache.end()) {
        auto sit = fit->second.find(segment);
        if(sit != fit->second.end()) {
            constraints = *sit->second;
            return true;
        }
    }
    return false;
}

std::shared_ptr<TrainingFilterConstraints> local_midline_length(const Individual *fish, const Range<Frame_t>& segment, const bool calculate_std) {
    std::shared_ptr<TrainingFilterConstraints> constraints = std::make_shared<TrainingFilterConstraints>();
    if(cached_filter(fish->identity().ID(), segment, *constraints, calculate_std))
        return constraints;
    
    Median<float> median_midline, median_outline, median_angle_diff;
    std::set<float> midline_lengths, outline_stds;
    
    const Individual::PostureStuff* previous_midline = nullptr;
    
    fish->iterate_frames(segment, [&](Frame_t frame, const auto&, auto basic, auto posture) -> bool
    {
        if(!basic || !posture || basic->blob.split())
            return true;
        
        auto bounds = basic->blob.calculate_bounds();
        if(!Tracker::instance()->border().in_recognition_bounds(bounds.pos() + bounds.size() * 0.5))
            return true;
        
        if(posture->cached()) {
            median_midline.addNumber(posture->midline_length);
            if(calculate_std)
                midline_lengths.insert(posture->midline_length);
            
            if(previous_midline && previous_midline->frame == frame - 1_f) {
                auto first = Vec2(sin(previous_midline->midline_angle), cos(previous_midline->midline_angle));
                auto second = Vec2(sin(posture->midline_angle), cos(posture->midline_angle));
                auto diff = (first - second).length();
                median_angle_diff.addNumber(diff);
            }
            
            previous_midline = posture;
        }
        
        if(posture->outline) {
            median_outline.addNumber(posture->outline->size());
            if(calculate_std)
                outline_stds.insert(posture->outline->size());
        }
        
        return true;
    });
    
    if(median_midline.added())
        constraints->median_midline_length_px = median_midline.getValue();
    if(median_outline.added())
        constraints->median_number_outline_pts = median_outline.getValue();
    
    if(!midline_lengths.empty())
        constraints->midline_length_px_std = standard_deviation(midline_lengths);
    if(!outline_stds.empty())
        constraints->outline_pts_std = standard_deviation(outline_stds);
    
    constraints->median_angle_diff = median_angle_diff.added() ? median_angle_diff.getValue() : 0;
    
    if(!constraints->empty()) {
        std::lock_guard<std::mutex> guard(_filter_mutex);
        if(calculate_std)
            _filter_cache_std[fish->identity().ID()][segment] = constraints;
        else
            _filter_cache_no_std[fish->identity().ID()][segment] = constraints;
    }
    
    return constraints;
}
}

class ImageExtractor {
public:
    
    static bool is(uint32_t flags, Flag flag) {
        return flags & (uint32_t)flag;
    }
    
protected:
    GETTER_I(uint64_t, pushed_items, 0)
    
private:
    Settings _settings{};
    ska::bytell_hash_map<Frame_t, std::vector<Task>> _tasks;
    std::promise<void> _promise{};
    std::future<void> _future{_promise.get_future()};
    pv::File& _video;
    
    std::thread _thread;
    
public:
    ImageExtractor(pv::File & video,
                   auto && selector,
                   auto && partial_apply,
                   auto && callback,
                   Settings&& settings = {})
        requires requires (Query q) { { selector(q) } -> std::convertible_to<bool>; }
    :   _settings(std::move(settings)),
        _video(video),
        _thread([this,
                 selector = std::move(selector),
                 callback = std::move(callback),
                 partial_apply = std::move(partial_apply)]()
                mutable
            {
                update_thread(selector, partial_apply, callback);
            })
    { }
    
    std::future<void>& future() {
        return _future;
    }
    
    void filter_tasks() {
        size_t counter = 0;
        for(auto &task : _tasks)
            counter += task.second.size();
        print("Before filtering: ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of tasks for ", _tasks.size(), " frames.");
        
        size_t N = 0;
        for(auto &[frame, samples] : _tasks)
            N += samples.size();
        
        std::set<Frame_t> remove;
        double average = double(N) / _tasks.size();
        print("On average ", average, " items per frame.");
        
        for (auto it = _tasks.begin(); it != _tasks.end(); ++it) {
            if(it->second.size() < average) {
                remove.insert(it->first);
            }
        }
        
        print("Removing ", remove.size(), " frames.");
        
        for(auto frame : remove)
            _tasks.erase(frame);
        
        counter = 0;
        for(auto &task : _tasks)
            counter += task.second.size();
        print("After filtering: ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of tasks for ", _tasks.size(), " frames.");
    }
    
    template<typename S>
    void collect(S&& selector) {
        //! we need a lock for this, since we're walking through all individuals
        //! maybe I can split this up later, but could be dangerous.
        Tracker::LockGuard guard("ImageExtractor");
        
        // go through all individuals
        Query q;
        Task task;
        
        for (auto &[fdx, fish] : Tracker::individuals()) {
            print("Individual ", fdx, " has ", fish->frame_count(), " frames.");
            
            fish->iterate_frames(Range<Frame_t>(fish->start_frame(), fish->end_frame()),
             [&, fdx=fdx](Frame_t frame, auto& seg, const Individual::BasicStuff* basic, auto posture)
                 -> bool
             {
                q.basic = basic;
                q.posture = posture;
                
                if(selector(q)) {
                    // initialize task lazily
                    task.fdx = fdx;
                    task.bdx = basic->blob.blob_id();
                    task.segment = seg->range;
                    
                    _tasks[frame].emplace_back(std::move(task));
                }
                
                return true;
             });
        }
        
        //! finished
        size_t counter = 0;
        for(auto &task : _tasks)
            counter += task.second.size();
        
        print("Collected ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of data in ", _tasks.size(), " frames.");
    }
    
    //! The main thread of the async extractor.
    //! Here, it'll do:
    //!     1. collect tasks to be done
    //!     2. filter tasks (optional)
    //!     3. retrieve image data
    //!     4. callback() and fulfill promise
    template<typename S, typename F, typename A>
    void update_thread(S&& selector, A&& partial_apply, F&& callback) {
        try {
            Timer timer;
            collect(std::forward<S>(selector));
            print("Took ", timer.elapsed(), "s to calculate all tasks.");
            timer.reset();
            
            if(is(_settings.flags, Flag::RemoveSmallFrames)) {
                filter_tasks();
                print("Took ", timer.elapsed(), "s for filtering step.");
            }
        
            // this will take the longest, since we actually
            // need to read the video:
            _pushed_items = retrieve_image_data(std::forward<A>(partial_apply));
            
            //! we are done.
            _promise.set_value();
            callback(this);
            
        } catch(...) {
            FormatWarning("[update_thread] Rethrowing exception for main.");
            try {
                _promise.set_exception(std::current_exception());
            } catch(...) {
                FormatExcept("[update_thread] Unrecoverable exception in retrieve_image_data.");
            }
        }
    }
    
    template<typename Apply>
    uint64_t retrieve_image_data(Apply&& apply) {
        GenericThreadPool pool(_settings.num_threads);
        std::vector<Image::UPtr> images;
        
        std::mutex mutex;
        uint64_t pushed_items{0};
        const auto recognition_normalization = _settings.normalization;
        const Size2 image_size = _settings.image_size;
        const uint64_t image_bytes = image_size.width * image_size.height * 1 * sizeof(uchar);
        const uint64_t max_images_per_step = max(1u, _settings.max_size_bytes / image_bytes);
        print("Pushing ", max_images_per_step, " per step (",FileSize{image_bytes},"/image).");
        
        // distribute the tasks across threads
        distribute_vector([&](size_t i,
                              decltype(_tasks)::const_iterator start,
                              decltype(_tasks)::const_iterator end,
                              auto)
        {
            PPFrame pp;
            std::vector<Result> results;
            size_t N = 0, pushed = 0;
            for(auto it = start; it != end; ++it) {
                auto &[index, samples] = *it;
                N += samples.size();
            }
            
            for(auto it = start; it != end; ++it) {
                auto &[index, samples] = *it;
                pp.set_index(index);
                _video.read_frame(pp.frame(), index.get());
                Tracker::preprocess_frame(pp, {}, NULL, NULL);
                
                for(const auto &[fdx, bdx, range] : samples) {
                    auto blob = pp.find_bdx(bdx);
                    if(!blob) {
                        blob = pp.find_original_bdx(bdx);
                        if(!blob) {
                            //print("Cannot find ", bdx, " in frame ", index);
                            continue;
                        }
                    }
                    
                    float median_midline_length_px = 0;
                    gui::Transform midline_transform;
                    
                    if(recognition_normalization == default_config::recognition_normalization_t::posture) {
                        Tracker::LockGuard guard("normalization");
                        auto fish = Tracker::individuals().at(fdx);
                        if(fish) {
                            //auto seg = fish->segment_for(index);
                            //if(seg && seg->contains(index)) {
                                //median_midline_length_px = fish->midline_length();
                            auto filter = constraints::local_midline_length(fish, range, false);
                            median_midline_length_px = filter->median_midline_length_px;
                            
                            auto basic = fish->basic_stuff(index);
                            auto posture = fish->posture_stuff(index);
                            if(posture && basic) {
                                Midline::Ptr midline = fish->calculate_midline_for(*basic, *posture);
                                midline_transform = midline->transform(recognition_normalization);
                            }
                            //}
                        }
                    }
                    
                    auto &&[image, pos] = constraints::diff_image(recognition_normalization, blob, midline_transform, median_midline_length_px, _settings.image_size);
                    //auto &&[image, pos] = Individual::calculate_diff_image(blob, _settings.image_size);
                    //auto &&[pos, image] = blob->difference_image(*Tracker::instance()->background(), FAST_SETTINGS(track_threshold));
                    if(!image) {
                        //! can this happen?
                        FormatWarning("Cannot generate image for ", bdx, " of ", fdx, " in frame ", index,".");
                        continue;
                    }
                    
                    tf::imshow("image", image->get());
                    
                    results.emplace_back(Result{
                        .frame = index,
                        .fdx = fdx,
                        .bdx = bdx,
                        .image = std::move(image)
                    });
                    
                    if(results.size() >= max_images_per_step) {
                        // need to take a break here
                        pushed += results.size();
                        print("Taking a break in thread ", i, " after ", pushed, "/",N," items (",results.size()," being pushed at once).");
                        apply(std::move(results));
                        results.clear();
                    }
                }
            }
            
            if(!results.empty()) {
                pushed += results.size();
                print("Thread ", i, " ended. Pushing: ", pushed, "/",N," items (",results.size()," being pushed at once).");
                apply(std::move(results));
            }
            
            std::unique_lock guard(mutex);
            pushed_items += pushed;
            
        }, pool, _tasks.begin(), _tasks.end());
        
        print("distribute ended.");
        return pushed_items;
    }
    
    ~ImageExtractor() {
        _thread.join();
    }
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

} //! /extract

int main(int argc, char**argv) {
    CommandLine cmd(argc, argv);
    cmd.cd_home();
    
    print("Sizeof transform = ", sizeof(gui::Transform));
    
    DebugHeader("LOADING DEFAULT SETTINGS");
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    default_config::register_default_locations();
    
    //file::Path path("/Users/tristan/Videos/group_1.pv");
    //file::Path path("/Users/tristan/Videos/tmp/20180505_100700_body.pv");
    file::Path path("/Users/tristan/Videos/locusts/converted/four_patches_tagged_60_locusts_top_right_high_top_left_low_20220610_142144_body.pv");
    SETTING(filename) = path.remove_extension("pv");
    
    pv::File video(path);
    video.start_reading();
    if(!video.open())
        throw U_EXCEPTION("Cannot open video file ",path,".");
    
    file::Path settings_file(path.replace_extension("settings"));
    GlobalSettings::map().set_do_print(true);
    DebugHeader("LOADING ", settings_file);
    try {
        auto content = utils::read_file(settings_file.str());
        default_config::load_string_with_deprecations(settings_file, content, GlobalSettings::map(), AccessLevelType::STARTUP);
        
    } catch(const cmn::illegal_syntax& e) {
        FormatError("Illegal syntax in settings file.");
        throw;
    }
    DebugHeader("LOADED ", settings_file);
    
    Tracker tracker;
    tracker.set_average(Image::Make(video.average()));
    
    video.print_info();
    
    Output::TrackingResults results(tracker);
    try {
        results.load();
    } catch(...) {
        print("Loading failed. Analysing instead...");
        
        PPFrame pp;
        Timer timer;
        double s = 0;
        for(size_t i=0; i<video.length(); ++i) {
            
            video.read_frame(pp.frame(), i);
            pp.frame().set_index(i);
            track::Tracker::preprocess_frame(pp, {}, NULL, NULL, false);
            //Tracker::preprocess_frame(pp, tracker.active_individuals(), nullptr);
            
            //Tracker::LockGuard guard("tracking");
            tracker.add(pp);
            
            s += timer.elapsed();
            if(i % 1000 == 0)
                print(1.0 / (s / double(i)), "fps ", i, "/", video.length());
            timer.reset();
        }
    }
    
    py::ensure_started().get();
    
    try {
        VINetwork visual;
        SETTING(terminate) = false;
        
        using namespace extract;
        Timer timer;
        ImageExtractor features{
            video,
            [](const Query& q)->bool{
                return !q.basic->blob.split();
            },
            [&](std::vector<Result>&& results) {
                // partial_apply
                std::vector<Image::UPtr> images;
                images.reserve(results.size());
                
                for(auto &&r : results)
                    images.emplace_back(std::move(r.image));
                
                visual.apply(std::move(images), [results = std::move(results)](auto&& result) mutable {
                    print("\tGot response for ", results.size(), " items (with ",result.size()," items).");
                }).get();
            },
            [](auto extractor){
                // callback
                print("All done extracting. Overall pushed ", extractor->pushed_items());
                SETTING(terminate) = true;
            },
            extract::Settings{
                .flags = (uint32_t)Flag::RemoveSmallFrames,
                .image_size = Size2(80,80),
                .max_size_bytes = 1000u * 1000u * 1000u / 5u / 10u,
                .num_threads = 5u,
                .normalization = SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>()
            }
        };
        
        
        while(features.future().wait_for(std::chrono::milliseconds(1)) != std::future_status::timeout) {
            tf::show();
        }
        
        features.future().get();
        print("Took ", timer.elapsed(), "s");
        
    } catch(const SoftExceptionImpl& e) {
        print("Python runtime error: ", e.what());
    }
    
    py::quit();
}
