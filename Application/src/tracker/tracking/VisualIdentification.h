#pragma once

#include <commons.pc.h>
#include <python/Network.h>
#include <tracker/misc/default_config.h>
#include <misc/Image.h>
#include <misc/PythonWrapper.h>
#include <misc/idx_t.h>
#include <file/Path.h>
#include <tracking/TrainingData.h>
#include <tracking/Stuffs.h>
#include <misc/DetectionTypes.h>

namespace Python {

ENUM_CLASS(TrainingMode,
    None,
    Restart,
    Apply,
    Continue,
    Accumulate,
    LoadWeights
)

template<typename T>
concept image_ptr =    cmn::_clean_same<T, cmn::Image::SPtr>
                    || cmn::_clean_same<T, cmn::Image::Ptr>;

class VINetwork {
protected:
    Network _network;
    using callback_t = cmn::package::F<void(std::vector<std::vector<float>>&&,std::vector<float>&&)>;
    std::shared_ptr<track::TrainingData> _last_training_data;
    
public:
    struct Status {
        bool busy{false};
        track::vi::VIWeights weights;
        
        auto operator<=>(const Status& other) const = default;
    };
    
protected:
    std::mutex _data_mutex;
    Status _status;
    std::map<const char*, std::function<void(float percent, const std::string&)>> _callbacks;
    std::function<bool()> abort_function, skip_function;
    
    using weights_list_t = std::optional<std::set<track::vi::VIWeights>>;
    std::mutex _weights_mutex;
    std::optional<std::shared_future<weights_list_t>> _loading_available_weights;
    weights_list_t _available_weights;
    
private:
    VINetwork();
    
public:
    //! VINetwork can only be instantiated once. The life-time is managed
    //! internally.
    static std::shared_ptr<VINetwork> instance();
    
    //! Returns the current status.
    static Status status();
    
    static void set_abort_training(std::function<bool()>);
    static void set_skip_button(std::function<bool()>);
    
    //! A path set according to output_dir and filename "_weights.npz"
    static cmn::file::Path network_path();
    
    //! Generated based on classes()
    static size_t number_classes();
    
    //! Generated based on track_max_individuals, a set of all ID numbers
    static std::set<track::Idx_t> classes();
    
    //! TODO: MISSING <placeholder>
    static float percent();
    
    static void add_percent_callback(const char* identifier, std::function<void(float percent, const std::string&)>&&);
    
    bool train(std::shared_ptr<track::TrainingData> data,
               const cmn::FrameRange& global_range,
               TrainingMode::Class load_results,
               uchar gpu_max_epochs,
               bool dont_save,
               float *worst_accuracy_per_class,
               int accumulation_step);
    
    //! Initializes network, and loads weights if available
    //! (according to network_path)
    track::vi::VIWeights load_weights(track::vi::VIWeights&& = track::vi::VIWeights{});
    
    //! Checks network_path() and sees if the file is available
    static bool weights_available();
    
    static std::optional<std::set<track::vi::VIWeights>> get_available_weights();
    
    //! Based on Basic/Posture information, determines whether an image
    //! is eligable for apply/recognition
    static bool is_good(const track::BasicStuff*, const track::PostureStuff*);
    
    //! A copy needs to be made :-(
    template<typename T>
        requires image_ptr<T>
    auto probabilities(const std::vector<T>& images) {
        return probabilities(std::vector<T>(images));
    }
    
    //! Return a list of probabilities per image passed.
    //! The list of probabilities is NxM, where M is the #fish
    //! and N #images.
    template<typename T>
        requires image_ptr<T>
    std::vector<float> probabilities(std::vector<T>&& images) {
        std::promise<std::vector<float>> prom;
        auto future = prom.get_future();
        
        probabilities(std::move(images), [N = images.size(), prom = std::move(prom)](auto &&values, auto &&indexes) mutable
        {
            try {
                prom.set_value(transform_results(N, std::move(indexes), std::move(values)));
            } catch(...) {
                prom.set_exception(std::current_exception());
                //throw;
            }
        }).get();
        
        if(not future.valid()) {
            throw cmn::SoftException("Invalid future.");
        }
        return future.get();
    }
    
    //! Holds a number of samples + probability values
    //! from a prediction.
    struct Average {
        float samples; // samples / value in values
        std::vector<float> values; // length = N identities
        
        std::string toStr() const;
    };
    
    template<typename T>
        requires image_ptr<T>
    std::map<track::Idx_t, Average> paverages(
              const std::vector<track::Idx_t>& ids,
              std::vector<T>&& images)
    {
        auto probs = probabilities(std::move(images));
        
        using namespace track;
        
        const auto N = number_classes();
        const auto identities = classes();
        
        std::map<Idx_t, Average> averages;
        
        for(size_t i=0; i<ids.size(); ++i) {
            auto start = probs.begin() + (i    ) * N;
            auto end   = probs.begin() + (i + 1) * N;
            
            auto& [samples, values] = averages[ids[i]];
            assert(*start >= 0);
            
            if(values.empty()) {
                values.resize(N);
                samples = 0;
            }
            
            ++samples;
            
            assert(values.size() == N);
            std::transform(start, end, values.begin(), values.begin(), std::plus<>{});
        }
        
        for(auto &[k, v] : averages)
            std::transform(v.values.begin(), v.values.end(), v.values.begin(), [N = float(v.samples)](auto v){ return v / N; });
        
        return averages;
    }
    
    //! Schedules a probabilities retrieval/prediction with a callback.
    //! Assumes that weights are valid and a network has been loaded.
    //! Can be async / sync depending on whether it was called in the correct thread.
    //! @param images the images to be identified
    //! @param callback is a callable that takes void(vector<vector<float>>&&, vector<float>&&)
    //! @throws SoftException if no weights have been loaded yet / trained
    [[nodiscard]] std::future<void> probabilities(auto&& images, auto&& callback) {
        return Python::schedule(PackagedTask{
            ._network = &_network,
            ._task = PromisedTask([callback = std::move(callback),
                      images = std::move(images)]()
                mutable
            {
                callback_t pc([callback = std::move(callback)](
                    std::vector<std::vector<float>>&& ps,
                    std::vector<float>&& indexes)
                   mutable
                {
                    callback(std::move(ps), std::move(indexes));
                });
                
                set_variables(std::move(images), std::move(pc));
            }),
            ._can_run_before_init = false
        });
    }
    
    static std::future<void> clear_caches();
    static std::future<void> unload_weights();
    
private:
    static void set_variables_internal(auto&&, callback_t&&);
    static void set_variables(std::vector<cmn::Image::Ptr>&&, callback_t&&);
    static void set_variables(std::vector<cmn::Image::SPtr>&&, callback_t&&);
    void setup(bool force);
    
    void set_work_variables(bool force);
public:
    static void unset_work_variables();
    static void set_status(Status);
    
private:
    void reinitialize_internal();
    std::optional<track::vi::VIWeights> load_weights_internal(track::vi::VIWeights&&);
    
public:
    static std::vector<float> transform_results(
         const size_t N,
         std::vector<float>&& indexes,
         std::vector<std::vector<float>>&& values);
};

}
