#pragma once
#include <commons.pc.h>
#include <misc/PackLambda.h>
#include <misc/Image.h>
#include <pv.h>
#include <misc/DetectionTypes.h>
#include <misc/Buffers.h>
#include <misc/ThreadManager.h>

namespace cmn {

/*class BaseTask {
protected:
    double _weight{0};
public:
    using Ptr = std::unique_ptr<BaseTask>;

    double weight() const noexcept { return _weight; }

    BaseTask(double w) : _weight(w) {}
    virtual ~BaseTask() {}
    virtual void operator()() = 0;
    auto operator<=>(const BaseTask& other) const noexcept {
        return weight() <=> other.weight();
    }
};

template<typename F, typename R = typename detail::return_type<F>::type>
class Task : public BaseTask {
    package::F<void()> task;
    std::promise<R> p;
public:

    template<typename... Args>
    Task(double weight, F fn, Args... args)
      : BaseTask(weight),
        task([this, bound = std::bind(std::forward<F>(fn), std::forward<Args>(args)...)]()
        mutable {
            try {
                if constexpr(std::same_as<R, void>) {
                    bound();
                    p.set_value();
                } else {
                    p.set_value(bound());
                }
            } catch(...) {
                p.set_exception(std::current_exception());
            }
        })
    {
    }
    
    auto get_future() {
        return p.get_future();
    }

    void operator()() override {
        task();
    }
};

template<typename... Args>
static auto MakeTask(Args... args) {
    auto task = new Task(std::forward<Args>(args)...);
    return std::unique_ptr<BaseTask>(task);
}*/

template<typename Data>
class BaseTask {
protected:
    double _weight{0};
public:
    using Ptr = std::unique_ptr<BaseTask>;

    double weight() const noexcept { return _weight; }

    BaseTask(double w) : _weight(w) {}
    virtual ~BaseTask() {}
    virtual void operator()() = 0;
    auto operator<=>(const BaseTask& other) const noexcept {
        return weight() <=> other.weight();
    }

    virtual void push(Data&& ptr) = 0;
};

struct SegmentationData {
    Image::Ptr image;
    pv::Frame frame;
    std::vector<Bounds> tiles;
    
    struct Assignment {
        size_t clid;
        float p;
    };
    
    std::vector<Assignment> predictions;
    //std::vector<std::vector<Vec2>> outlines;
    std::vector<track::detect::Keypoint> keypoints;
    
    operator bool() const {
        return image != nullptr;
    }

    SegmentationData();
    SegmentationData(SegmentationData&& other) = default;
    SegmentationData(Image::Ptr&& original);
    SegmentationData& operator=(SegmentationData&&);
    ~SegmentationData();
    
    Frame_t original_index() const;
    Frame_t written_index() const;
    
    std::string toStr() const;
    static std::string_view class_name() {
        return "SegmentationData";
    }
};

template<typename Data>
class ImageArray : public BaseTask<Data> {
    using R = void;
    using Vector = std::vector<Data>;
    package::packaged_func<R> task;
    Vector _images;
    std::mutex _mutex, _task_mutex;

public:
    template<typename F>
    ImageArray(F fn)
      : BaseTask<Data>(0),
        task([this, fn = std::move(fn)]() mutable {
            decltype(_images) packet;
            {
                std::scoped_lock guard(_mutex, _task_mutex);
#ifndef NDEBUG
                if(_images.empty())
                    FormatError("Images empty: ", BaseTask<Data>::_weight);
#endif
                packet.swap(_images);
                BaseTask<Data>::_weight = 0;
            }

            std::lock_guard guard(_task_mutex);
            fn(std::move(packet));
            
            for(auto &data : packet) {
                if(data) {
                    Print("Data not null");
                }
            }
            packet.clear();
        })
    { }

    void push(Data&& ptr) override {
        std::unique_lock guard(_mutex);
        BaseTask<Data>::_weight++;
        _images.push_back(std::move(ptr));
    }
    
    void operator()() override {
        task();
    }
};

template<typename Data>
class BasicManager {
    typename BaseTask<Data>::Ptr _c;
    mutable std::shared_mutex _mutex;
    std::mutex _future_mutex;
    std::future<void> _future;
    std::atomic<double> _weight_limit{ 0 };
    std::function<void()> _create;
    
    template <typename... Args>
    auto bind_arguments_to_lambda(Args&&... args) {
        auto tup = std::make_tuple(std::forward<Args>(args)...);
        auto func = [this](auto&&... args) {
            _c = std::make_unique<ImageArray<Data>>(std::forward<decltype(args)>(args)...);
        };
        return [tup = std::move(tup), func = std::move(func)]() mutable {
            std::apply(func, tup);
        };
    }
    
public:
    template<typename... Args>
    BasicManager(double weight_limit, Args... args)
        : _weight_limit(weight_limit), _create(bind_arguments_to_lambda(std::forward<Args>(args)...))
    {
        CMN_ASSERT(not Data{}, "Default constructed values should evaluate to 'false'.");
    }
    
    virtual ~BasicManager() {
        {
            std::unique_lock guard(_mutex);
            _create = {};
        }
        
#ifdef _WIN32
        clean_up();
#endif
    }
    
    bool is_terminated() const {
        //std::unique_lock g(_mutex);
        return _c == nullptr;
    }
    
    void clean_up() {
        {
            std::scoped_lock guard(_future_mutex);
            if(_future.valid())
                _future.get();
            _future = {};
        }
        
        std::unique_lock g(_mutex);
        _c = nullptr;
    }

    virtual void enqueue(Data&& ptr) {
        std::scoped_lock g(_future_mutex);
        if (_future.valid())
            _future.get();
        
        std::unique_lock guard(_mutex);
        if(not _create) {
            thread_print("[WARNING] _create method not set.");
            return;
        }
        
        if(not _c) {
            _create();
            assert(_c != nullptr);
            if(not _c)
                return;
        }
        
        if(ptr)
            _c->push(std::move(ptr));
        
        if(_c->weight() < _weight_limit.load()) {
            return;
        }
        
        _future = std::async(std::launch::async, [this]() {
            set_thread_name("pipeline_async");
            std::unique_lock guard(_mutex);
            if(not _c || _c->weight() == 0)
                return;
            
            (*_c)();
        });
    }

    void set_weight_limit(double w) {
        {
            //std::unique_lock guard(_mutex);
            _weight_limit = w;
        }

        //update();
        enqueue({});
    }
};

template<typename Data, bool init_paused = false>
class PipelineManager : public BasicManager<Data> {
    bool _paused{false};
    std::mutex _pause_mutex;
    PersistentCondition _pause_variable;
    
public:
    template<typename... Args>
    PipelineManager(double weight_limit, Args... args)
        : BasicManager<Data>(weight_limit, std::forward<Args>(args)...),
          _paused(init_paused)
    { }
    
    void enqueue(Data&& data) override {
        //if(not data)
        //    return;
        if(data) {
            std::unique_lock p(_pause_mutex);
            if(_paused)
                _pause_variable.wait(p, [&](){ return not _paused; });
            
            BasicManager<Data>::enqueue(std::move(data));
        } else {
            BasicManager<Data>::enqueue({});
        }
    }
    
    void set_paused(bool v) {
        {
            std::unique_lock g(_pause_mutex);
            _paused = v;
            _pause_variable.notify();
        }
    }
};

}
