#pragma once
#include <commons.pc.h>
#include <core/TileCoordinates.h>
#include <misc/PackLambda.h>
#include <misc/Image.h>
#include <pv.h>
#include <core/DetectionTypes.h>
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
    std::vector<track::SourceRect> tiles;
    
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
            //std::unique_lock guard(_task_mutex);
            {
                std::unique_lock image_guard(_mutex);
#ifndef NDEBUG
                if(_images.empty())
                    FormatError("Images empty: ", BaseTask<Data>::_weight);
#endif
                packet.swap(_images);
                _images.clear();
                BaseTask<Data>::_weight = 0;
            }

            fn(std::move(packet));
            
#ifndef NDEBUG
            for(auto &data : packet) {
                if(data) {
                    Print("Data not null");
                }
            }
#endif
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
protected:
    bool _input_closed{false};
    std::mutex _pause_mutex;
private:
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

    // Returns false when closed input rejects a new item. An empty item can still flush the batch.
    virtual bool enqueue(Data&& ptr) {
        std::scoped_lock g(_future_mutex);
        if (_future.valid())
            _future.get();

        std::unique_lock guard(_mutex);
        std::unique_lock input_guard(_pause_mutex);
        if(ptr && _input_closed)
            return false;

        if(not _create) {
            thread_print("[WARNING] _create method not set.");
            return true;
        }
        
        if(not _c) {
            _create();
            assert(_c != nullptr);
            if(not _c)
                return true;
        }
        
        if(ptr)
            _c->push(std::move(ptr));
        input_guard.unlock();
        
        if(_c->weight() < _weight_limit.load()) {
            return true;
        }
        
        _future = std::async(std::launch::async, [this]() {
            set_thread_name("pipeline_async");
            std::unique_lock guard(_mutex);
            if(not _c || _c->weight() == 0)
                return;
            
            (*_c)();
        });
        return true;
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

template<typename Data>
class PipelineManager : public BasicManager<Data> {
    bool _paused{false};
    PersistentCondition _pause_variable;

public:
    /// Rejects new items and wakes a paused producer. Queued items can still flush.
    void close_input() {
        std::unique_lock guard(this->_pause_mutex);
        this->_input_closed = true;
        _pause_variable.notify();
    }

    template<typename... Args>
    PipelineManager(double weight_limit, bool start_paused, Args... args)
        : BasicManager<Data>(weight_limit, std::forward<Args>(args)...),
          _paused(start_paused)
    { }
    
    bool enqueue(Data&& data) override {
        //if(not data)
        //    return;
        if(data) {
            {
                std::unique_lock p(this->_pause_mutex);
                while(_paused && not this->_input_closed)
                    _pause_variable.wait(p);
            }
            
            return BasicManager<Data>::enqueue(std::move(data));
        } else {
            return BasicManager<Data>::enqueue({});
        }
    }
    
    void set_paused(bool v) {
        {
            std::unique_lock g(this->_pause_mutex);
            _paused = v;
            _pause_variable.notify();
        }
    }
};

}
