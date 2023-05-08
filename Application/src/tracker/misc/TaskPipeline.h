#pragma once
#include <commons.pc.h>
#include <misc/PackLambda.h>
#include <misc/Image.h>
#include <pv.h>

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



namespace OverlayBuffers {

Image::Ptr get_buffer();
void put_back(Image::Ptr&& ptr);

}

struct SegmentationData {
    Image::Ptr image;
    pv::Frame frame;
    std::vector<Bounds> tiles;
    
    struct Assignment {
        size_t clid;
        float p;
    };
    
    std::map<pv::bid, Assignment> predictions;
    std::vector<std::vector<Vec2>> outlines;
    
    operator bool() const {
        return image != nullptr;
    }

    SegmentationData() = default;
    SegmentationData(SegmentationData&& other) = default;
    SegmentationData(Image::Ptr&& original) : image(std::move(original)) {}
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
    std::mutex _mutex;

public:
    template<typename F>
    ImageArray(F fn)
      : BaseTask<Data>(0),
        task([this, fn = std::move(fn)]() mutable {
            static std::mutex _mu;
            static decltype(_images) packet;
            {
                std::unique_lock guard(_mutex);
                std::swap(_images, packet);
                _images.clear();
                BaseTask<Data>::_weight = 0;
            }

            std::lock_guard guard(_mu);
            fn(std::move(packet));
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
class PipelineManager {
    typename BaseTask<Data>::Ptr _c;
    std::shared_mutex _mutex;
    std::future<void> _future;
    const double _weight_limit{0};

public:
    template<typename... Args>
    PipelineManager(double weight_limit, Args... args)
        : _c(new ImageArray<Data>(std::forward<Args>(args)...)), _weight_limit(weight_limit)
    {
    }

    ~PipelineManager() {
        /*std::unique_lock guard(_mutex);
        if(_c && _c->weight() > 0) {
            printf("Executing upon destruction with weight %f\n", _c->weight());
            (*_c)();
        } else if(_c) {
            printf("Not executing since weight == %f\n", _c->weight());
        } else
            printf("Not executing since object is nullptr\n");*/
    }
    
    void clean_up() {
        std::unique_lock guard(_mutex);
        if(_future.valid()) {
            guard.unlock();
            _future.get();
            guard.lock();
        }
        _c = nullptr;
    }

    void enqueue(std::vector<Data>&& v) {
        {
            std::shared_lock guard(_mutex);
            if(not _c)
                return;
            
            {
                for (auto&& ptr : v)
                    _c->push(std::move(ptr));
                v.clear();
            }
        }
        update();
    }

    void enqueue(Data&& ptr) {
        {
            std::shared_lock guard(_mutex);
            if(not _c)
                return;
            
            {
                assert(_c != nullptr);
                _c->push(std::move(ptr));
            }
        }
        update();
        
    }

private:
    bool update() {
        // wait for enough tasks
        std::shared_lock guard(_mutex);
        if(not _c || _c->weight() < _weight_limit) {
            return true;
        }

        if (_future.valid())
            _future.get();

        //if (not _future.valid()) 
        {
            _future = std::async(std::launch::async, [this]() {
                set_thread_name("pipeline_async");
                std::shared_lock guard(_mutex);
                (*_c)();
            });
        }
        //else {//if (_future.wait_for(std::chrono::microseconds(0)) == std::future_status::ready) {
        //    _future.get();
        //    assert(not _future.valid());
        //}
        return false;
    }
};

}
