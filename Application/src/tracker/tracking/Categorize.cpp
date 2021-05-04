#include "Categorize.h"
#include <tracking/Individual.h>
#include <gui/DrawStructure.h>
#include <gui/gui.h>
#include <tracking/Recognition.h>
#include <tracking/Accumulation.h>
#include <gui/types/Tooltip.h>
#include <python/GPURecognition.h>
#include <random>
#include <misc/default_settings.h>

namespace track {
namespace Categorize {

// indexes in _samples array
std::unordered_map<const Individual::SegmentInformation*, size_t> _used_indexes;

// holds original samples
std::vector<Sample::Ptr> _samples;

// holds all original Labels
std::unordered_map<Label::Ptr, std::vector<Sample::Ptr>> _labels;

std::random_device rd;

namespace Work {
    std::atomic_bool terminate = false, _learning = false;
    std::mutex _mutex;
    std::mutex _recv_mutex;
    std::condition_variable _variable, _recv_variable;
    std::queue<Sample::Ptr> _generated_samples;
    
    std::condition_variable _learning_variable;
    std::mutex _learning_mutex;

    static void add_training_sample(const Sample::Ptr& sample);
    static void start_learning();
    void loop();

    Sample::Ptr retrieve();

    size_t& requested_samples() {
        static size_t _request = 0;
        return _request;
    }
    
    bool& visible() {
        static bool _visible = false;
        return _visible;
    }

    bool& initialized_apply() {
        static bool _init = false;
        return _init;
    }
    
    auto& queue() {
        static std::queue<LearningTask> _tasks;
        return _tasks;
    }
    
    auto& status() {
        static std::string _status;
        return _status;
    }
    
    std::unique_ptr<std::thread> thread;
    constexpr float good_enough() {
        return 0.75;
    }
    
    void work() {
        set_thread_name("Categorize::work_thread");
        Work::loop();
    }
    
    size_t num_ready() {
        std::lock_guard guard(_mutex);
        return _generated_samples.size();
    }
    
    float& best_accuracy() {
        static float _a = 0;
        return _a;
    }
    
    void set_best_accuracy(float a) {
        best_accuracy() = a;
    }

    void add_task(LearningTask&& task) {
        Debug("Adding learning task of type %d", (int)task.type);

        {
            std::lock_guard guard(_learning_mutex);
            queue().push(std::move(task));
        }
        
        Work::_learning_variable.notify_one();
    }

    auto& task_queue() {
        static std::queue<std::function<void()>> _queue;
        return _queue;
    }
};

Sample::Sample(std::vector<long_t>&& frames, const std::vector<Image::Ptr>& images)
    :
        _frames(std::move(frames)),
        _images(images)
{
    assert(!_images.empty());
}

std::set<std::string> DataStore::label_names() {
    std::lock_guard guard(mutex());
    std::set<std::string> _names;
    for(auto &[l, s] : _labels)
        _names.insert(l->name);
    return _names;
}

Label::Ptr DataStore::label(const char* name) {
    std::lock_guard guard(mutex());
    for(auto &[l, s] : _labels) {
        if(l->name == name)
            return l;
    }
    
    auto l = Label::Make(std::string(name));
    _labels.insert({l, {}});
    return l;
}

Label::Ptr DataStore::label(int ID) {
    std::lock_guard guard(mutex());
    for(auto &[l, s] : _labels) {
        if(l->id == ID) {
            return l;
        }
    }
    Warning("ID %d not found", ID);
    return nullptr;
}

const Sample::Ptr& DataStore::random_sample(Idx_t fid) {
    static std::mt19937 mt(rd());
    std::shared_ptr<Individual::SegmentInformation> segment;
    Individual *fish;
    
    {
        Tracker::LockGuard guard("Categorize::random_sample");
        auto iit = Tracker::instance()->individuals().find(fid);
        if (iit != Tracker::instance()->individuals().end()) {
            fish = iit->second;
            auto& basic_stuff = fish->basic_stuff();
            if (basic_stuff.empty())
                return Sample::Invalid();

            std::uniform_int_distribution<remove_cvref<decltype(fish->frame_segments())>::type::difference_type> sample_dist(0, fish->frame_segments().size() - 1);
            auto it = fish->frame_segments().begin();
            std::advance(it, sample_dist(mt));
            segment = *it;
        }
    }
    
    if(!segment)
        return Sample::Invalid();
    
    return sample(segment, fish, 150u, 50u);
}

Sample::Ptr DataStore::get_random() {
    static std::mt19937 mt(rd());
    
    std::set<Idx_t> individuals(extract_keys(Tracker::instance()->individuals()));
    if(individuals.empty())
        return {};
    
    std::uniform_int_distribution<size_t> individual_dist(0, individuals.size()-1);
    
    auto fid = individual_dist(mt);
    return DataStore::random_sample(Idx_t(fid));
}

DataStore::Composition DataStore::composition() {
    std::lock_guard guard(mutex());
    Composition c;
    for (auto &[key, samples] : _labels) {
        for(auto &s : samples)
            c._numbers[key->name] += s->_images.size();
    }
    return c;
}

bool DataStore::Composition::empty() const {
    size_t N = 0;
    for (auto &[k, v] : _numbers) {
        N += v;
    }
    return N == 0;
}

std::unordered_map<Frame_t, std::unordered_map<uint32_t, Label::Ptr>> _probability_cache;
std::unordered_map<Idx_t, std::unordered_map<std::shared_ptr<Individual::SegmentInformation>, Label::Ptr>> _interpolated_probability_cache;

DataStore::const_iterator DataStore::begin() {
    return _samples.begin();
}

DataStore::const_iterator DataStore::end() {
    return _samples.end();
}

void DataStore::set_label(Frame_t idx, uint32_t bdx, const Label::Ptr& label) {
    std::lock_guard guard(mutex());
    if (_probability_cache[idx].count(bdx)) {
        auto str = Meta::toStr(_probability_cache[idx]);
        Warning("Cache already contains blob in frame %d.\n%S", bdx, (int)idx, &str);
    }
    _probability_cache[idx][bdx] = label;
}

void DataStore::set_label(Frame_t idx, const pv::CompressedBlob* blob, const Label::Ptr& label) {
    uint32_t bdx;
    /*if(blob->parent_id != -1)
        bdx = uint32_t(blob->parent_id);
    else*/
        bdx = blob->blob_id();
    
    set_label(idx, bdx, label);
}

Label::Ptr DataStore::label_interpolated(Idx_t fish, Frame_t frame) {
    Tracker::LockGuard guard("DataStore::label_interpolated");
    
    auto it = Tracker::individuals().find(fish);
    if(it == Tracker::individuals().end()) {
        Warning("Individual %d not found.", fish._identity);
        return nullptr;
    }
    
    auto kit = it->second->iterator_for(frame);
    if(kit == it->second->frame_segments().end()) {
        Warning("Individual %d, cannot find frame %d.", fish._identity, frame._frame);
        return nullptr;
    }
    
    if((*kit)->contains(frame)) {
        auto idx = (*kit)->basic_stuff(frame);
        if(idx != -1) {
            auto &basic = it->second->basic_stuff().at(idx);
            auto l = label(frame, &basic->blob);
            if(l)
                return l;
            
            // interpolate
            Label::Ptr before = nullptr;
            long_t index_before = -1;
            Label::Ptr after = nullptr;
            long_t index_after = -1;
            
            for(auto index : (*kit)->basic_index) {
                if(index == idx)
                    continue;
                
                assert(index > -1);
                auto &basic = it->second->basic_stuff().at(index);
                auto l = label(frame, &basic->blob);
                if(l && index < idx) {
                    before = l;
                    index_before = index;
                } else if(l && index > idx) {
                    after = l;
                    index_after = index;
                    break;
                }
            }
            
            if(before && after) {
                if(idx - index_before >= (index_after - index_before + 1) * 0.5) {
                    return after;
                } else
                    return before;
                
            } else if(before) {
                return before;
            } else
                return after;
        }
        
    } else {
        Warning("Individual %d does not contain frame %d.", fish._identity, frame._frame);
    }
    
    Warning("Individual %d not found. Other reason?", fish._identity);
    return nullptr;
}

Label::Ptr DataStore::label(Frame_t idx, uint32_t bdx) {
    std::lock_guard guard(mutex());
    auto fit = _probability_cache.find(idx);
    if(fit != _probability_cache.end()) {
        auto sit = fit->second.find(bdx);
        if(sit != fit->second.end()) {
            return sit->second;
        }
    }
    return nullptr;
}

Label::Ptr DataStore::label(Frame_t idx, const pv::CompressedBlob* blob) {
    return label(idx, /*blob->parent_id != -1 ? uint32_t(blob->parent_id) :*/ blob->blob_id());
}

constexpr size_t per_row = 4;

void Work::add_training_sample(const Sample::Ptr& sample) {
    try {
        Work::start_learning();
        
        LearningTask task;
        task.sample = sample;
        task.type = LearningTask::Type::Training;
        
        Work::add_task(std::move(task));
        
    } catch(...) {
        
    }
}

void terminate() {
    if(Work::thread) {
        Work::terminate = true;
        Work::_learning = false;
        Work::_learning_variable.notify_all();
        Work::_variable.notify_all();
        Work::thread->join();
        Work::thread = nullptr;
    }
}

void show() {
    if(!Work::visible()) {
        Work::set_state(Work::State::SELECTION);
        Work::visible() = true;
    }
}

void hide() {
    Work::visible() = false;
}

using namespace gui;
struct Row;

struct Cell {
private:
    std::vector<Layout::Ptr> _buttons;
    GETTER(std::shared_ptr<HorizontalLayout>, button_layout)
    GETTER_SETTER(bool, selected)
    
public:
    Row *_row;
    size_t _index;
    Sample::Ptr _sample;
    double _animation_time = 0;
    size_t _animation_index = 0;
    
    // gui elements
    std::shared_ptr<ExternalImage> _image;
    std::shared_ptr<StaticText> _text;
    std::shared_ptr<Entangled> _block;
    
public:
    Cell();
    ~Cell();
    
    void add(const Layout::Ptr& b) {
        _buttons.emplace_back(b);
        _button_layout->add_child(b);
    }
    
    void set_sample(const Sample::Ptr& sample);
    
    void update(float s) {
        for(auto &c : _buttons) {
            c.to<Button>()->set_text_clr(White.alpha(235 * s));
            c.to<Button>()->set_line_clr(Black.alpha(200 * s));
            c.to<Button>()->set_fill_clr(DarkCyan.exposure(s).alpha(150 * s));
        }
        
        if(_sample) {
            auto text = "<nr>"+Meta::toStr(_animation_index)+"</nr>/<nr>"+Meta::toStr(_sample->_images.size())+"</nr>";
            
            std::lock_guard guard(Work::_recv_mutex);
            if(!_sample->_probabilities.empty()) {
                std::map<std::string, float> summary;
                
                for (auto &[l, v] : _sample->_probabilities) {
                    if(l)
                        summary[l->name] = v;
                }
                
                text = settings::htmlify(Meta::toStr(summary)) + "\n" + text;
                
            } else if(!_sample->_requested) {
                if(Work::best_accuracy() >= Work::good_enough()) {
                    _sample->_requested = true;
                    
                    LearningTask task;
                    task.sample = _sample;
                    task.type = LearningTask::Type::Prediction;
                    task.callback = [](const LearningTask& task) {
                        std::lock_guard guard(Work::_recv_mutex);
                        for(size_t i=0; i<task.result.size(); ++i) {
                            task.sample->_probabilities[DataStore::label(task.result.at(i))] += float(1);
                        }
                        
                        auto str0 = Meta::toStr(task.sample->_probabilities);
                        for(auto &[k, v] : task.sample->_probabilities)
                            v /= float(task.result.size());
                        
                        auto str1 = Meta::toStr(task.sample->_probabilities);
                        Debug("%lu: %S -> %S", task.result.size(), &str0, &str1);
                    };
                    
                    Work::add_task(std::move(task));
                }
                
            } else
                text += " <key>(pred.)</key>";
            
            _text->set_txt(text);
        }
        
        _image->set_color(White.alpha(200 + 55 * s));
        _text->set_alpha(0.1 + s * 0.9);
        
        auto rscale = _button_layout->parent() ? _button_layout->parent()->stage()->scale().reciprocal().mul(_block->scale().reciprocal()) : Vec2(1);
        _text->set_scale(rscale);
        _button_layout->set_scale(rscale);
        
        //_text->set_base_text_color(White.alpha(100 + 155 * s));
        _button_layout->auto_size(Margin{0, 0});
        _text->set_pos(Vec2(5, _block->height() - 5));
    }
    
    const Bounds& bounds() {
        return _block->global_bounds();
    }
};

struct Row {
    int index;
    
    std::vector<Cell> _cells;
    std::shared_ptr<HorizontalLayout> layout;
    
    Row(int i)
        : index(i), layout(std::make_shared<HorizontalLayout>())
    { }
    
    void init(size_t additions) {
        _cells.clear();
        _cells.resize(additions);
        
        size_t i=0;
        for(auto &cell : _cells) {
            layout->add_child(Layout::Ptr(cell._block));
            cell._row = this;
            cell._index = i++;
        }
        
        layout->set_origin(Vec2(0.5));
        layout->set_background(Transparent, White.alpha(125));
    }
    
    Cell& cell(size_t i) {
        assert(length() > i);
        return _cells.at(i);
    }
    
    size_t length() const {
        assert(layout);
        return layout->children().size();
    }
    
    void update(DrawStructure& base, double dt) {
        if(!layout->parent())
            return;
        
        for (size_t i=0; i<length(); ++i) {
            auto &cell = this->cell(i);
            
            if(cell._sample) {
                auto d = euclidean_distance(base.mouse_position(), cell.bounds().pos() + cell.bounds().size() * 0.5) / (layout->parent()->global_bounds().size().length() * 0.45);
                cell._block->set_scale(Vec2(1.25 + 0.35 / (1 + d * d)) * (cell.selected() ? 1.5 : 1));
                
                const double seconds_for_all_samples = (cell._image->hovered() ? 15.0 : 2.0);
                const double samples_per_second = cell._sample->_images.size() / seconds_for_all_samples;
                
                cell._animation_time += dt * samples_per_second;
                
                if(size_t(cell._animation_time) != cell._animation_index) {
                    cell._animation_index = size_t(cell._animation_time);
                    
                    if(cell._animation_index >= cell._sample->_images.size()) {
                        cell._animation_index = 0;
                        cell._animation_time = 0;
                    }
                    
                    cell._image->update_with(*cell._sample->_images.at(cell._animation_index));
                    cell._block->auto_size(Margin{0, 0});
                }
                
            } else {
                //std::fill(cell._image->source()->data(), cell._image->source()->data() + cell._image->source()->size(), 0);
            }
            
            auto s = min(1, cell._block->scale().x / 1.5);
            s = SQR(s) * SQR(s);
            s = SQR(s) * SQR(s);
            
            cell.update(s);
        }
    }
    
    void update(size_t cell_index, const Sample::Ptr& sample) {
        auto &cell = this->cell(cell_index);
        cell.set_sample(sample);
        
        layout->auto_size(Margin{0, 0});
    }
    
    bool empty() const {
        return layout->empty();
    }
};

void Cell::set_sample(const Sample::Ptr &sample) {
    _sample = sample;
    
    if(!sample) {
        std::fill(_image->source()->data(),
                  _image->source()->data() + _image->source()->size(),
                  0);
    } else {
        _image->update_with(*_sample->_images.front());
        _animation_time = 0;
        _animation_index = 0;
    }
    
    double s = 0.5 / double(_row->_cells.size());
    auto base = button_layout()->stage();
    if(base) {
        if(base->width() < base->height())
            _image->set_scale(Vec2(base->width() * base->scale().x * s / _image->width()).div(base->scale()));
        else
            _image->set_scale(Vec2(base->height() * base->scale().y * s / _image->height()).div(base->scale()));
    }
}

static VerticalLayout layout;
static auto desc_text = Layout::Make<StaticText>();
static std::array<Row, 3> rows { Row(0), Row(1), Row(2) };

Sample::Ptr Work::retrieve() {
    _variable.notify_one();
    
    Sample::Ptr sample;
    //do
    {
        {
            std::unique_lock guard(_mutex);
            if(!_generated_samples.empty()) {
                sample = std::move(_generated_samples.front());
                _generated_samples.pop();
            }
        }
        
        Work::_variable.notify_one();
        
        if(sample != Sample::Invalid() && GUI::instance()) {
            /**
             * Search current rows and cells to see whether the sample is already assigned
             * to any of the cells.
             */
            std::lock_guard gui_guard(GUI::instance()->gui().lock());
            for(auto &row : rows) {
                for(auto &c : row._cells) {
                    if(c._sample == sample) {
                        sample = Sample::Invalid();
                        break;
                    }
                }
            }
        }
        
    } //while (sample == Sample::Invalid());
    
    return sample;
}

struct NetworkApplicationState {
    Individual *fish;
    
    //! basically the iterator, but not using pointers here,
    /// in case i want to change the iterator type later.
    Individual::segment_map::difference_type offset = 0;
    
    //! start the next prediction task
    void next() {
        std::shared_ptr<Individual::SegmentInformation> segment;
        LearningTask task;
        task.type = LearningTask::Type::Prediction;

        Individual::segment_map segments;

        {
            Tracker::LockGuard guard("next()");
            if (size_t(offset) == fish->frame_segments().size()) {
                Debug("Finished %d", fish->identity().ID());
                return; // no more segments
            }
            else if (size_t(offset) > fish->frame_segments().size()) {
                return; // probably something changed and we are now behind
            }

            segments = fish->frame_segments();
        }

        auto it = segments.begin();
        std::advance(it, offset);

        do {
            if (it == segments.end())
                break;

            segment = *it;
            task.sample = DataStore::temporary(segment, fish, 300u, 15u);
            ++offset;
            ++it;

        } while (task.sample == Sample::Invalid());
        
        if(task.sample != Sample::Invalid()) {
            task.callback = [this](const LearningTask& task)
            {
                for(size_t i=0; i<task.result.size(); ++i) {
                    auto frame = task.sample->_frames.at(i);
                    uint32_t bdx;
                    {
                        Tracker::LockGuard guard("task.callback");
                        auto blob = fish->compressed_blob(frame);
                        if (!blob) {
                            Except("Blob in frame %d not found", frame);
                            continue;
                        }

                        bdx = blob->blob_id();
                    }

                    DataStore::set_label(Frame_t(frame), bdx, DataStore::label(task.result.at(i)));
                }
                    
                Debug("Callback setting probabilities for %lu blobs. Assigning next...", task.result.size());

                {
                    std::lock_guard guard(Work::_mutex);
                    Work::task_queue().push([this]()
                        {
                            this->next();
                        });
                }

                Debug("Callback assigned next.");
                Work::_variable.notify_one();
            };
            
            Work::add_task(std::move(task));
        }
        
        bool done = size_t(offset) == fish->frame_segments().size();
        if(done || size_t(offset) % 100 == 0) {
            auto percent = NetworkApplicationState::percent();
            auto text = "Applying "+Meta::toStr(int(percent * 100))+"%...";
            if(!Work::visible()) {
                if(percent >= 1)
                    GUI::set_status("");
                else
                    GUI::set_status(text);
            } else
                Work::status() = text;
            
            Debug("Fish%d: %f%%", fish->identity().ID(), float(offset) / float(fish->frame_segments().size()) * 100);
        }
    }
    
    static auto& current() {
        static std::unordered_map<Individual*, NetworkApplicationState> _map;
        return _map;
    }
    
    static double percent() {
        Tracker::LockGuard guard("Categorize::percent");
        double percent = 0, N = current().size();
        for (auto &[k, v] : current()) {
            percent += float(v.offset) / float(v.fish->frame_segments().size());
        }
        return N>0 ? percent / N : 0;
    }
};

void start_applying() {
    Work::initialized_apply() = false;
    Work::start_learning(); // make sure the work-horse is started
    
    std::lock_guard guard(Work::_mutex);
    Work::task_queue().push([](){
        Debug("## Initializing APPLY.");
        {
            Tracker::LockGuard guard("Categorize::start_applying");
            NetworkApplicationState::current().clear();
            
            for (auto &[id, ptr] : Tracker::individuals()) {
                auto &obj = NetworkApplicationState::current()[ptr];
                obj.fish = ptr;
            }

            Debug("## Created %lu objects", NetworkApplicationState::current().size());
        }
        
        Work::status() = "Applying "+Meta::toStr(int(NetworkApplicationState::percent() * 100))+"%...";
        
        for(auto & [k, v] : NetworkApplicationState::current()) {
            v.next(); // start first task
            if (Work::terminate)
                break;
        }
        
        Debug("Done adding initial samples");
    });
        
    Work::_variable.notify_all();
}

file::Path output_location() {
    return pv::DataLocation::parse("output", file::Path((std::string)SETTING(filename).value<file::Path>().filename() + "_categories.npz"));
}

void Work::start_learning() {
    if(Work::_learning) {
        return;
    }
    
    Work::_learning = true;
    
    PythonIntegration::async_python_function([]() -> bool{
        Work::status() = "initializing...";
        
        using py = PythonIntegration;
        static const std::string module = "trex_learn_category";
        
        py::import_module(module);
        
        auto reset_variables = [](){
            Debug("Reset python functions and variables...");
            const auto dims = SETTING(recognition_image_size).value<Size2>();
            std::map<std::string, int> keys;
            {
                std::lock_guard guard(DataStore::mutex());
                for(auto & [key, v] : _labels)
                    keys[key->name] = key->id;
            }
            py::set_variable("categories", Meta::toStr(keys), module);
            py::set_variable("width", (int)dims.width, module);
            py::set_variable("height", (int)dims.height, module);
            py::set_variable("output_file", output_location().str(), module);
            py::set_function("set_best_accuracy", [&](float v) {
                Debug("Work::set_best_accuracy(%f);", v);
                Work::set_best_accuracy(v);
            }, module);
            py::set_function("recv_samples", [dims](std::vector<uchar> images, std::vector<std::string> labels) {
                Debug("Received %lu images and %lu labels", images.size(), labels.size());
                
                /*for (size_t i=0; i<labels.size(); ++i) {
                    size_t index = i * size_t(dims.width) * size_t(dims.height);
                    Sample::Make(Image::Make(dims.height, dims.width, 1, images.data() + index), );
                }*/
                
            }, module);
            
            py::run(module, "start");
            
            /*if(!DataStore::composition().empty()) {
                std::vector<Image::Ptr> _images;
                std::vector<std::string> _labels;
 
                {
                    std::lock_guard guard(DataStore::mutex());
                    for(auto it = DataStore::begin(); it != DataStore::end(); ++it) {
                        _images.insert(_images.end(), (*it)->_images.begin(), (*it)->_images.end());
                        _labels.insert(_labels.end(), (*it)->_images.size(), (*it)->_assigned_label->name);
                    }
                }
                
                // re-add images
                py::set_variable("additional", _images, module);
                py::set_variable("additional_labels", _labels, module);
                py::run(module, "add_images");
            }*/
        };
        
        reset_variables();
        
        Work::status() = "";
        
        std::unique_lock guard(Work::_learning_mutex);
        while(Work::_learning) {
            Work::_learning_variable.wait_for(guard, std::chrono::seconds(1));
            
            Debug("Waiting for learning tasks...");
            size_t executed = 0;
            bool clear_probs = false;

            std::vector<std::tuple<LearningTask, size_t>> prediction_tasks;
            std::vector<std::tuple<LearningTask, size_t, size_t>> training_tasks;
            std::vector<Image::Ptr> prediction_images, training_images;
            std::vector<std::string> training_labels;
            
            while(!queue().empty()) {
                auto item = std::move(queue().front());
                queue().pop();

                guard.unlock();
                /*if(py::check_module(module)) {
                    Debug("Module '%S' changed, reset variables...", &module);
                    reset_variables();
                }*/
                
                switch (item.type) {
                    case LearningTask::Type::Load: {
                        py::run(module, "load");
                        //py::run(module, "send_samples");
                        clear_probs = true;
                        if (item.callback)
                            item.callback(item);
                        break;
                    }
                        
                    case LearningTask::Type::Restart:
                        py::run(module, "clear_images");
                        if (item.callback)
                            item.callback(item);
                        break;
                        
                    case LearningTask::Type::Prediction: {
                        auto idx = prediction_images.size();
                        prediction_images.insert(prediction_images.end(), item.sample->_images.begin(), item.sample->_images.end());
                        prediction_tasks.emplace_back(std::move(item), idx);
                        break;
                    }
                        
                    case LearningTask::Type::Training: {
                        auto ldx = training_labels.size();
                        auto idx = training_images.size();

                        training_labels.insert(training_labels.end(), item.sample->_frames.size(), item.sample->_assigned_label->name);
                        training_images.insert(training_images.end(), item.sample->_images.begin(), item.sample->_images.end());
                        training_tasks.emplace_back(std::move(item), idx, ldx);
                        break;
                    }
                        
                    default:
                        break;
                }
                
                guard.lock();
            }

            if (!prediction_tasks.empty()) {
                guard.unlock();

                Work::status() = "prediction...";
                Debug("Predicting %lu samples, %lu collected", prediction_images.size(), prediction_tasks.size());
                py::set_variable("images", prediction_images, module);
                py::set_function("receive", [&](std::vector<float> results)
                {
                    Debug("Receive %lu values", results.size());
                    for (auto& [item, offset] : prediction_tasks) {
                        if (item.type == LearningTask::Type::Prediction) {
                            item.result.insert(item.result.end(), results.begin() + offset, results.begin() + offset + item.sample->_images.size());
                            if (item.callback)
                                item.callback(item);
                        }
                    }

                }, module);

                py::run(module, "predict");
                Work::status() = "";

                guard.lock();
            }

            if (!training_images.empty()) {
                Debug("Training on %lu additional samples", training_images.size());
                // train for a couple epochs
                py::set_variable("epochs", int(10));
                py::set_variable("additional", training_images, module);
                py::set_variable("additional_labels", training_labels, module);
                py::run(module, "add_images");
                clear_probs = true;

                guard.unlock();
                for (auto& [item, _, __] : training_tasks) {
                    if (item.type == LearningTask::Type::Training) {
                        if (item.callback)
                            item.callback(item);
                    }
                }

                Work::status() = "training...";
                py::run(module, "post_queue");
                Work::status() = "";
                guard.lock();
            }
            
            if(clear_probs) {
                clear_probs = false;
                Debug("# Clearing calculated probabilities...");
                guard.unlock();
                {
                    std::lock_guard g(Work::_recv_mutex);
                    for(auto &row : rows) {
                        for(auto &cell : row._cells) {
                            if(cell._sample) {
                                cell._sample->_probabilities.clear();
                                cell._sample->_requested = false;
                            }
                        }
                    }
                }
                guard.lock();
            }
        }
        
        return true;
        
    }, PythonIntegration::Flag::DEFAULT, false);
}

void Work::loop() {
    std::unique_lock guard(_mutex);
    while (!terminate) {
        // check for tasks
        while(!Work::task_queue().empty()) {
            auto task = std::move(Work::task_queue().front());
            Work::task_queue().pop();
            
            guard.unlock();
            task();
            guard.lock();
            
            if (terminate)
                break;
        }
        
        Sample::Ptr sample;
        while(_generated_samples.size() < requested_samples() && !terminate) {
            guard.unlock();
            {
                Tracker::LockGuard g("get_random::loop");
                sample = DataStore::get_random();
                if (sample && sample->_images.size() < 50) {
                    sample = Sample::Invalid();
                }
            }
            guard.lock();
            
            if(sample != Sample::Invalid() && !sample->_assigned_label) {
                _generated_samples.push(sample);
                _recv_variable.notify_one();
            }
        }

        if (_generated_samples.size() < requested_samples() && !terminate)
            _variable.notify_one();
        
        if(terminate)
            break;
        
        _variable.wait_for(guard, std::chrono::seconds(10));
    }
}

void DataStore::clear() {
    std::lock_guard guard(mutex());
    _samples.clear();
    _labels.clear();
    _used_indexes.clear();
}

Sample::Ptr DataStore::temporary(const std::shared_ptr<Individual::SegmentInformation>& segment,
                                 Individual* fish,
                                 const size_t sample_rate,
                                 const size_t min_samples)
{
    {
        std::lock_guard guard(mutex());
        auto fit = _used_indexes.find(segment.get());
        if(fit != _used_indexes.end()) {
            return _samples.at(fit->second); // already sampled
        }
    }
    
    std::set<long_t> stuff_indexes;
    
    std::vector<Image::Ptr> images;
    std::vector<long_t> indexes;
    
    size_t step = max(1u, segment->basic_index.size() / sample_rate);
    for (size_t i=0; i<segment->basic_index.size(); i += step) {
        stuff_indexes.insert(segment->basic_index.at(i));
    }
    
    if(stuff_indexes.size() < min_samples)
        return Sample::Invalid();

    size_t non = 0, cont = 0;
    for(auto index : stuff_indexes) {
        Midline::Ptr midline;
        std::shared_ptr<Individual::BasicStuff> basic;

        const auto normalize = SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>();
        const auto scale = FAST_SETTINGS(recognition_image_scale);
        const auto dims = SETTING(recognition_image_size).value<Size2>();

        {
            Tracker::LockGuard guard("Categorize::sample");
            basic = fish->basic_stuff().at(index);
            auto posture = fish->posture_stuff(basic->frame);
            midline = posture ? fish->calculate_midline_for(basic, posture) : nullptr;
        }

        PPFrame video_frame;
        auto active =
            basic->frame == Tracker::start_frame()
                ? std::unordered_set<Individual*>()
                : Tracker::active_individuals(basic->frame-1);
        
        auto &video_file = *GUI::instance()->video_source();
        video_file.read_frame(video_frame.frame(), sign_cast<uint64_t>(basic->frame));
        
        size_t idx;
        {
            std::lock_guard guard(mutex());
            idx = _samples.size();
        }

        Tracker::LockGuard guard("Categorize::sample");
        Tracker::instance()->preprocess_frame(video_frame, active, NULL);

        std::map<uint32_t, pv::BlobPtr> blob_to_id;
        for (auto b : video_frame.blobs)
            blob_to_id[b->blob_id()] = b;
        
        auto blob = Tracker::find_blob_noisy(blob_to_id, basic->blob.blob_id(), basic->blob.parent_id, basic->blob.calculate_bounds(), basic->frame);
        //auto it = fish->iterator_for(basic->frame);
        if (blob) { //&& it != fish->frame_segments().end()) {
            auto custom_len = Tracker::recognition()->local_midline_length(fish, segment->range);

            Recognition::ImageData image_data(
                Recognition::ImageData::Blob{
                    blob->num_pixels(),
                    blob->blob_id(),
                    -1,
                    blob->parent_id(),
                    blob->bounds()
                },
                basic->frame, FrameRange(), fish, fish->identity().ID(),
                midline ? midline->transform(normalize) : gui::Transform()
            );

            image_data.filters = std::make_shared<TrainingFilterConstraints>(custom_len);

            auto image = Recognition::calculate_diff_image_with_settings(normalize, blob, image_data, dims);
            if (image) {
                images.emplace_back(std::move(image));
                indexes.emplace_back(basic->frame);
            }
        }
        else
            ++non;
    }
    
    Debug("Of %lu frames, %lu couldnt find a blob and %lu continued (min_samples %lu).", stuff_indexes.size(), non, cont, min_samples);
    if(images.size() >= min_samples) {
        return Sample::Make(std::move(indexes), std::move(images));
    }
    
    return Sample::Invalid();
}

const Sample::Ptr& DataStore::sample(
        const std::shared_ptr<Individual::SegmentInformation>& segment,
        Individual* fish,
        const size_t max_samples,
        const size_t min_samples)
{
    {
        std::lock_guard guard(mutex());
        auto fit = _used_indexes.find(segment.get());
        if(fit != _used_indexes.end()) {
            return _samples.at(fit->second); // already sampled
        }
    }
    
    auto s = temporary(segment, fish, max_samples, min_samples);
    if(s == Sample::Invalid())
        return Sample::Invalid();
    
    std::lock_guard guard(mutex());
    _used_indexes[segment.get()] = _samples.size();
    _samples.emplace_back(s);
    return _samples.back();
}

std::string DataStore::Composition::toStr() const {
    return (Work::best_accuracy() > 0 ? "Accuracy: "+ Meta::toStr(int(Work::best_accuracy() * 100)) + "% " : "")
        + (!_numbers.empty() ? "Collected: " +Meta::toStr(_numbers) : "No samples collected yet.")
        + (Work::status().empty() ? "" : " "+Work::status());
}

static Tooltip tooltip(nullptr, 200);
static Layout::Ptr stext = nullptr;
static Entangled* selected = nullptr;
static Layout::Ptr apply(Layout::Make<Button>("Apply", Bounds(0, 0, 100, 33)));
static Layout::Ptr load(Layout::Make<Button>("Load", Bounds(0, 0, 100, 33)));
static Layout::Ptr close(Layout::Make<Button>("Hide", Bounds(0, 0, 100, 33)));
static Layout::Ptr restart(Layout::Make<Button>("Restart", Bounds(0, 0, 100, 33)));
static Layout::Ptr buttons(Layout::Make<HorizontalLayout>(std::vector<Layout::Ptr>{apply, restart, load, close}));

void initialize(DrawStructure& base) {
    static double R = 0, elap = 0;
    static Timer timer;
    //R += RADIANS(100) * timer.elapsed();
    elap += timer.elapsed();
    
    static bool initialized = false;
    if(!initialized && Work::num_ready() >= 1) {
        //PythonIntegration::ensure_started();
        //PythonIntegration::async_python_function([]()->bool{return true;});
        //Work::start_learning();
        
        elap = 0;
        initialized = true;
        
        layout.set_policy(gui::VerticalLayout::CENTER);
        layout.set_origin(Vec2(0.5));
        layout.set_pos(Size2(base.width(), base.height()) * 0.5);
        
        stext = Layout::Make<StaticText>(
          "<h2>Categorizing types of individuals</h2>"
          "Below, an assortment of randomly chosen clips is shown. They are compiled automatically to (hopefully) only contain samples belonging to the same category. Choose clips that best represent the categories you have defined before (<str>"+Meta::toStr(DataStore::label_names())+"</str>) and assign them by clicking the respective button. But be careful - with them being automatically collected, some of the clips may contain images from multiple categories. It is recommended to <b>Skip</b> these clips, lest risking to confuse the poor network. Regularly, when enough new samples have been collected (and for all categories), they are sent to said network for a training step. Each training step, depending on clip quality, should improve the prediction accuracy (see below).",
          Vec2(),
          Vec2(base.width() * 0.5 * base.scale().x, -1), Font(0.7)
        );
        
        layout.add_child(stext);
        //layout.add_child(Layout::Make<Text>("Categorizing types of individuals", Vec2(), Cyan, Font(0.75, Style::Bold)));
        
        apply->on_click([](auto) {
            Work::set_state(Work::State::APPLY);
        });
        close->on_click([](auto) {
            Work::set_state(Work::State::NONE);
        });
        load->on_click([](auto) {
            Work::set_state(Work::State::LOAD);
        });
        restart->on_click([](auto) {
            Work::set_state(Work::State::SELECTION);
        });
        
        apply.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Green.exposure(0.15)));
        close.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Red.exposure(0.2)));
        load.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Yellow.exposure(0.2)));
        
        tooltip.set_scale(base.scale().reciprocal());
        tooltip.text().set_default_font(Font(0.5));
        
        for (auto& row : rows) {
            /**
             * If the row is empty, that means that the whole grid has not been initialized yet.
             * Create images and put them into a per_row^2 grid, and add them to the layout.
             */
            if(row.empty()) {
                row.init(per_row);
                layout.add_child(Layout::Ptr(row.layout));
            }
            
            /**
             * After ensuring we do have rows, fill them with new samples:
             */
            for(size_t i=0; i<row.length(); ++i) {
                auto sample = Work::retrieve();
                row.update(i, sample);
            }
        }
        
        if(!layout.empty() && layout.children().back() != buttons.get()) {
            desc_text.to<StaticText>()->set_default_font(Font(0.6));
            desc_text.to<StaticText>()->set_max_size(stext.to<StaticText>()->max_size());
            
            layout.add_child(desc_text);
            layout.add_child(buttons);
        }
        
        layout.auto_size(Margin{0,0});
        layout.set_z_index(1);
        Work::_variable.notify_one();
    }
    
    timer.reset();
}

Cell::Cell() :
    _button_layout(std::make_shared<HorizontalLayout>()),
    _selected(false),
    _image(std::make_shared<ExternalImage>(Image::Make(50,50,1))),
    _text(std::make_shared<StaticText>("", Vec2(), Vec2(-1), Font(0.5))),
    _block(std::make_shared<Entangled>([this](Entangled& e){
        /**
         * This is the block that contains all display-elements of a Cell.
         * 1. A sample image animation
         * 2. TODO: Buttons to assign classes
         * 3. Text with current playback status
         */
        e.advance_wrap(*_image);
        e.advance_wrap(*_button_layout);
        e.advance_wrap(*_text);
        
        auto labels = DataStore::label_names();
        for(auto &c : labels) {
            auto b = Layout::Make<Button>(c, Bounds(Size2(Base::default_text_bounds(c, nullptr, Font(0.75)).width + 10, 33)));
            
            b->on_click([this, c, ptr = &e](auto){
                if(_sample && _row) {
                    try {
                        _sample->set_label(DataStore::label(c.c_str()));
                        {
                            std::lock_guard guard(DataStore::mutex());
                            _labels[_sample->_assigned_label].push_back(_sample);
                        }
                        
                        Work::add_training_sample(_sample);
                        
                    } catch(...) {
                        
                    }
                    
                    _row->update(_index, Work::retrieve());
                }
            });
            
            add(b);
        }
        
        auto b = Layout::Make<Button>("Skip", Bounds(Vec2(), Size2(50,33)));
        b->on_click([this](Event e) {
            if(_row) {
                if(_sample) {
                    _sample->set_label(NULL);
                }
                
                _row->update(_index, Work::retrieve());
            }
        });
        add(b);
        
        _button_layout->auto_size(Margin{0, 0});
    }))
{
    _image->set_clickable(true);
    _block->set_origin(Vec2(0.5));
        
    /**
     * Handle clicks on cells
     */
    _image->on_click([this](Event e) {
        static Cell* _selected = nullptr;
        
        if(e.mbutton.button == 0 && _image.get() == _image->parent()->stage()->hovered_object()) {
            if(_sample) {
                if(_selected == this) {
                    this->set_selected(false);
                    _selected = nullptr;
                    
                } else {
                    if(_selected) {
                        _selected->set_selected(false);
                        _selected = this;
                    }
                    
                    _selected = this;
                    this->set_selected(true);
                }
            }
        }
    });
    
    _block->auto_size(Margin{0, 0});
    _text->set_origin(Vec2(0, 1));
    _text->set_pos(Vec2(5, _block->height() - 5));
}

Cell::~Cell() {
    _button_layout = nullptr;
    _block = nullptr;
    _image = nullptr;
}

Work::State& Work::state() {
    static State _state = Work::State::NONE;
    return _state;
}

void Work::set_state(State state) {
    switch (state) {
        case State::LOAD: {
            Work::start_learning();
            
            LearningTask task;
            task.type = LearningTask::Type::Load;
            Work::add_task(std::move(task));
            Work::_variable.notify_one();
            break;
        }
        case State::NONE:
            hide();
            {
                std::lock_guard g(Work::_recv_mutex);
                for(auto &row : rows) {
                    for(auto &cell : row._cells) {
                        if(cell._sample) {
                            cell._sample->_probabilities.clear();
                            cell._sample->_requested = false;
                        }
                        cell.set_sample(nullptr);
                    }
                }
            }
            break;
            
        case State::SELECTION: {
            if(Work::state() == State::SELECTION) {
                // restart
                LearningTask task;
                task.type = LearningTask::Type::Restart;
                task.callback = [](const LearningTask& task) {
                    DataStore::clear();
                };
                Work::add_task(std::move(task));
                Work::start_learning();
                
            } else {
                Work::status() = "initializing...";
                Work::requested_samples() = per_row * 3;
                Work::_variable.notify_one();
                Work::visible() = true;
                PythonIntegration::ensure_started();
                Work::start_learning();
            }
            
            break;
        }
            
        case State::APPLY: {
            //assert(Work::state() == State::SELECTION);
            hide();
            start_applying();
            break;
        }
            
        default:
            break;
    }
    
    Work::state() = state;
}

void draw(gui::DrawStructure& base) {
    if(!Work::visible())
        return;
    
    {
        std::lock_guard guard(DataStore::mutex());
        if(_labels.empty()) {
            _labels.insert({Label::Make("W"), {}});
            _labels.insert({Label::Make("S"), {}});
            //DataStore::_labels.insert({Label::Make("X"), {}});
        }
        
        if(_labels.empty()) {
            static bool asked = false;
            if(!asked) {
                asked = true;
                
                using namespace gui;
                static Textfield textfield("W,S", Bounds(Size2(base.width() * base.scale().x * 0.4,33)));
                
                auto d = base.dialog([](Dialog::Result r){
                    if(r == Dialog::OKAY) {
                        for(auto text : utils::split(textfield.text(), ',')) {
                            text = utils::trim(text);
                            if(!text.empty())
                                DataStore::label(text.c_str()); // create labels
                        }
                    }
                    
                }, "Please enter the categories (comma-separated), e.g.:\n<i>Worker,Soldier,Trash</i>", "Categorize", "Okay", "Cancel");
                
                d->set_custom_element(Layout::Make<Entangled>([](Entangled& e){
                    e.advance_wrap(textfield);
                }));
            }
            return;
        }
    }
    
    using namespace gui;
    static Rect rect(Bounds(0, 0, 0, 0), Black.alpha(125));
    
    auto window = (GUI::instance() && GUI::instance()->base() ? (GUI::instance()->base()->window_dimensions().div(base.scale())) : Size2(base.width(), base.height())) * gui::interface_scale();
    auto center = window * 0.5;
    layout.set_pos(center);
    
    rect.set_z_index(1);
    rect.set_size(window);
    
    base.wrap_object(rect);
    
    if(!Work::thread) {
        Work::thread = std::make_unique<std::thread>(Work::work);
    }
    
    initialize(base);
    
    layout.auto_size(Margin{0,0});
    base.wrap_object(layout);
    
    static Timer timer;
    
    float max_w = 0;
    for(auto &row : rows) {
        for(auto &cell: row._cells) {
            if(!cell._sample)
                row.update(cell._index, Work::retrieve());
            
            cell._block->auto_size(Margin{0,0});
            max_w = max(max_w, cell._block->width());
        }
        row.update(base, timer.elapsed());
    }
    
    max_w = per_row * (max_w + 10);
    
    if(buttons) buttons->set_scale(base.scale().reciprocal());
    if(desc_text) desc_text->set_scale(base.scale().reciprocal());
    
    if(stext) {
        stext->set_scale(base.scale().reciprocal());
        stext.to<StaticText>()->set_max_size(Size2(max_w * 1.5 * base.scale().x, -1));
        if(desc_text)
            desc_text.to<StaticText>()->set_max_size(stext.to<StaticText>()->max_size());
    }
    
    timer.reset();
    
    auto txt = settings::htmlify(Meta::toStr(DataStore::composition()));
    if(Work::best_accuracy() < Work::good_enough()) {
        txt = "<i>Predictions for all visible tiles will be displayed as soon as the network becomes confident enough.</i>\n"+txt;
    }
    desc_text.to<StaticText>()->set_txt(txt);
}

}
}
