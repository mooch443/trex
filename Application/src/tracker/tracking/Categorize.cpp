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
        {
            std::lock_guard guard(_learning_mutex);
            queue().push(std::move(task));
        }
        
        Work::_learning_variable.notify_one();
    }
};

Sample::Sample(Idx_t fish, const decltype(_segment)& segment, std::vector<long_t>&& frames, const std::vector<Image::Ptr>& images)
    :   _fish(fish),
        _segment(segment),
        _frames(std::move(frames)),
        _images(images)
{
    assert(!_images.empty());
}

std::set<std::string> DataStore::label_names() {
    std::set<std::string> _names;
    for(auto &[l, s] : _labels)
        _names.insert(l->name);
    return _names;
}

Label::Ptr DataStore::label(const char* name) {
    for(auto &[l, s] : _labels) {
        if(l->name == name)
            return l;
    }
    
    auto l = Label::Make(std::string(name));
    _labels.insert({l, {}});
    return l;
}

Label::Ptr DataStore::label(int ID) {
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
        fish = Tracker::instance()->individuals().at(fid);
        auto &basic_stuff = fish->basic_stuff();
        if(basic_stuff.empty())
            return Sample::Invalid();
        
        std::uniform_int_distribution<remove_cvref<decltype(fish->frame_segments())>::type::difference_type> sample_dist(0, fish->frame_segments().size()-1);
        auto it = fish->frame_segments().begin();
        std::advance(it, sample_dist(mt));
        segment = *it;
    }
    
    if(!segment)
        return Sample::Invalid();
    
    return sample(segment, fish);
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
    Composition c;
    for (auto &[key, samples] : _labels) {
        for(auto &s : samples)
            c._numbers[key->name] += s->_images.size();
    }
    return c;
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
        Work::requested_samples() = per_row * per_row;
        Work::_variable.notify_one();
        Work::visible() = true;
        
        PythonIntegration::ensure_started();
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
    
    if(!sample)
        std::fill(_image->source()->data(),
                  _image->source()->data() + _image->source()->size(),
                  0);
    else {
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

void Work::start_learning() {
    if(Work::_learning) {
        return;
    }
    
    Work::_learning = true;
    
    PythonIntegration::async_python_function([]() -> bool{
        using py = PythonIntegration;
        static const std::string module = "trex_learn_category";
        py::import_module(module);
        const auto dims = SETTING(recognition_image_size).value<Size2>();
        std::map<std::string, int> keys;
        for(auto & [key, v] : _labels)
            keys[key->name] = key->id;
        py::set_variable("categories", Meta::toStr(keys), module);
        py::set_variable("width", (int)dims.width, module);
        py::set_variable("height", (int)dims.height, module);
        py::run(module, "start");
        
        std::unique_lock guard(Work::_learning_mutex);
        while(Work::_learning) {
            Work::_learning_variable.wait_for(guard, std::chrono::seconds(1));
            
            //Debug("Waiting for learning tasks...");
            size_t executed = 0;
            
            while(!queue().empty()) {
                auto item = std::move(queue().front());
                queue().pop();
                
                guard.unlock();
                switch (item.type) {
                    case LearningTask::Type::Prediction:
                        Work::status() = "prediction...";
                        Debug("Predicting %lu samples", item.sample->_frames.size());
                        py::set_variable("images", item.sample->_images, module);
                        py::set_function("receive", [&](std::vector<float> results)
                        {
                            Debug("Receive %lu values", results.size());
                            item.result = std::move(results);
                        }, module);
                        py::run(module, "predict");
                        Work::status() = "";
                        
                        break;
                        
                    case LearningTask::Type::Training: {
                        Debug("Training on %lu additional samples", item.sample->_frames.size());
                        // train for a couple epochs
                        py::set_variable("epochs", int(10));
                        std::vector<std::string> labels;
                        for(size_t i=0; i<item.sample->_frames.size(); ++i)
                            labels.push_back(item.sample->_assigned_label->name);
                        py::set_variable("additional", item.sample->_images, module);
                        py::set_variable("additional_labels", labels, module);
                        py::set_function("set_best_accuracy", [&](float v) {
                            Debug("%f", v);
                            Work::set_best_accuracy(v);
                        }, module);
                        py::run(module, "add_images");
                        
                        ++executed;
                        break;
                    }
                        
                    default:
                        break;
                }
                
                if(item.callback)
                    item.callback(item);
                
                guard.lock();
            }
            
            if(executed) {
                guard.unlock();
                Work::status() = "training...";
                py::run(module, "post_queue");
                Work::status() = "";
                
                Debug("# Clearing calculated probabilities...");
                std::lock_guard g(Work::_recv_mutex);
                for(auto &row : rows) {
                    for(auto &cell : row._cells) {
                        if(cell._sample) {
                            cell._sample->_probabilities.clear();
                            cell._sample->_requested = false;
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
        while(_generated_samples.size() < requested_samples() && !terminate) {
            guard.unlock();
            auto sample = DataStore::get_random();
            guard.lock();
            
            if(sample != Sample::Invalid() && !sample->_assigned_label) {
                _generated_samples.push(sample);
                _recv_variable.notify_one();
            }
        }
        
        if(terminate)
            break;
        
        _variable.wait_for(guard, std::chrono::seconds(10));
    }
}

void DataStore::predict(
    const std::shared_ptr<Individual::SegmentInformation>& segment,
    Individual* fish,
    std::function<void(Sample::Ptr)>&& callback)
{
    static std::unordered_map<const Sample*, Sample::Ptr> map; // holds references until not required
    auto s = temporary(segment, fish);
    
}

Sample::Ptr DataStore::temporary(const std::shared_ptr<Individual::SegmentInformation>& segment,
                                 Individual* fish)
{
    auto fit = _used_indexes.find(segment.get());
    if(fit != _used_indexes.end()) {
        return _samples.at(fit->second); // already sampled
    }
    
    std::set<long_t> frames;
    
    std::vector<Image::Ptr> images;
    std::vector<long_t> indexes;
    
    size_t step = max(1u, segment->basic_index.size() / 150u);
    for (size_t i=0; i<segment->basic_index.size(); i += step) {
        frames.insert(segment->basic_index.at(i));
    }
    
    if(frames.size() < 150)
        return Sample::Invalid();
    
    auto str = Meta::toStr(frames);
    Debug("Adding %lu frames (%S)", frames.size(), &str);
    
    for(auto frame : frames) {
        PPFrame video_frame;
        auto active =
            frame == Tracker::start_frame()
                ? std::unordered_set<Individual*>()
                : Tracker::active_individuals(frame-1);
        
        auto &video_file = *GUI::instance()->video_source();
        video_file.read_frame(video_frame.frame(), sign_cast<uint64_t>(frame));

        Tracker::LockGuard guard("Categorize::sample");
        Tracker::instance()->preprocess_frame(video_frame, active, NULL);

        std::map<uint32_t, pv::BlobPtr> blob_to_id;
        for (auto b : video_frame.blobs)
            blob_to_id[b->blob_id()] = b;
        
        const auto scale = FAST_SETTINGS(recognition_image_scale);
        const auto dims = SETTING(recognition_image_size).value<Size2>();
        
        auto idx = _samples.size();
        auto basic = fish->basic_stuff(frame);
        auto posture = fish->posture_stuff(frame);
        
        if(!basic)
            continue;

        auto normalize = SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>();
        auto midline = posture ? fish->calculate_midline_for(basic, posture) : nullptr;
        //auto blob = basic->blob.unpack();
        
        auto blob = Tracker::find_blob_noisy(blob_to_id, basic->blob.blob_id(), basic->blob.parent_id, basic->blob.calculate_bounds(), basic->frame);
        
        auto it = fish->iterator_for(basic->frame);
        if(blob && it != fish->frame_segments().end()) {
            auto custom_len = Tracker::recognition()->local_midline_length(fish, (*it)->range);
            
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
            if(image) {
                images.emplace_back(std::move(image));
                indexes.emplace_back(frame);
            }
        }
    }
    
    Debug("Added %lu frames", frames.size());
    
    if(images.size() > 5) {
        return Sample::Make(fish->identity().ID(), segment, std::move(indexes), std::move(images));
    }
    
    return Sample::Invalid();
}

const Sample::Ptr& DataStore::sample(
        const std::shared_ptr<Individual::SegmentInformation>& segment,
        Individual* fish)
{
    auto fit = _used_indexes.find(segment.get());
    if(fit != _used_indexes.end()) {
        return _samples.at(fit->second); // already sampled
    }
    
    auto s = temporary(segment, fish);
    if(s == Sample::Invalid())
        return Sample::Invalid();
    
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
static Layout::Ptr close(Layout::Make<Button>("Hide", Bounds(0, 0, 100, 33)));
static Layout::Ptr buttons(Layout::Make<HorizontalLayout>(std::vector<Layout::Ptr>{apply, close}));

void initialize(DrawStructure& base) {
    static double R = 0, elap = 0;
    static Timer timer;
    //R += RADIANS(100) * timer.elapsed();
    elap += timer.elapsed();
    
    static bool initialized = false;
    if(!initialized && Work::num_ready() >= per_row * rows.size()) {
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
        
        apply->on_click([](auto) { hide(); });
        close->on_click([](auto) { hide(); });
        
        apply.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Green.exposure(0.15)));
        close.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Red.exposure(0.2)));
        
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
        
        for(auto &[c, s] : _labels) {
            auto b = Layout::Make<Button>(c->name, Bounds(Size2(Base::default_text_bounds(c->name, nullptr, Font(0.75)).width + 10, 33)));
            
            b->on_click([this, c = c->name.c_str(), ptr = &e](auto){
                if(_sample && _row) {
                    Debug("%s: %d (cell: %d,%d)", c, _sample->_segment->start(), _row->index, _index);
                    
                    try {
                        _sample->set_label(DataStore::label(c));
                        _labels[_sample->_assigned_label].push_back(_sample);
                        
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
     * Update tooltip
     * and Cell-background
     */
    _image->on_hover([this](Event e) {
        if(e.hover.hovered) {
            _block->set_background(Transparent, White.alpha(225));
            
            if(_sample) {
                tooltip.set_text("<h2>"+FAST_SETTINGS(individual_prefix)+Meta::toStr(_sample->_fish)+"</h2>"+Meta::toStr(_sample->_segment->range));
                tooltip.set_other(_block.get());
            }
            
        } else {
            _block->set_background(Transparent, Transparent);
        }
    });
        
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

void draw(gui::DrawStructure& base) {
    show();
    
    if(!Work::visible())
        return;
    
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
    base.wrap_object(tooltip);
    
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