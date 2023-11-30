#include "CategorizeInterface.h"
#include <tracking/Categorize.h>
#include <tracking/Tracker.h>
#include <misc/default_settings.h>
#include <gui/IMGUIBase.h>

namespace track {
namespace Categorize {

struct Row;

struct Cell {
private:
    std::vector<Layout::Ptr> _buttons;
    GETTER(std::shared_ptr<HorizontalLayout>, button_layout)
    GETTER_SETTER_I(bool, selected, false)
    
public:
    Row *_row = nullptr;
    size_t _index = 0;
    Sample::Ptr _sample;
    double _animation_time = 0;
    size_t _animation_index = 0;
    int _max_id = -1;
    
    // gui elements
    std::shared_ptr<ExternalImage> _image;
    std::shared_ptr<StaticText> _text;
    std::shared_ptr<Rect> _cat_border;
    std::shared_ptr<Entangled> _block;
    
public:
    Cell();
    ~Cell();
    
    void add(const Layout::Ptr& b);
    
    void set_sample(const Sample::Ptr& sample);
    void update_scale();
    
    //static void receive_prediction_results(const LearningTask& task);
    
    void update(float s);
    
    const Bounds& bounds();
};
    
struct Row {
    int index;
    
    std::vector<Cell> _cells;
    std::shared_ptr<HorizontalLayout> layout;
    
    Row(int i);
    
    void init(size_t additions);
    void clear();
    
    Cell& cell(size_t i);
    
    size_t length() const;
    
    void update(DrawStructure& base, double dt);
    
    void update(size_t cell_index, const Sample::Ptr& sample);
    
    bool empty() const;
    
    static std::array<Row, 2>& rows() {
        static std::array<Row, 2> rows{ Row(0), Row(1) };
        return rows;
    }
};

Sample::Ptr retrieve() {
    Sample::Ptr sample = Work::front_sample();
        
    if(sample != Sample::Invalid()) {
        /**
         * Search current rows and cells to see whether the sample is already assigned
         * to any of the cells.
         */
        LOGGED_LOCK_TYPE<std::recursive_mutex> gui_guard;
        if(Interface::get().layout.stage()) {
            gui_guard = GUI_LOCK(Interface::get().layout.stage()->lock());
        }
        for(auto &row : Row::rows()) {
            for(auto &c : row._cells) {
                if(c._sample == sample) {
                    sample = Sample::Invalid();
                    break;
                }
            }
        }
    }
    
    return sample;
}

Cell::Cell() :
    _button_layout(std::make_shared<HorizontalLayout>()),
    _selected(false),
    _image(std::make_shared<ExternalImage>(Image::Make(50,50,1))),
    _text(std::make_shared<StaticText>(Font(0.5))),
    _cat_border(std::make_shared<Rect>(Box(50,50))),
    _block(std::make_shared<Entangled>([this](Entangled& e){
        /**
         * This is the block that contains all display-elements of a Cell.
         * 1. A sample image animation
         * 2. TODO: Buttons to assign classes
         * 3. Text with current playback status
         */
        e.advance_wrap(*_cat_border);
        e.advance_wrap(*_image);
        e.advance_wrap(*_button_layout);
        e.advance_wrap(*_text);
        
        auto labels = DataStore::label_names();
        for(auto &c : labels) {
            auto b = Layout::Make<Button>(Str(c), Box(Size2(Base::default_text_bounds(c, nullptr, Font(0.75)).width + 10, 33)));
            
            b->on_click([this, c, ptr = &e](auto){
                if(_sample && _row) {
                    try {
                        _sample->set_label(DataStore::label(c.c_str()));
                        Work::add_training_sample(_sample);
                        
                    } catch(...) {
                        
                    }
                    
                    _row->update(_index, retrieve());
                }
            });
            
            add(b);
        }
        
        auto b = Layout::Make<Button>(Str("Skip"), Box(Vec2(), Size2(50,33)));
        b->on_click([this](Event e) {
            if(_row) {
                if(_sample) {
                    _sample->set_label(NULL);
                }
                
                _row->update(_index, retrieve());
            }
        });
        add(b);
        
        _button_layout->auto_size();
    }))
{
    _image->set_clickable(true);
    _block->set_origin(Vec2(0.5));
        
    /**
     * Handle clicks on cells
     */
    static Cell* _selected = nullptr;

    _image->on_click([this](Event e) {
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

void Cell::add(const Layout::Ptr& b) {
    _buttons.emplace_back(b);
    _button_layout->add_child(b);
}

void receive_prediction_results(const LearningTask& task) {
    std::lock_guard guard(Work::recv_mutex());
    auto cats = FAST_SETTING(categories_ordered);
    task.sample->_probabilities.resize(cats.size());
    std::fill(task.sample->_probabilities.begin(), task.sample->_probabilities.end(), 0);
    
    for(size_t i=0; i<task.result.size(); ++i) {
        assert(task.result.at(i) == DataStore::label(task.result.at(i))->id);
        task.sample->_probabilities[task.result.at(i)] += float(1);
    }
    
#ifndef NDEBUG
    auto str0 = Meta::toStr(task.sample->_probabilities);
#endif
    float S = narrow_cast<float>(task.result.size());
    for (size_t i=0; i<cats.size(); ++i) {
        task.sample->_probabilities[i] /= S;
#ifndef NDEBUG
        if(task.sample->_probabilities[i] > 1) {
            FormatWarning("Probability > 1? ", task.sample->_probabilities[i]," for k '",cats[i].c_str(),"'");
        }
#endif
    }
    
#ifndef NDEBUG
    auto str1 = Meta::toStr(task.sample->_probabilities);
    print(task.result.size(),": ",str0.c_str()," -> ",str1.c_str());
#endif
}

void Cell::update(float s) {
    for(auto &c : _buttons) {
        c.to<Button>()->set_text_clr(White.alpha(235 * s));
        c.to<Button>()->set_line_clr(Black.alpha(200 * s));
        c.to<Button>()->set_fill_clr(DarkCyan.exposure(s).alpha(150 * s));
    }
    
    if(_sample) {
        auto text = "<nr>"+Meta::toStr(_animation_index+1)+"</nr>/<nr>"+Meta::toStr(_sample->_images.size())+"</nr>";
        
        std::lock_guard guard(Work::recv_mutex());
        if(!_sample->_probabilities.empty()) {
            std::string summary;
            double max_p = -1;
            size_t index = 0;
            
            for(size_t i=0; i<_sample->_probabilities.size(); ++i)
                if(_sample->_probabilities[i] > max_p) {
                    index = i;
                    max_p = _sample->_probabilities[i];
                }
            
            for(size_t i=0; i<_sample->_probabilities.size(); ++i) {
                if(!summary.empty())
                    summary += " ";
                summary += std::string(index == i ? "<b><str>" : "<ref>")
                    +  DataStore::label(i)->name
                    + (index == i ? "</str></b>" : "</ref>")
                    + ":<nr>" + dec<2>(_sample->_probabilities[i] * 100).toStr()+"</nr>%";
            }
            
            text = summary + "\n" + text;
            
        } else if(!_sample->_requested) {
            if(Work::best_accuracy() >= Work::good_enough()) {
                _sample->_requested = true;
                
                LearningTask task;
                task.sample = _sample;
                task.type = LearningTask::Type::Prediction;
                task.callback = receive_prediction_results;
                
                Work::add_task(std::move(task));
            }
            
        } else
            text += " <key>(pred.)</key>";
        
        _text->set_txt(text);
    }
    
    _image->set_color(White.alpha(200 + 55 * s));
    _text->set_alpha(Alpha{0.25 + s * 0.75});
    
    auto rscale = _button_layout->parent() ? _button_layout->parent()->stage()->scale().reciprocal().mul(_block->scale().reciprocal()) : Vec2(1);
    _text->set_scale(rscale);
    _button_layout->set_scale(rscale);
    
    if(_sample && _max_id == -1) {
        std::lock_guard g(Work::recv_mutex());
        float max_p = 0;
        if(!_sample->_probabilities.empty()) {
            for(size_t j=0; j<_sample->_probabilities.size(); ++j) {
                auto p = _sample->_probabilities[j];
                if(p > max_p) {
                    max_p = p;
                    _max_id = j;
                }
            }
        }
    }
    
    Color color = DarkGray;
    if(_max_id != -1)
        color = ColorWheel(_max_id).next();
    _cat_border->set_fillclr(color.alpha(255 * (0.75 * s + 0.25)));
    
    auto bds = _image->bounds();
    _cat_border->set_scale(_image->scale());
    _cat_border->set_pos(bds.pos() - 5);
    _cat_border->set_size(bds.size() + 10 / _cat_border->scale().x);
    
    //_text->set_base_text_color(White.alpha(100 + 155 * s));
    _button_layout->auto_size();
    _text->set_pos(Vec2(10, _block->height() - 15));
}

const Bounds& Cell::bounds() {
    return _block->global_bounds();
}

void Cell::update_scale() {
    double s = 1 / double(_row->_cells.size());
    auto base = button_layout()->stage();

    if (base && _image->width() > 0) {
        Size2 bsize(base->width(), base->height());
        //print("DPI = ", ((IMGUIBase*)GUI::instance()->best_base())->dpi_scale(), " bsize = ", bsize);
        if(Interface::get()._window)
            bsize = bsize / Interface::get()._window->dpi_scale();
        bsize = bsize.div(base->scale());

        if (base->width() * s < base->height() / 4.0)
            _image->set_scale(Vec2(bsize.width * s / _image->width()));
        else
            _image->set_scale(Vec2(bsize.height * (1.0/4.0) / _image->height()));
        
    }
}

void Cell::set_sample(const Sample::Ptr &sample) {
    if(sample != _sample)
        _max_id = -1;
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

    update_scale();
}

Row::Row(int i)
    : index(i), layout(std::make_shared<HorizontalLayout>())
{ }

void Row::init(size_t additions) {
    _cells.clear();
    _cells.resize(additions);
    
    size_t i=0;
    for(auto &cell : _cells) {
        layout->add_child(Layout::Ptr(cell._block));
        cell._row = this;
        cell._index = i++;
    }
    
    layout->set_origin(Vec2(0.5));
    layout->set_background(Transparent);
}

void Row::clear() {
    for(auto &cell : _cells) {
        cell.set_sample(nullptr);
    }
}

Cell& Row::cell(size_t i) {
    assert(length() > i);
    return _cells.at(i);
}

size_t Row::length() const {
    assert(layout);
    return layout->children().size();
}

void Row::update(DrawStructure& base, double dt) {
    if(!layout->parent())
        return;
    
    for (size_t i=0; i<length(); ++i) {
        auto &cell = this->cell(i);
        
        if(cell._sample) {
            auto d = euclidean_distance(base.mouse_position(), cell.bounds().pos() + cell.bounds().size() * 0.5)
                / (layout->parent()->global_bounds().size().length() * 0.45);
            if(d > 0)
                cell._block->set_scale(Vec2(0.8 + 0.2 / (1 + d * d)) * (cell.selected() ? 1.5 : 1));
            
            const double seconds_for_all_samples = (cell._image->hovered() ? 15.0 : 2.0);
            const double samples_per_second = cell._sample->_images.size() / seconds_for_all_samples;
            
            cell._animation_time += dt * samples_per_second;
            
            if(size_t(cell._animation_time) != cell._animation_index) {
                cell._animation_index = size_t(cell._animation_time);
                
                if(cell._animation_index >= cell._sample->_images.size()) {
                    cell._animation_index = 0;
                    cell._animation_time = 0;
                }
                
                auto &ptr = cell._sample->_images.at(cell._animation_index);
                Image inverted(ptr->rows, ptr->cols, 1);
                std::transform(ptr->data(), ptr->data() + ptr->size(), inverted.data(),
                    [&ptr, s = ptr->data(), pos = cell._sample->_positions.at(cell._animation_index)](uchar& v) -> uchar
                    {
                        /*auto d = std::distance(s, &v);
                        auto x = d % ptr->cols;
                        auto y = (d - x) / ptr->cols;
                        auto bg = Tracker::instance()->background();
                        if(bg->bounds().contains(Vec2(x+pos.x, y+pos.y)))
                            return saturate((int)Tracker::instance()->background()->color(x + pos.x, y + pos.y) - (int)v);*/
                        return 255 - v;
                    });
                
                cell._image->update_with(std::move(inverted));
                cell.update_scale();
                cell._block->auto_size(Margin{0, 0});
            }
            
        } else {
            //std::fill(cell._image->source()->data(), cell._image->source()->data() + cell._image->source()->size(), 0);
        }
        
        auto s = min(1, cell._block->scale().x / 1.5);
        //s = SQR(s) * SQR(s);
        //s = SQR(s) * SQR(s);
        
        cell.update(s);
    }
}

void Row::update(size_t cell_index, const Sample::Ptr& sample) {
    auto &cell = this->cell(cell_index);
    cell.set_sample(sample);
    
    layout->auto_size();
}

bool Row::empty() const {
    return layout->empty();
}

Interface& Interface::get() {
    static std::unique_ptr<Interface> obj;
    if (!obj) {
        obj = std::make_unique<Interface>();
    }
    return *obj;
}

void Interface::clear_probabilities() {
    std::lock_guard g(Work::recv_mutex());
    for(auto &row : Row::rows()) {
        for(auto &cell : row._cells) {
            if(cell._sample) {
                cell._sample->_probabilities.clear();
                cell._max_id = -1;
                cell._sample->_requested = false;
            }
        }
    }
}

void Interface::clear_rows() {
    LOGGED_LOCK_TYPE<std::recursive_mutex> gui_guard;
    if(Interface::get().layout.stage()) {
        gui_guard = GUI_LOCK(Interface::get().layout.stage()->lock());
    }
    
    for(auto &row : Row::rows()) {
        row.clear();
    }
}

void Interface::reset() {
    std::lock_guard g(Work::recv_mutex());
    for(auto &row : Row::rows()) {
        for(auto &cell : row._cells) {
            if(cell._sample) {
                cell._sample->_probabilities.clear();
                cell._sample->_requested = false;
            }
            cell.set_sample(nullptr);
        }
    }
}

void Interface::init(pv::File& video, IMGUIBase* window, DrawStructure& base) {
    static double R = 0, elap = 0;
    static Timer timer;
    //R += RADIANS(100) * timer.elapsed();
    elap += timer.elapsed();

    static bool initialized = false;
    if (!initialized) {
        //PythonIntegration::ensure_started();
        //PythonIntegration::async_python_function([]()->bool{return true;});
        //Work::start_learning();

        elap = 0;
        initialized = true;

        _window = window;
        _video = &video; 

        layout.set_policy(gui::VerticalLayout::CENTER);
        layout.set_origin(Vec2(0.5));
        layout.set_pos(Size2(base.width(), base.height()) * 0.5);

        stext = Layout::Make<StaticText>(
            Str("<h2>Categorizing types of individuals</h2>"
            "Below, an assortment of randomly chosen clips is shown. They are compiled automatically to (hopefully) only contain samples belonging to the same category. Choose clips that best represent the categories you have defined before (<str>" + Meta::toStr(DataStore::label_names()) + "</str>) and assign them by clicking the respective button. But be careful - with them being automatically collected, some of the clips may contain images from multiple categories. It is recommended to <b>Skip</b> these clips, lest risking to confuse the poor network. Regularly, when enough new samples have been collected (and for all categories), they are sent to said network for a training step. Each training step, depending on clip quality, should improve the prediction accuracy (see below)."),
            SizeLimit(base.width() * 0.75 * base.scale().x, -1), Font(0.7)
            );

        layout.add_child(stext);

        apply->on_click([video = _video](auto) {
            Work::set_state(video, Work::State::APPLY);
        });
        close->on_click([video = _video](auto) {
            Work::set_state(video, Work::State::NONE);
            });
        load->on_click([video = _video](auto) {
            Work::set_state(video, Work::State::LOAD);
            });
        restart->on_click([video = _video](auto) {
            Work::learning() = false;
            Work::learning_variable().notify_all();
            DataStore::clear();
            //PythonIntegration::quit();

            Work::set_state(video, Work::State::SELECTION);
        });
        reapply->on_click([video = _video](auto) {
            DataStore::clear();
            Categorize::clear_labels();
            Work::set_state(video, Work::State::APPLY);
        });
        train->on_click([](auto) {
            if (Work::state() == Work::State::SELECTION) {
                Work::add_training_sample(nullptr);
            }
            else
                FormatWarning("Not in selection mode. Can only train while samples are being selected, not during apply or inactive.");
            });
        shuffle->on_click([](auto) {
            LOGGED_LOCK_TYPE<std::recursive_mutex> gui_guard;
            if(Interface::get().layout.stage()) {
                gui_guard = GUI_LOCK(Interface::get().layout.stage()->lock());
            }
            
            for (auto& row : Row::rows()) {
                for (size_t i = 0; i < row._cells.size(); ++i) {
                    row.update(i, retrieve());
                }
            }
            });

        apply.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Green.exposure(0.15)));
        close.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Red.exposure(0.2)));
        load.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Yellow.exposure(0.2)));
        shuffle.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Yellow.exposure(0.5)));

        tooltip.set_scale(base.scale().reciprocal());
        tooltip.text().set_default_font(Font(0.5));

        for (auto& row : Row::rows()) {
            /**
             * If the row is empty, that means that the whole grid has not been initialized yet.
             * Create images and put them into a per_row^2 grid, and add them to the layout.
             */
            if (row.empty()) {
                row.init(per_row);
                layout.add_child(Layout::Ptr(row.layout));
            }

            /**
             * After ensuring we do have rows, fill them with new samples:
             */
            for (size_t i = 0; i < row.length(); ++i) {
                auto sample = retrieve();
                row.update(i, sample);
            }
        }

        if (!layout.empty() && layout.children().back() != buttons.get()) {
            desc_text.to<StaticText>()->set_default_font(Font(0.6));
            desc_text.to<StaticText>()->set_max_size(stext.to<StaticText>()->max_size());

            layout.add_child(desc_text);
            layout.add_child(buttons);
        }

        layout.auto_size();
        layout.set_z_index(1);
    }

    timer.reset();
}

void Interface::draw(pv::File& video, IMGUIBase* window, DrawStructure& base) {
    {
        std::lock_guard guard(DataStore::mutex());
        /*if(_labels.empty()) {
            _labels.insert({Label::Make("W"), {}});
            _labels.insert({Label::Make("S"), {}});
            //DataStore::_labels.insert({Label::Make("X"), {}});
        }*/

        if (FAST_SETTING(categories_ordered).empty()) {
            static bool asked = false;
            if (!asked) {
                asked = true;

                using namespace gui;
                static Layout::Ptr textfield;

                auto d = base.dialog([](Dialog::Result r) {
                    if (r == Dialog::OKAY) {
                        std::vector<std::string> categories;
                        for (auto text : utils::split(textfield.to<Textfield>()->text(), ',')) {
                            text = utils::trim(text);
                            if (!text.empty())
                                categories.push_back(text);
                        }
                        SETTING(categories_ordered) = categories;

                        for (auto& cat : categories)
                            DataStore::label(cat.c_str()); // create labels
                    }

                    }, "Please enter the categories (comma-separated), e.g.:\n<i>W,S</i> for categories <str>W</str> and <str>S</str>.", "Categorize", "Okay", "Cancel");
                
                textfield = Layout::Make<Textfield>(Str("W,S"), Box(Size2(d->layout().width() * 0.75, 33)));
                textfield->set_size(Size2(d->layout().width() * 0.75, 33));
                d->set_custom_element(textfield);
                d->layout().Layout::update_layout();
            }
            return;
        }
    }

    using namespace gui;
    static Rect rect(FillClr{Black.alpha(125)});

    auto screen_size = (window ? (window->window_dimensions().div(base.scale())) : Size2(base.width(), base.height())) * gui::interface_scale();
    auto center = screen_size * 0.5;
    layout.set_pos(center);
    
    rect.set_z_index(1);
    rect.set_size(screen_size);

    base.wrap_object(rect);

    init(video, window, base);

    layout.auto_size();
    base.wrap_object(layout);

    static Timer timer;

    float max_w = 0;
    for (auto& row : Row::rows()) {
        for (auto& cell : row._cells) {
            if (!cell._sample)
                row.update(cell._index, retrieve());

            cell._block->auto_size(Margin{ 0,0 });
            max_w = max(max_w, cell._block->global_bounds().width);
        }
        row.update(base, timer.elapsed());
    }

    max_w = per_row * (max_w + 10);
    
    const double hard_limit = min(1200, screen_size.width * base.scale().x * 0.8);
    max_w = min(hard_limit, abs(max_w));
    
#if __APPLE__
    //max_w = min(window.width * 0.9, max_w * 1.25 * base.scale().x);
#else
    //max_w = min(window.width * 0.9, max_w * 1.25 * base.scale().x);
#endif
    //max_w = max(base.width() * 0.5, max_w * 1.25);

    static bool redrawing = true;
    static float previous_max = 100;
    static Timer draw_timer;
    if (abs(max_w - previous_max) > abs(previous_max)
        || previous_max > hard_limit)
    {
        if (redrawing) {
            previous_max += (max_w - previous_max) * 0.5 * draw_timer.elapsed() * 10;
            draw_timer.reset();
        }
        else if (draw_timer.elapsed() > 1) {
            redrawing = true;
            draw_timer.reset();
        }
    }
    else
        redrawing = false;
    
    previous_max = max_w;

    if (Work::initialized()) {
        auto all_options = std::vector<Layout::Ptr>{ restart, load, train, shuffle, close };
        if (Work::best_accuracy() >= Work::good_enough() * 0.5) {
            if(!DataStore::empty()) {
                all_options.insert(all_options.begin(), reapply);
                apply.to<Button>()->set_txt("Continue");
            } else
                apply.to<Button>()->set_txt("Apply");
            
            all_options.insert(all_options.begin(), apply);
        }

        if (buttons.to<HorizontalLayout>()->children().size() != all_options.size())
            buttons.to<HorizontalLayout>()->set_children(all_options);
    }
    else {
        auto no_network = std::vector<Layout::Ptr>{
            shuffle, close
        };

        if (buttons.to<HorizontalLayout>()->children().size() != no_network.size())
            buttons.to<HorizontalLayout>()->set_children(no_network);
    }

    if (buttons) buttons->set_scale(base.scale().reciprocal());
    if (desc_text) desc_text->set_scale(base.scale().reciprocal());

    if (stext) {
        stext->set_scale(base.scale().reciprocal());
        stext.to<StaticText>()->set_max_size(Size2(int(previous_max), -1));
        if (desc_text)
            desc_text.to<StaticText>()->set_max_size(stext.to<StaticText>()->max_size());
    }

    timer.reset();

    auto txt = settings::htmlify(Meta::toStr(DataStore::composition()));
    if (Work::best_accuracy() < Work::good_enough()) {
        txt = "<i>Predictions for all visible tiles will be displayed as soon as the network becomes confident enough.</i>\n" + txt;
    }
    desc_text.to<StaticText>()->set_txt(txt);
}

}
}
