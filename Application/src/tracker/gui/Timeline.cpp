#include "Timeline.h"
#include <gui/DrawCVBase.h>
#include <file/CSVExport.h>
#include <gui/HttpGui.h>
#include <processing/PadImage.h>
#include <tracking/FOI.h>
#include <misc/ReverseAdapter.h>
#include <tracking/DatasetQuality.h>
#include <gui/types/Button.h>
#include <misc/EventAnalysis.h>
#include <gui/GUICache.h>
#include <gui/GuiTypes.h>
#include <misc/vec2.h>
#include <tracking/IndividualManager.h>

namespace gui {
    //! NeighborDistances drawn out
    struct ProximityBar {
        Size2 _dimensions;
        Frame_t start, end;
        std::map<Frame_t, std::set<FOI::fdx_t>> changed_frames;
        std::vector<uint32_t> samples_per_pixel;
        std::mutex mutex;

    } _proximity_bar;

    std::atomic<Frame_t> tracker_endframe, tracker_startframe;

    std::string _thread_status;
    std::atomic_bool _terminate;
    std::mutex _terminate_mutex;
    std::condition_variable _terminate_variable;

    const Tracker* _tracker{nullptr};
    FrameInfo* _frame_info{nullptr};
    Timeline* _instance{ nullptr };

    std::shared_ptr<std::thread> _update_events_thread;

    struct {
        uint64_t last_change;
        FOI::foi_type::mapped_type changed_frames;
        std::string name;
        Color color;
    } _foi_state;

    const float bar_height = 30;

    struct Interface {
        HorizontalLayout _title_layout{ {}, Vec2(20, 20), Bounds(0, 0, 17, 0) };
        Text _status_text{ Font(0.8f) },
             _status_text2{ Font(0.8f) },
             _status_text3{ Font(0.8f) };
        Text _raw_text{ "[RAW]", Black, Font(0.8f, Style::Bold) },
             _auto_text{Black, Font(0.8f, Style::Bold) };
        Button _pause{ "pause", Bounds(0, 0, 100, 27) };
        Timeline* _ptr;

        Interface(Timeline * ptr) : _ptr(ptr) {
            std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
            //if(_proximity_bar.image.empty())
            //    _proximity_bar.image.create(cv::Mat::zeros(20, 20, CV_8UC4));

            _status_text.set_clickable(true);
            _status_text.on_hover([this](auto e) {
                if(_ptr->_hover_status_text)
                    _ptr->_hover_status_text(e.hover.hovered);
            });

            _status_text2.on_click([](auto) {
                SETTING(gui_frame) = _frame_info->global_segment_order.empty() ? Frame_t(0) : _frame_info->global_segment_order.front().start;
            });
            _status_text2.on_hover([this](auto) {
                _status_text2.set_dirty();
            });
            _status_text2.set_clickable(true);

            _pause.on_click([](auto) {
                Tracker::analysis_state(SETTING(analysis_paused).value<bool>() ? Tracker::AnalysisState::UNPAUSED : Tracker::AnalysisState::PAUSED);
            });
        }

        static Interface& get(Timeline* timeline) {
            static std::unique_ptr<Interface> obj;
            if (!obj) {
                obj = std::make_unique<Interface>(timeline);
            }
            return *obj;
        }

        void draw(DrawStructure& base) {
            auto& _bar = _ptr->_bar;

            gui::DrawStructure::SectionGuard section(base, "timeline");
            const Vec2 use_scale = base.scale().reciprocal();
            auto&& [offset, max_w] = Timeline::timeline_offsets(_ptr->_base);
            const float y = 55;

            section._section->set_scale(use_scale);

            if (FAST_SETTING(analysis_paused)) {
                _pause.set_txt("continue");
                _pause.set_fill_clr(Color(25, 75, 25, GUI_SETTINGS(gui_timeline_alpha)));
            }
            else {
                _pause.set_txt("pause");
                _pause.set_fill_clr(Color(75, 25, 25, GUI_SETTINGS(gui_timeline_alpha)));
            }
            
            std::stringstream number;
            decltype(_frame_info->global_segment_order) segments;
            decltype(segments)::value_type consec;
            std::vector<Range<Frame_t>> other_consec;
            
            {
                std::unique_lock info_guard(Timeline::_frame_info_mutex);
                number << _frame_info->frameIndex.load().toStr() << "/" << _frame_info->video_length << ", " << _frame_info->big_count << " tracks";
                if (_frame_info->small_count)
                    number << " (" << _frame_info->small_count << " short)";
                if (_frame_info->up_to_this_frame != _frame_info->big_count)
                    number << ", " << _frame_info->up_to_this_frame << " known here";

                if (_frame_info->current_count != FAST_SETTING(track_max_individuals))
                    number << " " << _frame_info->current_count << " this frame";

                if (!FAST_SETTING(analysis_paused))
                    number << " (analysis: " << _frame_info->current_fps << " fps)";
                else
                    number << " (analysis paused)";

                DurationUS duration{ uint64_t((double(_frame_info->frameIndex.load().get()) / double(FAST_SETTING(frame_rate)))) * 1000u * 1000u };
                number << " " << Meta::toStr(duration);

                _status_text.set_txt(number.str());
                number.str("");

                segments = _frame_info->global_segment_order;
                consec = segments.empty() ? Range<Frame_t>({}, {}) : segments.front();
                
                if (segments.size() > 1) {
                    for (size_t i = 0; i < 3 && i < segments.size(); ++i) {
                        other_consec.push_back(segments.at(i));
                    }
                }
                number << "consec: " << consec.start.toStr() << "-" << consec.end.toStr() << " (" << (consec.end - consec.start).toStr() << ")";

                Color consec_color = White;
                if (consec.contains(_frame_info->frameIndex))
                    consec_color = Green;

                if (_status_text2.hovered())
                    consec_color = consec_color.exposureHSL(0.9f);
                _status_text2.set_color(consec_color);

                _status_text2.set_txt(number.str());
                number.str("");

                number << "delta: " << _frame_info->tdelta << "s";
            }
            //number << ", " << tracker_endframe << ")";
            //number << " tdelta:" << std::fixed << std::setprecision(3) << tdelta << " " << std::fixed << std::setprecision(3) << _frame_info->tdelta_gui;
            number << NetworkStats::status();

            auto status = EventAnalysis::status();
            if (!status.empty())
                number << " " << status;
            //number << " midline-err/frame:" << Tracker::instance()->midline_errors_frame();

            _title_layout.set_pos(Vec2(20, 25) - offset);
            _title_layout.set_origin(Vec2(0, 0.5));
            _status_text3.set_txt(number.str());

            _title_layout.set_policy(HorizontalLayout::Policy::CENTER);
            std::vector<Layout::Ptr> in_layout{ &_pause, &_status_text, &_status_text2, &_status_text3 };
            if (_frame_info->video_length == uint64_t(tracker_endframe.load().get())) {
                in_layout.erase(in_layout.begin());
            }

            if (GUI_SETTINGS(gui_mode) == mode_t::blobs && _ptr->bar()) {
                base.rect(Bounds(-offset, Size2(max_w, bar_height + y)), FillClr{Red.alpha(75)});
                in_layout.insert(in_layout.begin(), &_raw_text);
            }

            if (GUI_SETTINGS(auto_categorize)) {
                base.rect(Bounds(-offset, Size2(max_w, bar_height + y)), FillClr{Purple.alpha(75)});
                _auto_text.set_txt("[auto_categorize]");
                in_layout.insert(in_layout.begin(), &_auto_text);

            }
            else if (GUI_SETTINGS(auto_train)) {
                base.rect(Bounds(-offset, Size2(max_w, bar_height + y)), FillClr{Red.alpha(75)});
                _auto_text.set_txt("[auto_train]");
                in_layout.insert(in_layout.begin(), &_auto_text);

            }
            else if (GUI_SETTINGS(auto_apply)) {
                base.rect(Bounds(-offset, Size2(max_w, bar_height + y)), FillClr{Red.alpha(75)});
                _auto_text.set_txt("[auto_apply]");
                in_layout.insert(in_layout.begin(), &_auto_text);
            }
            else if (GUI_SETTINGS(auto_quit)) {
                base.rect(Bounds(-offset, Size2(max_w, bar_height + y)), FillClr{Red.alpha(75)});
                _auto_text.set_txt("[auto_quit]");
                in_layout.insert(in_layout.begin(), &_auto_text);
            }

            _title_layout.set_children(in_layout);
            base.wrap_object(_title_layout);

            Vec2 pos = Vec2(0, y) - offset;

            gui::Color red_bar_clr(250, 250, 250, uchar(GUI_SETTINGS(gui_timeline_alpha) * (_bar && _bar->hovered() ? 1.f : 0.8f)));

            base.rect(Bounds(pos, Size2(max_w, bar_height)), FillClr{White.alpha(175)});

            float percent = float(tracker_endframe.load().get()) / _frame_info->video_length;
            base.rect(Bounds(pos, Size2(max_w * percent, bar_height)), FillClr{red_bar_clr});

            std::unique_lock info_lock(Timeline::_frame_info_mutex);
            if (_bar && use_scale.y > 0) {
                std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
                float new_height = roundf(bar_height);

                _bar->set_scale(Vec2(1, new_height));
                _bar->set_color(White.alpha(GUI_SETTINGS(gui_timeline_alpha)));
                _bar->set_pos(pos);
                base.wrap_object(*_bar);

                {
                    _ptr->update_consecs(max_w, consec, other_consec, 1);
                    if (_ptr->_consecutives) {
                        _ptr->_consecutives->set_pos(pos - Vec2(0, 5));
                        base.wrap_object(*_ptr->_consecutives);
                    }
                }

                base.add_object(new Text(Meta::toStr(tracker_endframe.load()), Loc(pos + Vec2(max_w * tracker_endframe.load().get() / float(_frame_info->video_length) + 5, bar_height * 0.5f)), Black, Font(0.5), Origin(0, 0.5)));

                // display hover sign with frame number
                if (_ptr->_mOverFrame.valid()) {
                    //auto it = _proximity_bar.changed_frames.find(_mOverFrame);
                    //if(it != _proximity_bar.changed_frames.end() || _mOverFrame >= _proximity_bar.end)
                    {
                        std::string t = "Frame " + _ptr->_mOverFrame.toStr();
                        auto dims = Base::text_dimensions(t, &_title_layout, Font(0.7f));

                        Vec2 pp(
                            max_w / float(_frame_info->video_length) * _ptr->_mOverFrame.get(),
                            _bar->pos().y + _bar->global_bounds().height / use_scale.y + dims.height * 0.5f + 2);

                        if (pp.x < dims.width * 0.5f)
                            pp.x = dims.width * 0.5f;
                        if (pp.x + dims.width * 0.5f > max_w)
                            pp.x = max_w - dims.width * 0.5f;

                        pp -= offset;

                        base.rect(Bounds(pp - dims * 0.5f - Vec2(5, 2), dims + Vec2(10, 4)), FillClr{Black.alpha(125)});
                        base.text(t, Loc(pp), Font(0.7f, Align::Center));
                    }
                }
            }

            if (_frame_info->analysis_range.start != 0_f) {
                auto start_pos = pos;
                auto end_pos = Vec2(max_w / float(_frame_info->video_length) * _frame_info->analysis_range.start.get(), bar_height);
                base.rect(Bounds(start_pos, end_pos), FillClr{Gray});
            }
            if (uint64_t(_frame_info->analysis_range.end.get()) <= _frame_info->video_length) {
                auto start_pos = pos + Vec2(max_w / float(_frame_info->video_length) * _frame_info->analysis_range.end.get(), 0);
                auto end_pos = Vec2(max_w, bar_height);
                base.rect(Bounds(start_pos, end_pos), FillClr{Gray});
            }

            // current position indicator
            auto current_pos = Vec2(max_w / float(_frame_info->video_length) * _frame_info->frameIndex.load().get(), y) - offset;
            base.rect(Bounds(current_pos - Vec2(2),
                             Size2(5, bar_height + 4)),
                      FillClr{Black.alpha(255)},
                      LineClr{White.alpha(255)});
            //base.rect(current_pos - Vec2(2, 1), Vec2(5, 2), DarkCyan, Black);
            //base.rect(current_pos + Vec2(-2, bar_height + 3), Vec2(5, 2), DarkCyan, Black);
        }
    };

    Timeline::~Timeline() {
        _terminate = true;
        _terminate_variable.notify_all();
        _update_events_thread->join();
    }
    
    std::tuple<Vec2, float> Timeline::timeline_offsets(Base* base) {
        //const float max_w = Tracker::average().cols;
        const float max_w = _instance && !_terminate && base ? base->window_dimensions().width * gui::interface_scale() : Tracker::average().cols;
        Vec2 offset(0);
        return {offset, max_w};
    }

    Timeline& instance() {
        if (!_instance)
            throw U_EXCEPTION("No timeline has been created.");
        return *_instance;
    }
    
    Timeline::Timeline(Base* base, std::function<void(bool)> hover_status, std::function<void()> updated_rec_rect, FrameInfo& info)
        : _bar(nullptr),
        tdelta(0),
        _visible(true),
        _base(base),
        _updated_recognition_rect(updated_rec_rect),
        _hover_status_text(hover_status)
    {
        _instance = this;
        //_gui = &gui;
        _tracker = Tracker::instance();
        _frame_info = &info;
        
        Interface::get(this);
        _update_events_thread = std::make_shared<std::thread>([this]() {
            set_thread_name("Timeline::update_thread");
            this->update_thread();
        });
    }

void Timeline::update_consecs(float max_w, const Range<Frame_t>& consec, const std::vector<Range<Frame_t>>& other_consec, float _scale) {
    if(!_bar)
        return;
    
    static Range<Frame_t> previous_consec({}, {});
    static std::vector<Range<Frame_t>> previous_other_consec = {};
    static uchar previous_alpha = 0;
    static float previous_scale = 0;
    const float scale = max(1, min(_scale, CV_MAX_THICKNESS));
    float new_height = roundf(bar_height) + 5 * scale;
    
    const uchar alpha = GUI_SETTINGS(gui_timeline_alpha);
    
    if(consec == previous_consec
       && other_consec == previous_other_consec
       && scale == previous_scale
       && _consecutives && _bar
       && _bar->source()->cols == _consecutives->source()->cols
       && _consecutives->source()->rows == uint(new_height)
       && alpha == previous_alpha)
        return;
    
    previous_consec = consec;
    previous_other_consec = other_consec;
    previous_scale = scale;
    previous_alpha = alpha;
    
    if(!_consecutives
       || _bar->source()->cols != _consecutives->source()->cols
       || _consecutives->source()->rows != new_height)
    {
        auto image = Image::Make(new_height, _bar->source()->cols, 4);
        if(!_consecutives)
            _consecutives = std::make_unique<ExternalImage>(std::move(image), Vec2());
        else
            _consecutives->update_with(std::move(*image));
    }
    
    std::fill(_consecutives->source()->data(), _consecutives->source()->data() + _consecutives->source()->size(), 0);
    auto mat = _consecutives->source()->get();
    auto offset = Vec2(0,5) * scale;
    
    if(!consec.empty()) {
        std::deque<Color> colors {
            Green,
            Yellow,
            Color(255,128,57,255),
            Gray
        };
        
        for(auto &consec : other_consec) {
            auto position = offset + Vec2(max_w * consec.start.get() / float(_frame_info->video_length), 0);
            auto size = Size2(max_w * (consec.end - consec.start).get() / float(_frame_info->video_length), bar_height);
            
            DEBUG_CV(cv::rectangle(mat, position - Vec2(1), position - Vec2(1) + size + Size2(2), colors.front().alpha(50), cv::FILLED));
            DEBUG_CV(cv::rectangle(mat, position - Vec2(1), position - Vec2(1) + size + Size2(2), Color(alpha, alpha, alpha, 255)));
            
            if(!colors.empty())
                colors.pop_front();
        }
    }
    
    //base.line(pos - Vec2(0,1), pos + Vec2(max_w * tracker_endframe / float(_frame_info->video_length), 0) - Vec2(0,1), 1, Red.alpha(255));
    
    auto thickness = narrow_cast<int>(scale);
    assert(thickness > 0);
    
    for(auto &consec : _frame_info->consecutive) {
        if( consec.length().get() > 2 && consec.length().get() >= consec.length().get() * 0.25) {
            auto position = offset + Vec2(max_w * consec.start.get() / float(_frame_info->video_length), 0);
            auto size = Size2(max_w * consec.length().get() / float(_frame_info->video_length), bar_height);
            
            --position.y;
            DEBUG_CV(cv::line(mat, position, position + Vec2(size.width, 0), Green.alpha(alpha)));
            DEBUG_CV(cv::line(mat, position, position - offset, Green.alpha(alpha), thickness));
        }
    }
    
    if(!_frame_info->training_ranges.empty()) {
        for(auto range : _frame_info->training_ranges) {
            auto position = offset + Vec2(max_w * range.start.get() / float(_frame_info->video_length), 0);
            auto size = Size2(max_w * range.length().get() / float(_frame_info->video_length), bar_height);
            
            DEBUG_CV(cv::rectangle(mat, position - Vec2(1), position - Vec2(1, 0) + size + Size2(2), Red.alpha(50), cv::FILLED));
            DEBUG_CV(cv::rectangle(mat, position - Vec2(1), position - Vec2(1, 0) + size + Size2(2), Color(alpha, alpha, alpha, 255)));
        }
    }
    
    _consecutives->set_dirty();
}
    
    void Timeline::draw(gui::DrawStructure &base) {
        Interface::get(this).draw(base);
    }
    
    void Timeline::update_fois() {
        static Timing timing("update_fois", 10);
        TakeTiming take(timing);
        
        const Vec2 use_scale = _bar ? _bar->stage_scale() : Vec2(1);
        auto && [offset, max_w] = timeline_offsets(_base);
        //const float max_h = Tracker::average().rows;
        
        // initialize and check FOI status
        {
            uint64_t last_change = FOI::last_change();
            auto name = SETTING(gui_foi_name).value<std::string>();

            if (last_change != _foi_state.last_change || name != _foi_state.name) {
                _foi_state.name = name;

                if (!_foi_state.name.empty()) {
                    long_t id = FOI::to_id(_foi_state.name);
                    if (id != -1) {
                        _foi_state.changed_frames = FOI::foi(id);//_tracker->changed_frames();
                        _foi_state.color = FOI::color(_foi_state.name);
                    }
                }

                _foi_state.last_change = last_change;
            }
        }
        
        // initialize the bar if not yet done
        if (_bar == NULL) {
            _bar = std::make_unique<ExternalImage>(Image::Make(), Vec2());
            _bar->set_color(White.alpha(GUI_SETTINGS(gui_timeline_alpha)));
            _bar->set_clickable(true);
            _bar->on_hover([this](Event e) 
                {
                if (!GUICache::exists())
                    return;

                auto&& [offset_, max_w_] = timeline_offsets(_base);

                //if(!_proximity_bar.changed_frames.empty())
                {
                    float distance2frame = FLT_MAX;
                    Frame_t framemOver;

                    if (_bar && _bar->hovered()) {
                        std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
                        //Vec2 pp(max_w / float(_frame_info->video_length) * idx.first, 50);
                        //float dis = abs(e.hover.x - pp.x);
                        static Timing timing("Scrubbing", 0.01);
                        Frame_t idx = Frame_t(roundf(e.hover.x / max_w_ * float(_frame_info->video_length)));
                        auto it = _proximity_bar.changed_frames.find(idx);
                        if (it != _proximity_bar.changed_frames.end()) {
                            framemOver = idx;
                            distance2frame = 0;
                        }
                        else if ((it = _proximity_bar.changed_frames.find(idx - 1_f)) != _proximity_bar.changed_frames.end()) {
                            framemOver = idx - 1_f;
                            distance2frame = 1;
                        }
                        else if ((it = _proximity_bar.changed_frames.find(idx + 1_f)) != _proximity_bar.changed_frames.end()) {
                            framemOver = idx + 1_f;
                            distance2frame = 1;
                        }
                    }

                    if (distance2frame < 2) {
                        _mOverFrame = framemOver;

                    }
                    else if (_bar->hovered()) {
                        if (tracker_endframe.load().valid()) {
                            _mOverFrame = Frame_t(min(float(_frame_info->video_length), e.hover.x * float(_frame_info->video_length) / max_w_));
                            _bar->set_dirty();
                        }
                    }
                    else
                        _mOverFrame.invalidate();
                }

                if (_bar->hovered() && _bar->pressed() && this->mOverFrame().valid())
                {
                    SETTING(gui_frame) = Frame_t(this->mOverFrame());
                }
            });

            _bar->add_event_handler(MBUTTON, [this](Event e) {
                if (e.mbutton.pressed && this->mOverFrame().valid() && e.mbutton.button == 0) {
                    //_gui->set_redraw();
                    GUICache::instance().set_redraw();
                    SETTING(gui_frame) = this->mOverFrame();
                }
            });
        }

        bool changed = false;
        
        if(use_scale.y > 0) {
            std::unique_lock guard(_proximity_bar.mutex);
            static std::string last_name;
            
            // update proximity bar, whenever the FOI to display changed
            if(last_name != _foi_state.name)
                _proximity_bar.end.invalidate();
            last_name = _foi_state.name;
            
            if(!_bar
               || (uint)max_w != _bar->source()->cols
               || (uint)1 != _bar->source()->rows
               || !_proximity_bar.end.valid())
            {
                auto image = Image::Make(1, max_w, 4);
                image->set_to(0);
                
                _proximity_bar.end.invalidate();
                _proximity_bar.start.invalidate();
                
                changed = true;
                _proximity_bar.changed_frames.clear();

                if (_bar->parent() && _bar->parent()->stage()) {
                    guard.unlock();
                    try {
                        std::lock_guard<std::recursive_mutex> lock(_bar->parent()->stage()->lock());
                        _bar->set_source(std::move(image));
                    }
                    catch (...) {}
                    guard.lock();
                    
                } else
                    return;
            }
        }
        
        if(tracker_endframe.load().valid() && _proximity_bar.end < tracker_endframe.load()) {
            cv::Mat img;
            if (_bar->parent() && _bar->parent()->stage()) {
                std::lock_guard lock(_bar->parent()->stage()->lock());
                img = _bar->source()->get();
            } else
                return;

            std::unique_lock guard(_proximity_bar.mutex);
            auto individual_coverage = [](Frame_t frame) {
                LockGuard guard(ro_t{}, "Timeline::individual_coverage", 100);
                float count = 0;
                if(Tracker::properties(frame)) {
                    for(auto fish : Tracker::instance()->active_individuals(frame)) {
                        if(fish->centroid(frame))
                            count++;
                    }
                }
                return (1 - count / float(IndividualManager::num_individuals()));
            };
            
            if(!_proximity_bar.end.valid()) {
                _proximity_bar.start = _proximity_bar.end = _tracker->start_frame();
            }
            
            Vec2 pos(max_w / float(_frame_info->video_length) * _proximity_bar.end.get(), 0);
            
            if(_proximity_bar.end < _tracker->end_frame()) {
                _proximity_bar.end = _tracker->end_frame();
                changed = true;
            }
            
            Vec2 previous_point(-1, 0);
            _proximity_bar.samples_per_pixel.resize(max_w);
            for(auto & px : _proximity_bar.samples_per_pixel)
                px = 0;
            assert(max_w > 0);
            std::set<uint32_t> multiple_assignments;
            
            for (auto &c : _foi_state.changed_frames) {
                float x = round(max_w / float(_frame_info->video_length) * c.frames().start.get());
                if(x >= _proximity_bar.samples_per_pixel.size())
                    x = _proximity_bar.samples_per_pixel.size()-1;
                ++_proximity_bar.samples_per_pixel[uint32_t(x)];
                if(_proximity_bar.samples_per_pixel[uint32_t(x)] > 1)
                    multiple_assignments.insert(uint32_t(x));
                
                auto it = _proximity_bar.changed_frames.find(c.frames().start);
                if(it == _proximity_bar.changed_frames.end() || it->second != c.fdx())
                {
                    _proximity_bar.changed_frames[c.frames().start] = c.fdx();
                    changed = true;
                }
            }
            
            if(changed) {
                uint32_t px;
                uint32_t N = 1;
                if(multiple_assignments.size() >= _foi_state.changed_frames.size() * 0.1)
                {
                    uint32_t Nx = 0;
                    for(auto x : multiple_assignments) {
                        auto n = _proximity_bar.samples_per_pixel[x];
                        if(n > N) {
                            N = n;
                            Nx = x;
                        }
                    }
                }
                
                if(img.cols > 0 && img.rows > 0) {
                    for(size_t i=0; i<_proximity_bar.samples_per_pixel.size(); ++i) {
                        px = _proximity_bar.samples_per_pixel[i];
                        if(px > 0) {
                            Vec2 pp(i, 0);
                            float d = float(px) / float(N);
                            img(Bounds(pp, Size2(1, img.rows))) = (cv::Scalar)_foi_state.color.alpha(50 + 205 * min(1, SQR(d)));
                        }
                        //cv::rectangle(img, pp, pp+Vec2(1,img.rows), _foi_state.color, -1);
                    }
                    
                    float x = max_w / float(_frame_info->video_length) * tracker_endframe.load().get();
                    if(previous_point.x != -1 && previous_point.x != x)
                    {
                        Vec2 point(x, individual_coverage(tracker_endframe) * img.rows);
                        DEBUG_CV(cv::line(img, previous_point, Vec2(x-1, previous_point.y), Red));
                        DEBUG_CV(cv::line(img, Vec2(x-1, previous_point.y), point, Red));
                    }
                }
            }
            
            /*if(!_bar || !_bar->source()->operator==(_proximity_bar.image)) {
                _proximity_bar.changed = true;
            }*/
        }
    }
    
    void Timeline::update_thread() {
        _terminate = false;
        _terminate_variable.notify_all();
        
        _foi_state.color = Color(255, 200, 100, 255);
        _foi_state.last_change = 0;
        _foi_state.name = "";
        
        auto long_wait_time = std::chrono::seconds(1);
        auto short_wait_time = std::chrono::milliseconds(5);
        
        std::unique_lock tmut(_terminate_mutex);
        while(!_terminate) {
            bool changed = false;
            
            //! Update the cached data
            if(GUICache::exists() && Tracker::instance()) {
                {
                    LockGuard guard(ro_t{}, "Timeline::update_thread", 100);
                    if (guard.locked()) {
                        Timer timer;

                        auto index = _frame_info->frameIndex.load();
                        auto props = index.valid() ? Tracker::properties(index) : nullptr;
                        auto prev_props = index.valid() && index > 0_f ? Tracker::properties(index - 1_f) : nullptr;

                        {
                            std::unique_lock info_lock(_frame_info_mutex);

                            _frame_info->tdelta = props && prev_props ? props->time - prev_props->time : 0;
                            _frame_info->small_count = 0;
                            _frame_info->big_count = 0;
                            _frame_info->current_count = 0;
                            _frame_info->up_to_this_frame = 0;
                            _frame_info->analysis_range = Tracker::analysis_range();
                            tracker_endframe = _tracker->end_frame();
                            tracker_startframe = _tracker->start_frame();

                            if (prev_props && props) {
                                tdelta = _frame_info->tdelta;
                            }

                            //_frame_info->training_ranges = _tracker->recognition() ? _tracker->recognition()->trained_ranges() : std::set<Range<Frame_t>>{};
                            _frame_info->consecutive = _tracker->consecutive();
                            _frame_info->global_segment_order = track::Tracker::global_segment_order();

                            if (props) {
                                for (auto& fish : index >= tracker_startframe.load() && index < tracker_endframe.load() ? Tracker::active_individuals(index) : set_of_individuals_t{}) {
                                    if ((int)fish->frame_count() < FAST_SETTING(frame_rate) * 3) {
                                        _frame_info->small_count++;
                                    }
                                    else
                                        _frame_info->big_count++;

                                    if (fish->has(_frame_info->frameIndex))
                                        ++_frame_info->current_count;

                                    if (fish->start_frame() <= _frame_info->frameIndex) {
                                        _frame_info->up_to_this_frame++;
                                    }
                                }
                            }
                        }

                        //if(FAST_SETTING(calculate_posture))
                        //    changed = EventAnalysis::update_events(_frame_info->frameIndex < tracker_endframe ? Tracker::active_individuals(_frame_info->frameIndex) : std::set<Individual*>{});

                        // needs Tracker lock
                        if (_updated_recognition_rect)
                            _updated_recognition_rect();


                        _update_thread_updated_once = true;

                        if (timer.elapsed() > 0.1 && !FAST_SETTING(analysis_paused)) {
                            if (long_wait_time < std::chrono::seconds(30)) {
                                long_wait_time = std::chrono::seconds(30);
                                short_wait_time = std::chrono::seconds(30);

                                if (!FAST_SETTING(analysis_paused))
                                    FormatWarning("Throtteling some non-essential gui functions until analysis is over.");
                            }

                        }
                        else {
                            long_wait_time = std::chrono::seconds(1);
                            short_wait_time = std::chrono::milliseconds(5);
                        }
                    }
                }

                //! TODO: Need to implement thread-safety for the GUI here. Currently very unsafe, for example when the GUI is deleted.
                update_fois();
            }

            _terminate_variable.wait_for(tmut, !changed ? long_wait_time : short_wait_time);
        }
    }
    
    void Timeline::reset_events(Frame_t after_frame) {
        EventAnalysis::reset_events(after_frame);
        
        {
            std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
            if(!after_frame.valid()) {
                _proximity_bar.changed_frames.clear();
                
            } else {
                for(auto iter = _proximity_bar.changed_frames.begin(), endIter = _proximity_bar.changed_frames.end(); iter != endIter; ) {
                    if (iter->first >= after_frame) {
                        _proximity_bar.changed_frames.erase(iter++);
                    } else {
                        ++iter;
                    }
                }
                
                if(_bar && !_bar->source()->empty()) {
                    cv::Mat img = _bar->source()->get();
                    
                    Vec2 pos(0, 0);
                    float x0 = Tracker::average().cols / float(_frame_info->video_length) * after_frame.get();
                    float x1 = Tracker::average().cols;
                    
                    print("Clearing from ", x0," to ",x1 + pos.x);
                    DEBUG_CV(cv::rectangle(img, Vec2(x0, 0), Vec2(pos + Vec2(x1, img.rows)), Transparent, -1));
                }
            }
            
            _proximity_bar.end = !after_frame.valid() ? Frame_t() : (after_frame - 1_f);
            //_proximity_bar.changed = true;
            
            tracker_endframe = _proximity_bar.end;
        }
    }

    void Timeline::set_visible(bool v) {
        if(instance()._visible != v) {
            GUICache::instance().set_tracking_dirty();
            GUICache::instance().set_redraw();
            //_gui->set_redraw();
            instance()._visible = v;
        }
    }

    bool Timeline::visible() {
        return instance()._visible;
    }
    
    void Timeline::next_poi(Idx_t _s_fdx) {
        auto frame = GUICache::instance().frame_idx;
        auto next_frame = frame;
        std::set<FOI::fdx_t> fdx;
        
        {
            std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
            for(auto && [idx, number] : _proximity_bar.changed_frames) {
                if(_s_fdx.valid()) {
                    if(number.find(FOI::fdx_t(_s_fdx)) == number.end())
                        continue;
                }
                
                if(idx > frame) {
                    next_frame = idx;
                    fdx = number;
                    break;
                }
            }
        }
        
        if(frame != next_frame) {
            SETTING(gui_frame) = next_frame;
            
            if(!_s_fdx.valid())
            {
                auto &cache = GUICache::instance();
                if(!fdx.empty()) {
                    cache.deselect_all();
                    for(auto id : fdx) {
                        if(!cache.is_selected(Idx_t(id.id)))
                            cache.do_select(Idx_t(id.id));
                    }
                }
            }
        }
    }
    
    void Timeline::prev_poi(Idx_t _s_fdx) {
        auto frame = GUICache::instance().frame_idx;
        auto next_frame = frame;
        std::set<FOI::fdx_t> fdx;
        
        {
            std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
            for(auto && [idx, number] : MakeReverse(_proximity_bar.changed_frames)) {
                if(_s_fdx.valid()) {
                    if(number.find(FOI::fdx_t(_s_fdx)) == number.end())
                        continue;
                }
                
                if(idx < frame) {
                    next_frame = idx;
                    fdx = number;
                    break;
                }
            }
        }
        
        if(frame != next_frame && next_frame.valid()) {
            SETTING(gui_frame) = next_frame;
            
            if(!_s_fdx.valid())
            {
                auto &cache = GUICache::instance();
                if(!fdx.empty()) {
                    cache.deselect_all();
                    for(auto id : fdx) {
                        if(!cache.is_selected(Idx_t(id.id)))
                            cache.do_select(Idx_t(id.id));
                    }
                }
            }
        }
    }
}
