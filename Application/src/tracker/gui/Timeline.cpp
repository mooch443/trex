#include "Timeline.h"
#include <gui/DrawCVBase.h>
#include <gui/gui.h>
#include <file/CSVExport.h>
#include <gui/HttpGui.h>
#include <processing/PadImage.h>
#include <tracking/FOI.h>
#include <misc/ReverseAdapter.h>
#include <tracking/Recognition.h>
#include <tracking/DatasetQuality.h>

namespace gui {
    const float bar_height = 30;
    static Timeline* _instance = nullptr;
    
    std::tuple<Vec2, float> Timeline::timeline_offsets() {
        //const float max_w = Tracker::average().cols;
        const float max_w = _instance && !_instance->_terminate && GUI::instance() && GUI::instance()->best_base() ? GUI::instance()->best_base()->window_dimensions().width * gui::interface_scale() : Tracker::average().cols;
        Vec2 offset(0);
        return {offset, max_w};
    }
    
    Timeline::Timeline(GUI& gui, FrameInfo& info)
        : _bar(nullptr),
        tracker_endframe(-1), tracker_startframe(-1),
        tdelta(0),
        _gui(gui),
        _tracker(*Tracker::instance()),
        _frame_info(info),
        _visible(true),
        _mOverFrame(-1),
        _title_layout({}, Vec2(20, 20), Bounds(0, 0, 17, 0)),
        _status_text("", Vec2(), White, 0.8f),
        _status_text2("", Vec2(), White, 0.8f),
        _status_text3("", Vec2(), White, 0.8f),
        _raw_text("[RAW]", Vec2(), Black, Font(0.8f, Style::Bold)),
        _auto_text("", Vec2(), Black, Font(0.8f, Style::Bold)),
        _pause("pause", Size2(100,27))
    {
        _instance = this;
        
        std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
        //if(_proximity_bar.image.empty())
        //    _proximity_bar.image.create(cv::Mat::zeros(20, 20, CV_8UC4));
        
        _status_text.set_clickable(true);
        _status_text.on_hover([](auto e) {
            if(!GUI::instance())
                return;
            if(e.hover.hovered) {
                GUI::instance()->set_info_visible(true);
            } else
                GUI::instance()->set_info_visible(false);
        });
        
        _status_text2.on_click([this](auto){
            SETTING(gui_frame) = _frame_info.global_segment_order.empty() ? 0 : _frame_info.global_segment_order.front().start;
        });
        _status_text2.on_hover([this](auto){
            _status_text2.set_dirty();
        });
        _status_text2.set_clickable(true);
        
        _pause.on_click([](auto) {
            Tracker::analysis_state((Tracker::AnalysisState)!SETTING(analysis_paused));
        });
        
        _update_events_thread = std::make_shared<std::thread>([this](){
            set_thread_name("Timeline::update_thread");
            this->update_thread();
        });
    }
    
    /*void Timeline::update_border() {
        std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
        _recognition_image.set_source(Image());
    }*/

void Timeline::update_consecs(float max_w, const Range<long_t>& consec, const std::vector<Rangel>& other_consec, float _scale) {
    if(!_bar)
        return;
    
    static Range<long_t> previous_consec(-1, -1);
    static std::vector<Rangel> previous_other_consec = {};
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
            _consecutives->update_with(*image);
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
            auto position = offset + Vec2(max_w * consec.start / float(_frame_info.video_length), 0);
            auto size = Size2(max_w * (consec.end - consec.start) / float(_frame_info.video_length), bar_height);
            
            DEBUG_CV(cv::rectangle(mat, position - Vec2(1), position - Vec2(1) + size + Size2(2), colors.front().alpha(50), cv::FILLED));
            DEBUG_CV(cv::rectangle(mat, position - Vec2(1), position - Vec2(1) + size + Size2(2), Color(alpha, alpha, alpha, 255)));
            
            if(!colors.empty())
                colors.pop_front();
        }
    }
    
    //base.line(pos - Vec2(0,1), pos + Vec2(max_w * tracker_endframe / float(_frame_info.video_length), 0) - Vec2(0,1), 1, Red.alpha(255));
    
    auto thickness = narrow_cast<int>(scale);
    assert(thickness > 0);
    
    for(auto &consec : _frame_info.consecutive) {
        if( consec.length() > 2 && consec.length() >= consec.length() * 0.25) {
            auto position = offset + Vec2(max_w * consec.start / float(_frame_info.video_length), 0);
            auto size = Size2(max_w * consec.length() / float(_frame_info.video_length), bar_height);
            
            --position.y;
            DEBUG_CV(cv::line(mat, position, position + Vec2(size.width, 0), Green.alpha(alpha)));
            DEBUG_CV(cv::line(mat, position, position - offset, Green.alpha(alpha), thickness));
        }
    }
    
    if(!_frame_info.training_ranges.empty()) {
        for(auto range : _frame_info.training_ranges) {
            auto position = offset + Vec2(max_w * range.start / float(_frame_info.video_length), 0);
            auto size = Size2(max_w * range.length() / float(_frame_info.video_length), bar_height);
            
            DEBUG_CV(cv::rectangle(mat, position - Vec2(1), position - Vec2(1, 0) + size + Size2(2), Red.alpha(50), cv::FILLED));
            DEBUG_CV(cv::rectangle(mat, position - Vec2(1), position - Vec2(1, 0) + size + Size2(2), Color(alpha, alpha, alpha, 255)));
        }
    }
    
    _consecutives->set_dirty();
}
    
    void Timeline::draw(gui::DrawStructure &base) {
        gui::DrawStructure::SectionGuard section(base, "timeline");
        const Vec2 use_scale = base.scale().reciprocal();
        auto && [offset, max_w] = timeline_offsets();
        const float y = 55;
        
        section._section->set_scale(use_scale);
        
        if(FAST_SETTINGS(analysis_paused)) {
            _pause.set_txt("continue");
            _pause.set_fill_clr(Color(25,75,25,GUI_SETTINGS(gui_timeline_alpha)));
        }
        else {
            _pause.set_txt("pause");
            _pause.set_fill_clr(Color(75,25,25,GUI_SETTINGS(gui_timeline_alpha)));
        }
        
        std::stringstream number;
        number << _frame_info.frameIndex << "/" << _frame_info.video_length << ", " << _frame_info.big_count << " tracks";
        if(_frame_info.small_count)
            number << " (" << _frame_info.small_count << " short)";
        if(_frame_info.up_to_this_frame != _frame_info.big_count)
            number << ", " << _frame_info.up_to_this_frame << " known here";
        
        if(_frame_info.current_count != FAST_SETTINGS(track_max_individuals))
            number << " " << _frame_info.current_count << " this frame";
        
        if(!FAST_SETTINGS(analysis_paused))
            number << " (analysis: " << _frame_info.current_fps << " fps)";
        else
            number << " (analysis paused)";
        
        DurationUS duration{uint64_t((double(_frame_info.frameIndex.load()) / double(FAST_SETTINGS(frame_rate)))) * 1000u * 1000u};
        number << " " << Meta::toStr(duration);
        
        _status_text.set_txt(number.str());
        number.str("");
        
        auto consec = _frame_info.global_segment_order.empty() ? Rangel(-1,-1) : _frame_info.global_segment_order.front();
        std::vector<Rangel> other_consec;
        if(_frame_info.global_segment_order.size() > 1) {
            for (size_t i=0; i<3 && i < _frame_info.global_segment_order.size(); ++i) {
                other_consec.push_back(_frame_info.global_segment_order.at(i));
            }
        }
        number << "consec: " << consec.start << "-" << consec.end << " (" << consec.end - consec.start << ")";
        
        Color consec_color = White;
        if(consec.contains(_frame_info.frameIndex))
            consec_color = Green;
        
        if(_status_text2.hovered())
            consec_color = consec_color.exposureHSL(0.9f);
        _status_text2.set_color(consec_color);
        
        _status_text2.set_txt(number.str());
        number.str("");
        
        number << "delta: " << _frame_info.tdelta << "s";
        
        //number << ", " << tracker_endframe << ")";
        //number << " tdelta:" << std::fixed << std::setprecision(3) << tdelta << " " << std::fixed << std::setprecision(3) << _frame_info.tdelta_gui;
        number << NetworkStats::status();
        
        auto status = EventAnalysis::status();
        if(!status.empty())
            number << " " << status;
        //number << " midline-err/frame:" << Tracker::instance()->midline_errors_frame();
        
        _title_layout.set_pos(Vec2(20, 25) - offset);
        _title_layout.set_origin(Vec2(0, 0.5));
        _status_text3.set_txt(number.str());
        
        _title_layout.set_policy(HorizontalLayout::Policy::CENTER);
        std::vector<Layout::Ptr> in_layout{&_pause, &_status_text, &_status_text2, &_status_text3};
        if(_frame_info.video_length == tracker_endframe) {
            in_layout.erase(in_layout.begin());
        }
        
        if(GUI_SETTINGS(gui_mode) == mode_t::blobs && _bar) {
            base.rect(-offset, Size2(max_w, bar_height + y), Red.alpha(75));
            in_layout.insert(in_layout.begin(), &_raw_text);
        }
        
        if(GUI_SETTINGS(auto_train)) {
            base.rect(-offset, Size2(max_w, bar_height + y), Red.alpha(75));
            _auto_text.set_txt("[auto_train]");
            in_layout.insert(in_layout.begin(), &_auto_text);
            
        } else if(GUI_SETTINGS(auto_apply)) {
            base.rect(-offset, Size2(max_w, bar_height + y), Red.alpha(75));
            _auto_text.set_txt("[auto_apply]");
            in_layout.insert(in_layout.begin(), &_auto_text);
        } else if(GUI_SETTINGS(auto_quit)) {
            base.rect(-offset, Size2(max_w, bar_height + y), Red.alpha(75));
            _auto_text.set_txt("[auto_quit]");
            in_layout.insert(in_layout.begin(), &_auto_text);
        }
        
        _title_layout.set_children(in_layout);
        base.wrap_object(_title_layout);
        
        pos = Vec2(0, y) - offset;
        
        gui::Color red_bar_clr(250, 250, 250, uchar(GUI_SETTINGS(gui_timeline_alpha) * (_bar && _bar->hovered() ? 1.f : 0.8f)));
        
        base.rect(pos, Vec2(max_w, bar_height), White.alpha(175));
        
        float percent = float(tracker_endframe) / _frame_info.video_length;
        base.rect(pos, Size2(max_w * percent, bar_height), red_bar_clr);
        
        if(_bar && use_scale.y > 0) {
            std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
            float new_height = roundf(bar_height);
            _bar->set_scale(Vec2(1, new_height));
            _bar->set_color(White.alpha(GUI_SETTINGS(gui_timeline_alpha)));
            _bar->set_pos(pos);
            base.wrap_object(*_bar);
            
            if(FAST_SETTINGS(recognition_enable)) {
                update_consecs(max_w, consec, other_consec, 1);
                if(_consecutives) {
                    _consecutives->set_pos(pos - Vec2(0,5));
                    base.wrap_object(*_consecutives);
                }
            }
            
            base.add_object(new Text(Meta::toStr(tracker_endframe.load()), pos + Vec2(max_w * tracker_endframe / float(_frame_info.video_length) + 5, bar_height * 0.5f), Black, Font(0.5), Vec2(1), Vec2(0,0.5)));
            
            // display hover sign with frame number
            if(_mOverFrame != -1) {
                //auto it = _proximity_bar.changed_frames.find(_mOverFrame);
                //if(it != _proximity_bar.changed_frames.end() || _mOverFrame >= _proximity_bar.end)
                {
                    std::string t = "Frame "+std::to_string(_mOverFrame);
                    auto dims = Base::text_dimensions(t, &_title_layout, Font(0.7f));
                    
                    Vec2 pp(max_w / float(_frame_info.video_length) * _mOverFrame, _bar->pos().y + _bar->global_bounds().height / use_scale.y + dims.height * 0.5f + 2);
                    
                    if(pp.x < dims.width * 0.5f)
                        pp.x = dims.width * 0.5f;
                    if(pp.x + dims.width * 0.5f > max_w)
                        pp.x = max_w - dims.width * 0.5f;
                    
                    pp -= offset;
                    
                    base.rect(pp - dims * 0.5f - Vec2(5, 2), dims + Vec2(10, 4), Black.alpha(125));
                    base.text(t, pp, gui::White, Font(0.7f, Align::Center));
                }
            }
        }
        
        if(_frame_info.analysis_range.start != 0) {
            auto start_pos = pos;
            auto end_pos = Vec2(max_w / float(_frame_info.video_length) * _frame_info.analysis_range.start, bar_height);
            base.rect(start_pos, end_pos, Gray);
        }
        if(_frame_info.analysis_range.end < _frame_info.video_length-1) {
            auto start_pos = pos + Vec2(max_w / float(_frame_info.video_length) * _frame_info.analysis_range.end, 0);
            auto end_pos = Vec2(max_w, bar_height);
            base.rect(start_pos, end_pos, Gray);
        }
        
        // current position indicator
        auto current_pos = Vec2(max_w / float(_frame_info.video_length) * _frame_info.frameIndex, y) - offset;
        base.rect(current_pos - Vec2(2),
                  Size2(5,bar_height + 4),
                  Black.alpha(255), White.alpha(255));
        //base.rect(current_pos - Vec2(2, 1), Vec2(5, 2), DarkCyan, Black);
        //base.rect(current_pos + Vec2(-2, bar_height + 3), Vec2(5, 2), DarkCyan, Black);
    }
    
    void Timeline::update_fois() {
        static Timing timing("update_fois", 10);
        TakeTiming take(timing);
        
        const Vec2 use_scale = _bar ? _bar->stage_scale() : Vec2(1);
        auto && [offset, max_w] = timeline_offsets();
        //const float max_h = Tracker::average().rows;
        
        uint64_t last_change = FOI::last_change();
        auto name = SETTING(gui_foi_name).value<std::string>();
        
        if(last_change != _foi_state.last_change || name != _foi_state.name) {
            _foi_state.name = name;
            
            if(!_foi_state.name.empty()) {
                long_t id = FOI::to_id(_foi_state.name);
                if(id != -1) {
                    _foi_state.changed_frames = FOI::foi(id);//_tracker.changed_frames();
                    _foi_state.color = FOI::color(_foi_state.name);
                }
            }
            
            _foi_state.last_change = last_change;
        }
        
        //if(_proximity_bar.image.rows && _proximity_bar.image.cols) {
        if(_bar == NULL) {
            _bar = std::make_unique<ExternalImage>(Image::Make(), pos);
            _bar->set_color(White.alpha(GUI_SETTINGS(gui_timeline_alpha)));
            _bar->set_clickable(true);
            _bar->on_hover([this](Event e) {
                if(!GUI::instance())
                    return;
                
                auto && [offset_, max_w_] = timeline_offsets();
                
                //if(!_proximity_bar.changed_frames.empty())
                {
                    float distance2frame = FLT_MAX;
                    long_t framemOver = -1;
                    
                    if(_bar && _bar->hovered()) {
                        //Vec2 pp(max_w / float(_frame_info.video_length) * idx.first, 50);
                        //float dis = abs(e.hover.x - pp.x);
                        static Timing timing("Scrubbing", 0.01);
                        int64_t idx = roundf(e.hover.x / max_w_ * float(_frame_info.video_length));
                        auto it = _proximity_bar.changed_frames.find(idx);
                        if(it != _proximity_bar.changed_frames.end()) {
                            framemOver = idx;
                            distance2frame = 0;
                        } else if((it = _proximity_bar.changed_frames.find(idx - 1)) != _proximity_bar.changed_frames.end()) {
                            framemOver = idx - 1;
                            distance2frame = 1;
                        } else if((it = _proximity_bar.changed_frames.find(idx + 1)) != _proximity_bar.changed_frames.end()) {
                            framemOver = idx + 1;
                            distance2frame = 1;
                        }
                    }
                    
                    if (distance2frame < 2) {
                        _mOverFrame = framemOver;
                        
                    } else if(_bar->hovered()) {
                        if(tracker_endframe != -1) {
                            _mOverFrame = min(_frame_info.video_length, e.hover.x * float(_frame_info.video_length) / max_w_);
                            _bar->set_dirty();
                        }
                    }
                    else
                        _mOverFrame = -1;
                }
                
                if(_bar->hovered() && _bar->pressed() && this->mOverFrame() != -1)
                {
                    SETTING(gui_frame) = this->mOverFrame();
                }
            });
            _bar->add_event_handler(MBUTTON, [this](Event e) {
                if(e.mbutton.pressed && this->mOverFrame() != -1 && e.mbutton.button == 0) {
                    _gui.set_redraw();
                    SETTING(gui_frame) = this->mOverFrame();
                }
            });
        }
        
        float new_height = roundf(bar_height / use_scale.y);
        //_bar->set_scale(Vec2(1, new_height));
        
        /*if(_proximity_bar._image) {
            _bar->set_source(std::move(_proximity_bar._image));
            //_proximity_bar.changed = false;
        }*/
        
        _bar->set_pos(pos);
        
        bool changed = false;
        
        if(use_scale.y > 0) {
            std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
            static std::string last_name;
            
            // update proximity bar, whenever the FOI to display changed
            if(last_name != _foi_state.name)
                _proximity_bar.end = -1;
            last_name = _foi_state.name;
            
            if(!_bar
               || (uint)max_w != _bar->source()->cols
               || (uint)1 != _bar->source()->rows
               || _proximity_bar.end == -1)
            {
                auto image = Image::Make(1, max_w, 4);
                image->set_to(0);
                _bar->set_source(std::move(image));
                
                _proximity_bar.end = _proximity_bar.start = -1;
                changed = true;
                _proximity_bar.changed_frames.clear();
            }
        }
        
        if(tracker_endframe != -1 && _proximity_bar.end < tracker_endframe) {
            std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
            auto individual_coverage = [](long_t frame) {
                float count = 0;
                if(Tracker::properties(frame)) {
                    for(auto fish : Tracker::instance()->active_individuals(frame)) {
                        if(fish->centroid(frame))
                            count++;
                    }
                }
                return (1 - count / float(Tracker::instance()->individuals().size()));
            };
            
            if(_proximity_bar.end == -1) {
                _proximity_bar.start = _proximity_bar.end = _tracker.start_frame();
            }
            
            Vec2 pos(max_w / float(_frame_info.video_length) * _proximity_bar.end, 0);
            cv::Mat img = _bar->source()->get();
            
            if(_proximity_bar.end < _tracker.end_frame()) {
                _proximity_bar.end = _tracker.end_frame();
                changed = true;
            }
            
            Vec2 previous_point(-1, 0);
            _proximity_bar.samples_per_pixel.resize(max_w);
            for(auto & px : _proximity_bar.samples_per_pixel)
                px = 0;
            assert(max_w > 0);
            std::set<uint32_t> multiple_assignments;
            
            for (auto &c : _foi_state.changed_frames) {
                float x = round(max_w / float(_frame_info.video_length) * c.frames().start);
                if(x >= _proximity_bar.samples_per_pixel.size())
                    x = _proximity_bar.samples_per_pixel.size()-1;
                ++_proximity_bar.samples_per_pixel[uint32_t(x)];
                if(_proximity_bar.samples_per_pixel[uint32_t(x)] > 1)
                    multiple_assignments.insert(uint32_t(x));
                //auto it = _proximity_bar.changed_frames.find(c.frames().start);
                //if(it == _proximity_bar.changed_frames.end()) {
                    /*float x = max_w / float(_frame_info.video_length) * c.frames().start;
                    float w = max(2, max_w / float(_frame_info.video_length) * c.frames().end - x);
                    Vec2 pp(x, 0);
                    cv::rectangle(img, pp, pp+Vec2(w,img.rows), _foi_state.color, -1);*/
                
                    //if(!_proximity_bar.changed) {
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
                    //Debug("Too many multi-assignments (%d / %d). Normalizing colors.", multiple_assignments.size(), _foi_state.changed_frames.size());
                    uint32_t Nx = 0;
                    for(auto x : multiple_assignments) {
                        auto n = _proximity_bar.samples_per_pixel[x];
                        if(n > N) {
                            N = n;
                            Nx = x;
                        }
                    }
                }
                
                for(size_t i=0; i<_proximity_bar.samples_per_pixel.size(); ++i) {
                    px = _proximity_bar.samples_per_pixel[i];
                    if(px > 0) {
                        Vec2 pp(i, 0);
                        float d = float(px) / float(N);
                        img(Bounds(pp, Size2(1, img.rows))) = (cv::Scalar)_foi_state.color.alpha(50 + 205 * min(1, SQR(d)));
                    }
                    //cv::rectangle(img, pp, pp+Vec2(1,img.rows), _foi_state.color, -1);
                }
                
                float x = max_w / float(_frame_info.video_length) * tracker_endframe;
                if(previous_point.x != -1 && previous_point.x != x)
                {
                    Vec2 point(x, individual_coverage(tracker_endframe) * img.rows);
                    DEBUG_CV(cv::line(img, previous_point, Vec2(x-1, previous_point.y), Red));
                    DEBUG_CV(cv::line(img, Vec2(x-1, previous_point.y), point, Red));
                }
            }
            
            /*if(!_bar || !_bar->source()->operator==(_proximity_bar.image)) {
                _proximity_bar.changed = true;
            }*/
        }
    }
    
    void Timeline::update_thread() {
        _terminate = false;
        
        _foi_state.color = Color(255, 200, 100, 255);
        _foi_state.last_change = 0;
        _foi_state.name = "";
        
        auto long_wait_time = std::chrono::seconds(1);
        auto short_wait_time = std::chrono::milliseconds(5);
        
        while(!_terminate) {
            bool changed = false;
            
            //! Update the cached data
            if(GUI::instance() && Tracker::instance()) {
                if(!GUI::instance())
                    break;
                
                std::lock_guard<std::recursive_mutex> lock(GUI::instance()->gui().lock());
                
                Tracker::LockGuard guard("Timeline::update_thread", 100);
                
                if(guard.locked()) {
                    Timer timer;
                    
                    long_t index = _frame_info.frameIndex;
                    auto props = Tracker::properties(index);
                    auto prev_props = Tracker::properties(index - 1);
                    
                    _frame_info.tdelta = props && prev_props ? props->time - prev_props->time : 0;
                    _frame_info.small_count = 0;
                    _frame_info.big_count = 0;
                    _frame_info.current_count = 0;
                    _frame_info.up_to_this_frame = 0;
                    _frame_info.analysis_range = Tracker::analysis_range();
                    tracker_endframe = _tracker.end_frame();
                    tracker_startframe = _tracker.start_frame();
                    
                    if(prev_props && props) {
                        tdelta = _frame_info.tdelta;
                    }
                    
                    _frame_info.training_ranges = _tracker.recognition() ? _tracker.recognition()->trained_ranges() : std::set<Rangel>{};
                    _frame_info.consecutive = _tracker.consecutive();
                    _frame_info.global_segment_order = track::Tracker::global_segment_order();
                    
                    //if(longest != _frame_info.longest_consecutive && Tracker::recognition())
                    //if(Tracker::recognition()) {
                        /*for(auto &consec : _frame_info.global_segment_order) {
                            if(!Tracker::recognition()->dataset_quality() || !Tracker::recognition()->dataset_quality()->has(consec)) {
                                Tracker::recognition()->update_dataset_quality();
                                break;
                            }
                        }*/
                        //Tracker::recognition()->update_dataset_quality();
                    //}
                    
                    if(Tracker::properties(_frame_info.frameIndex)) {
                        for (auto& fish : _frame_info.frameIndex >= tracker_startframe && _frame_info.frameIndex < tracker_endframe ?  Tracker::active_individuals(_frame_info.frameIndex) : Tracker::set_of_individuals_t{}) {
                            if ((int)fish->frame_count() < FAST_SETTINGS(frame_rate)*3) {
                                _frame_info.small_count++;
                            } else
                                _frame_info.big_count++;
                            
                            if(fish->has(_frame_info.frameIndex))
                                ++_frame_info.current_count;
                            
                            if(fish->start_frame() <= _frame_info.frameIndex) {
                                _frame_info.up_to_this_frame++;
                            }
                        }
                    }
                    
                    //if(FAST_SETTINGS(calculate_posture))
                    //    changed = EventAnalysis::update_events(_frame_info.frameIndex < tracker_endframe ? Tracker::active_individuals(_frame_info.frameIndex) : std::set<Individual*>{});
                    
                    // needs Tracker lock
                    GUI::instance()->update_recognition_rect();
                    update_fois();
                    
                    _update_thread_updated_once = true;
                    
                    if(timer.elapsed() > 0.1 && !FAST_SETTINGS(analysis_paused)) {
                        if(long_wait_time < std::chrono::seconds(30)) {
                            long_wait_time = std::chrono::seconds(30);
                            short_wait_time = std::chrono::seconds(30);
                            
                            if(!FAST_SETTINGS(analysis_paused))
                                Warning("Throtteling some non-essential gui functions until analysis is over.");
                        }
                        
                    } else {
                        long_wait_time = std::chrono::seconds(1);
                        short_wait_time = std::chrono::milliseconds(5);
                    }
                }
            }
            
            if(!changed)
                std::this_thread::sleep_for(long_wait_time);
            else
                std::this_thread::sleep_for(short_wait_time);
        }
    }
    
    void Timeline::reset_events(long_t after_frame) {
        EventAnalysis::reset_events(after_frame);
        
        {
            std::lock_guard<std::mutex> guard(_proximity_bar.mutex);
            if(after_frame == -1) {
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
                    float x0 = Tracker::average().cols / float(_frame_info.video_length) * after_frame;
                    float x1 = Tracker::average().cols;
                    
                    Debug("Clearing from %f to %f", x0, x1 + pos.x);
                    DEBUG_CV(cv::rectangle(img, Vec2(x0, 0), Vec2(pos + Vec2(x1, img.rows)), Transparent, -1));
                }
            }
            
            _proximity_bar.end = after_frame == -1 ? -1 : (after_frame - 1);
            //_proximity_bar.changed = true;
            
            tracker_endframe = _proximity_bar.end;
        }
    }
    
    void Timeline::set_visible(bool v) {
        if(_visible != v) {
            _gui.cache().set_tracking_dirty();
            _gui.set_redraw();
            _visible = v;
        }
    }
    
    void Timeline::next_poi(Idx_t _s_fdx) {
        auto frame = GUI::frame();
        long_t next_frame = frame;
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
                auto &cache = GUI::instance()->cache();
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
        auto frame = GUI::frame();
        long_t next_frame = frame;
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
        
        if(frame != next_frame && next_frame != -1) {
            SETTING(gui_frame) = next_frame;
            
            if(!_s_fdx.valid())
            {
                auto &cache = GUI::instance()->cache();
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
