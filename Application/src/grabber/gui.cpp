#include "gui.h"
#include <misc/GlobalSettings.h>
#include <gui/DrawSFBase.h>
#include <gui/colors.h>
#include <gui/DrawHTMLBase.h>
#include <tracking/Tracker.h>
#include <tracker/gui/DrawFish.h>
#include <gui/IMGUIBase.h>
#include <tracking/IndividualManager.h>

namespace grab {

#define KEY(NAME) setting_keys.at( # NAME )
#define VALUE(NAME) GlobalSettings::get( KEY(NAME) )

IMPLEMENT(GUI::setting_keys) = {
    { "mode", "gui_mode" },
    { "terminate", "terminate" },
    { "run", "gui_run" }
};

GUI *_instance = nullptr;
CallbackCollection callback;

GUI* GUI::instance() {
    return _instance;
}

GUI::GUI(DrawStructure* graph, FrameGrabber& grabber)
: _grabber(grabber),
    _crop_offsets(SETTING(crop_offsets).value<CropOffsets>()),
    _size(grabber.cam_size().width, grabber.cam_size().height),
    _cropped_size(grabber.cropped_size()),
    //_window_scale(min((sf::VideoMode::getDesktopMode().height - 250 > _cropped_size.height * 1.8f ? 1.8f : ((sf::VideoMode::getDesktopMode().height - 250) / float(_cropped_size.height))),
    //               (sf::VideoMode::getDesktopMode().width - 250 > _cropped_size.width * 1.8f ? 1.8f : ((sf::VideoMode::getDesktopMode().width - 250) / float(_cropped_size.width))))),
    _redraw(false),
    _record_alpha(0.f),
    _record_direction(true),
    _pulse_direction(false),
    _pulse(0),
    _sf_base(NULL),
    _gui(graph)
{
    _gui->set_size(Size2(max(150, _cropped_size.width), max(150, _cropped_size.height)));
    _instance = this;
    
    callback = GlobalSettings::map().register_callbacks({"mode"}, [this](std::string_view name){
        set_redraw();
    });
}

void GUI::set_base(gui::Base *base) {
    _sf_base = base;
    
    if(base) {
        SETTING(gui_interface_scale) = float(1);
        _crop_offsets = (SETTING(crop_offsets).value<CropOffsets>());
        _size = cv::Size(_grabber.cam_size().width, _grabber.cam_size().height);
        _cropped_size = (_grabber.cropped_size());
        gui().set_size(Size2(max(150, _cropped_size.width), max(150, _cropped_size.height)));
        if(base && dynamic_cast<gui::IMGUIBase*>(base))
            ((gui::IMGUIBase*)base)->init(base->title(), true);
        
        auto desktop_mode = base->window_dimensions();
        gui::Event e(gui::EventType::WINDOW_RESIZED);
        e.size.width = desktop_mode.width;
        e.size.height = desktop_mode.height;
        event(e);
    }
}

GUI::~GUI() {
    if(callback)
        GlobalSettings::map().unregister_callbacks(std::move(callback));
}

#if WITH_MHD
Httpd::Response GUI::render_html() {
    {
        std::lock_guard<std::mutex> lock(_gui_frame_lock);
        if(_gui_bytes_timer.elapsed() >= 1) {
            _gui_bytes_per_second = _gui_bytes_count;
            _gui_bytes_count = 0;
            _gui_bytes_timer.reset();
        }
        
        if(_gui_timer.elapsed() < SETTING(web_time_threshold).value<float>())
            return Httpd::Response({}, "text/html");
    }
    
    //cv::Size size = SETTING(cam_resolution);
    
    //this->draw(base);
    _html_base.paint(_gui);
    const auto& tmp = _html_base.to_bytes();
    
    std::lock_guard<std::mutex> lock(_gui_frame_lock);
    bool not_equal = true;//tmp.size() != bytes.size();
    /*if(!not_equal) {
     if(memcmp(tmp.data(), bytes.data(), tmp.size())) {
     not_equal = true;
     }
     }*/
    
    if(/*_gui_frame == frame() ||*/ !not_equal && _gui_timer.elapsed() < SETTING(web_time_threshold).value<float>()*5) {
        return Httpd::Response({}, "text/html");
    }
    
    _gui_bytes_count += tmp.size();
    _gui_timer.reset();
    
    return Httpd::Response(tmp, "text/html");
}
#endif

void GUI::run_loop() {
    while (!terminated()) {
        update_loop();
    }
    
    print("GUI thread ended.");
}

void GUI::update_loop() {
    static Timer timer;
    if(_redraw || timer.elapsed() >= 1.f/30.f) {
        timer.reset();
        
        update();
        
        {
            auto guard = GUI_LOCK(gui().lock());
            draw(gui());
        }
        //_gui.print(&_sf_base);
        
        if(_sf_base)
            _sf_base->paint(gui());
        else
            gui().before_paint(_sf_base);
        
        {
            std::lock_guard<std::mutex> guard(_display_queue_lock);
            _redraw = false;
        }
        
    } else {
        const std::chrono::milliseconds ms(75);
        std::this_thread::sleep_for(ms);
    }
}

void GUI::update() {
    float e = pulse_timer.elapsed();
    float pulse_time = 5;
    if (e >= pulse_time) {
        pulse_timer.reset();
        _pulse_direction = !_pulse_direction;
        
        e = min(pulse_time, e);
        
        if(_pulse_direction)
            e = pulse_time - e;
        else
            e = pulse_time - e;
    }
    
    e /= pulse_time;
    _pulse = (_pulse_direction ? 0 : 1) + (_pulse_direction ? 1 : -1) * e / 1;
}

void GUI::draw(gui::DrawStructure &base) {
    using namespace gui;
    static Timer last_frame_time;
    
    {
        gui::DrawStructure::SectionGuard guard(base, "draw()");
        if (_sf_base) {
            Size2 size(gui().width(), gui().height());
            float scale = min(size.width / float(_cropped_size.width),
                size.height / float(_cropped_size.height));
            guard._section->set_scale(Vec2(scale).div(base.scale()));//.div();
        }

        float tdelta = last_frame_time.elapsed();
        last_frame_time.reset();

        //static cv::Rect2f raw_offsets = SETTING(crop_offsets);
        static Vec2 offset(0, 0);//(_size.width * (raw_offsets.x + raw_offsets.width) / 2,
                          // _size.height * (raw_offsets.y + raw_offsets.height) / 2);

        std::unique_ptr<pv::Frame> last_noise;
        
        {
            auto frame = _grabber.last_frame();
            auto image = _grabber.latest_image();
            auto noise = _grabber.noise();

#ifndef NDEBUG
            static long last_index = -1;
            if (image && image->index() < last_index)
                FormatWarning("Last index = ", last_index," and current = ",image->index());
#endif

            if (frame)
                _frame = std::move(frame);
            if (image)
                _image = std::move(image);
            if (noise) {
                last_noise = std::move(_noise);
                _noise = std::move(noise);
            }
        }

        Scale scale{guard._section->scale().mul(base.scale()).reciprocal()};

        Color text_color(255, 255, 255, 255);
        if (_image && _image->cols > 20 && _image->rows > 20) {
            cv::Mat tmp = _image->get();
            double val = 0;
            size_t samples = 0;
            for (int i = 0; i < 100 * scale.x; i += 10 * scale.x) {
                for (int j = 0; j < 20 * scale.y; j += 4 * scale.y) {
                    val += tmp.at<uchar>(j, i);
                    samples++;
                }
            }
            val /= samples;

            if (val < 150) {
                text_color = White;
            }
            else {
                text_color = Black;
            }
        }

        if (_grabber.average_finished()) {
            if (_frame && _image) {
                //float scale = SETTING(web_quality).value<int>() / 100.f;
                float scale = 1;

                if (!background || background->source()->cols != uint(_image->cols) || background->source()->rows != uint(_image->rows))
                {
                    if (background) {
                        background->set_source(std::move(_image));
                        background->set_pos(offset);
                        background->set_scale(Vec2(1 / scale));

                        if (noise_image && background->source()) {
                            noise_image->set_source(Image::Make(background->source()->rows, background->source()->cols, 4));
                            noise_image->set_pos(offset);
                            noise_image->set_scale(Vec2(1 / scale));
                        }
                    }
                    else {
                        noise_image = new ExternalImage(Image::Make(_image->rows, _image->cols, 4), offset, Vec2(1 / scale));
                        background = new ExternalImage(std::move(_image), offset, Vec2(1 / scale));
                        print("Creating images.");
                    }
                }
                else {
                    background->set_scale(Vec2(1 / scale));
                    background->set_source(std::move(_image));
                }

                if (noise_image && !noise_image->empty() && _noise) {
                    auto mat = noise_image->source()->get();
                    
                    if(last_noise) {
                        // reverse last image
                        auto N = last_noise->n();
                        for (size_t i = 0; i < N; i++) {
                            auto& m = last_noise->mask().at(i);

                            for (auto& line : *m) {
                                for (ushort x = line.x0; x <= line.x1; x++) {
                                    mat.at<cv::Vec4b>(line.y, x) = cv::Vec4b(0, 0, 0, 0);
                                }
                            }
                        }
                        
                    } else
                        std::fill(noise_image->source()->data(), noise_image->source()->data() + noise_image->source()->size(), 0);

                    auto N = _noise ? _noise->n() : 0u;
                    for (size_t i = 0; i < N; i++) {
                        auto& m = _noise->mask().at(i);

                        for (auto& line : *m) {
                            for (ushort x = line.x0; x <= line.x1; x++) {
                                mat.at<cv::Vec4b>(line.y, x) = cv::Vec4b(255, 0, 255, 255);
                            }
                        }
                    }
                    noise_image->set_dirty();
                }
            }

            if (background)
                base.wrap_object(*background); 
            if (noise_image && _noise)
                base.wrap_object(*noise_image);
            //base.image(offset, convert, 1/scale);

            if (_frame) {
                gui::DrawStructure::SectionGuard guard(base, "blobs");
                ColorWheel wheel;
                static cv::Mat output;
                static Background bg(Image::Make(_grabber.average()), nullptr);
                for (size_t i = 0; i < _frame->mask().size(); i++) {
                    auto& m = _frame->mask().at(i);
                    if (m->empty())
                        continue;

                    pv::Blob blob(*m, *_frame->pixels().at(i), _frame->flags().at(i), _frame->predictions().empty() ? blob::Prediction{} : blob::Prediction{_frame->predictions().at(i)});
                    auto pos = blob.bounds().pos();
                    auto clr = wheel.next();
                    base.rect(Box(pos + offset, blob.bounds().size()), FillClr{Transparent}, LineClr{clr.alpha(150)});
                    
                    //! only display images if there arent too many of them.
                    if(_frame->mask().size() < 100) {
                        auto&& [_, image] = blob.alpha_image(bg, 0);
                        /*auto clr = wheel.next().alpha((_pulse * 0.6 + 0.2) * 255);
                        cv::cvtColor(output, output, cv::COLOR_GRAY2RGBA);

                        for (cv::Mat4b::iterator it = output.begin<cv::Vec4b>(); it != output.end<cv::Vec4b>(); it++)
                        {
                            if((*it)[0] || (*it)[1] || (*it)[2]) {
                                (*it)[0] = clr.r;
                                (*it)[1] = clr.g;
                                (*it)[2] = clr.b;
                                (*it)[3] = 255;
                            } else
                                (*it)[3] = 0;
                        }*/

                        base.image(pos + offset, std::move(image), Vec2(1.0), clr.alpha(150));
                    }
                    
                    base.text(Str(Meta::toStr(i)), Loc(pos + offset), TextClr(Yellow), Font(0.5), scale);
                }
            }

            if (!_grabber.is_recording()) {
                base.text(Str("waiting for commands"), Loc(_size.width / 2, _size.height / 2), TextClr(Red), Font(0.8, Align::Center), scale);
                base.rect(Box(Vec2(8, 14), Size2(7, 7)), FillClr{White.alpha(125)}, LineClr{Black.alpha(125)});
            }
            else {
                const float speed = 0.5;
                if (_record_direction) {
                    _record_alpha += speed * tdelta;
                }
                else {
                    _record_alpha -= speed * tdelta;
                }

                if (_record_alpha >= 1) {
                    _record_alpha = 1;
                    _record_direction = !_record_direction;
                }
                else if (_record_alpha <= 0) {
                    _record_alpha = 0;
                    _record_direction = !_record_direction;
                }

                float alpha = min(0.8f, max(0.25f, _record_alpha));
                if (_grabber.is_paused()) {
                    base.rect(Box(Vec2(8, 14).mul(scale), Vec2(2, 7).mul(scale)), FillClr{White.alpha(alpha * 255)}, LineClr{Black.alpha(alpha * 255)});
                    base.rect(Box(Vec2(12, 14).mul(scale), Vec2(2, 7).mul(scale)), FillClr{White.alpha(255 * alpha)}, LineClr{Black.alpha(255 * alpha)});

                }
                else {
                    base.circle(Loc(Vec2(13, 18).mul(scale)), Radius{5}, LineClr{White.alpha(255 * alpha)}, FillClr{text_color.alpha(255 * alpha)}, scale, Origin(0.5));
                }
            }

        }
        else {
            if (!_grabber.average_finished()) {
                if (_image) {
                    cv::Mat mat = _image->get();

                    float scale = 1;

                    static cv::Mat convert;
                    cv::cvtColor(mat, convert, cv::COLOR_GRAY2RGBA);

                    if (!background) {
                        background = new ExternalImage(Image::Make(convert), offset, Vec2(1 / scale));
                    }
                    else {
                        if (background->source()->rows != (uint)convert.rows || background->source()->cols != (uint)convert.cols) {
                            background->set_source(Image::Make(convert));
                        }
                        else
                            background->set_source(Image::Make(convert));
                        background->set_dirty();
                    }

                    base.wrap_object(*background);
                }

                base.text(Str{
                            "generating average (" + std::to_string(_grabber.average_samples()) + "/" + std::to_string(SETTING(average_samples).value<uint32_t>()) + ")"
                          },
                          Loc(_size.width / 2, _size.height / 2),
                          TextClr{Red},
                          Font(0.8f, Align::Center),
                          scale);
            }
            else {
                base.text(Str{"waiting for frame..."}, Loc(_size.width / 2, _size.height / 2), TextClr{Red}, Font(0.8f, Align::Center), scale);
            }
        }

        {
            auto shadowed_text = [&](Vec2 pos, const std::string& text, Color color, float font_size = 0.75, bool shadow = true)
            {
                // shadow
                if(shadow)
                    base.text(Str{text}, Loc((pos + Vec2(0.5, 0.5)).mul(scale)), TextClr{Black}, Font(font_size, Align::VerticalCenter), scale);
                // text
                return base.text(Str{text}, Loc(pos.mul(scale)), TextClr{color}, Font(font_size, Align::VerticalCenter), scale)->width();
            };

            auto frame = _grabber.last_index().load();

            Vec2 offset(25, 17);

            offset.x += (shadowed_text(offset, "frame", text_color)) + 5;
            offset.x += (shadowed_text(offset,std::to_string(frame), Cyan)) + 5;
            //offset.x = offset.x + 15 - int(offset.x) % 15;
            if (_grabber.video()) {
                static Vec2 previous = offset;
                static Timer timer;

                auto diff = offset - previous;
                if (diff.x > 0)
                    previous = offset;
                else
                    previous += diff * 0.1 * timer.elapsed() * 1;

                offset = previous;
                timer.reset();

                offset.x += (shadowed_text(offset, "/", text_color)) + 5;
                offset.x += (shadowed_text(offset, _grabber.video()->length().toStr(), text_color)) + 5;
                //offset.x = offset.x + 5 - int(offset.x) % 5;
            }

            offset.x += base.line((offset + Vec2(0, 0.5)).mul(scale), (offset + Vec2(10, 0.5)).mul(scale), Gray, scale)->width() + 5;
            offset.x += (shadowed_text(offset, dec<2>(_grabber.fps().load()).toStr(), Cyan)) + 2;
            offset.x += shadowed_text(offset, "fps", text_color);

            offset.x = 25;
            offset.y += 22;

            std::vector<std::string> values;
            if (SETTING(enable_live_tracking)) values.push_back("tracking " +std::to_string(_grabber.tracker_current_individuals().load()));
            if (SETTING(enable_closed_loop))   values.push_back("closed-loop");
            if (SETTING(correct_luminance))    values.push_back("normalizing luminance");
            values.push_back("threshold: " + std::to_string(SETTING(threshold).value<int>()));
            if (SETTING(tags_enable)) values.push_back("tags");
            values.push_back("<nl>");
            
            {
                std::unique_lock fps_lock(_grabber._fps_lock);
                if(_grabber.loading.timestamp.valid()) {
                    values.push_back("load "+_grabber.loading.toStr());
                    values.push_back("proc "+_grabber.processing.toStr());
                    values.push_back("track "+_grabber.tracking.toStr());
                }
            }

            bool darker = false;
            for (size_t i = 0; i < values.size(); ++i) {
                if(values[i] == "<nl>") {
                    offset.x = 25;
                    offset.y += 18;
                    darker = false;
                    continue;
                    
                } else if(offset.x > 25) {
                    offset.x += base.line((offset + Vec2(0, 0.5)).mul(scale), (offset + Vec2(5, 0.5)).mul(scale), Gray, scale)->width() + 5;
                }
                
                offset.x += shadowed_text(offset, values[i], darker ? (text_color.r < 100 ? Color(70,70,70,255) : text_color.exposure(0.8)) :text_color, 0.5, false) + 5;
                //if(i + 1 < values.size())
                //    offset.x += base.line((offset + Vec2(0, 0.5)).mul(scale), (offset + Vec2(5, 0.5)).mul(scale), Gray, scale)->width() + 5;
                darker = !darker;
            }
        }

        draw_tracking(base, scale);
    }

    auto scale = base.scale().reciprocal();
    auto dim = _sf_base ? _sf_base->window_dimensions().mul(scale * gui::interface_scale()) : Size2(_grabber.average());
    base.draw_log_messages(Bounds(Vec2(0, 85).mul(scale* gui::interface_scale()), dim - Size2(10, 85).mul(scale * gui::interface_scale())));
}

void GUI::draw_tracking(gui::DrawStructure &base, const attr::Scale& scale) {
    using namespace gui;
    
    if(!_grabber.tracker_instance())
        return;
    
    base.section("tracking", [this, &scale](gui::DrawStructure& base, Section* section) {
        track::LockGuard guard(ro_t{}, "drawing", 100);
        if (!guard.locked()) {
            section->reuse_objects();
            return;
        }

        using namespace track;
        static const auto gui_outline_thickness = SETTING(gui_outline_thickness).value<uint8_t>();
        
        auto tracker = _grabber.tracker_instance();
        auto result = IndividualManager::active_individuals(Tracker::end_frame());
        set_of_individuals_t individuals;
        if(result) {
            individuals = *result.value();
        }

#if !COMMONS_NO_PYTHON
        ska::bytell_hash_map<int64_t, std::tuple<float, float>> speeds;
        const auto displayed_range = Frame_t(FAST_SETTING(frame_rate) * 5u);

        const Frame_t min_display_frame = max(0_f, (Tracker::end_frame().valid()
                                                    ? Tracker::end_frame()
                                                    : 0_f)
                                              .try_sub(displayed_range));
#endif
        static const auto tags_recognize = SETTING(tags_recognize).value<bool>();
        static const auto gui_show_midline = SETTING(gui_show_midline).value<bool>();
        
        std::vector<Vertex> oline;
        std::vector<Vec2> positions;
        
        for (auto& fish : individuals) {
            if (fish->end_frame() < min_display_frame)
                continue;
            
            auto it = fish->iterator_for(min_display_frame);//search);
            for (; it != fish->frame_segments().end(); ++it) {
                const auto &seg = *it;
                //if(seg->end() < search)
                if(seg->end() < min_display_frame)
                    continue;
                
                positions.clear();
                positions.reserve(min((size_t)displayed_range.get(), seg->basic_index.size()));
                
                const auto code =
                    tags_recognize
                      ? fish->qrcode_at((*it)->start())
                      : Individual::IDaverage{.best_id = -1, .p = -1, .samples = 0};
                
                decltype(speeds)::mapped_type* speeds_ptr = nullptr;
                Color color;
                
                if (code.best_id != -1) {
                    speeds_ptr = &speeds[code.best_id];
                    color = ColorWheel(uint32_t(code.best_id)).next();
                } else {
                    color = fish->identity().color();
                }
                
                //! only draw the lines and collect the data if we either:
                //!     - do not recognize tags anyway (so its just the normal tracking view)
                //!     - or we have a qrcode detected in here and want to display it
                if(tags_recognize && speeds_ptr == nullptr)
                    continue;
                
                size_t idx = seg->basic_index.front();
                if(seg->start() < min_display_frame) {
                    assert(seg->end() >= min_display_frame);
                    idx = seg->basic_stuff(min_display_frame);
                }
                
                auto bit = fish->basic_stuff().begin() + idx;
                auto end = fish->basic_stuff().begin() + seg->basic_index.back();
                
                for(; bit != end; ++bit) {
                    const auto &basic = *bit;//fish->basic_stuff()[idx];
                    const auto &frame = basic->frame;
                    
                    //auto bounds = basic->blob.calculate_bounds();
                    
                    /*if(speeds_ptr) {
                        std::get<0>(*speeds_ptr) += basic->centroid.speed<Units::CM_AND_SECONDS>();
                        std::get<1>(*speeds_ptr)++;
                    }*/

                    //! only go further into the drawing / acquisition code
                    //! if we actually want to draw it
                    if(frame < min_display_frame)
                        continue;
                    
                    //auto p = bounds.pos() + bounds.size() * 0.5;
                    Loc p{basic->centroid.pos<Units::PX_AND_SECONDS>()};
                    positions.push_back(p);

                    //! if this is the last frame, also add the outline to the drawing
                    if (frame == fish->end_frame()) {
                        auto bounds = basic->blob.calculate_bounds();
                        base.circle(p, Radius{10}, LineClr{fish->identity().color()});
                        
                        auto posture_index = seg->posture_stuff(frame);
                        if (posture_index != -1) {
                            auto &posture = fish->posture_stuff()[posture_index];
                            auto &_cached_outline = posture->outline;
                            auto &_cached_midline = posture->cached_pp_midline;
                            auto &clr = fish->identity().color();
                            auto max_color = 255;
                            auto points = _cached_outline->uncompress();

                            // check if we actually have a tail index
                            if (gui_show_midline && _cached_midline && _cached_midline->tail_index() != -1) {
                                base.circle(Loc(points.at(_cached_midline->tail_index()) + bounds.pos()), Radius{5}, LineClr{Blue.alpha(max_color * 0.3)});
                                if (_cached_midline->head_index() != -1)
                                    base.circle(Loc(points.at(_cached_midline->head_index()) + bounds.pos()), Radius{5}, LineClr{Red.alpha(max_color * 0.3)});
                            }

                            //float right_side = outline->tail_index() + 1;
                            //float left_side = points.size() - outline->tail_index();
                            oline.clear();
                            oline.reserve(points.size());
                            
                            for (size_t i = 0; i < points.size(); i++) {
                                auto pt = points[i] + bounds.pos();
                                Color c = clr.alpha(max_color);
                                /*if(outline->tail_index() != -1) {
                                    float d = cmn::abs(float(i) - float(outline->tail_index())) / ((long_t)i > outline->tail_index() ? left_side : right_side) * 0.4 + 0.5;
                                    c = Color(clr.r, clr.g, clr.b, max_color * d);
                                }*/
                                oline.push_back(Vertex(pt, c));
                            }
                            oline.push_back(Vertex(points.front() + bounds.pos(), clr.alpha(0.04 * max_color)));
                            //auto line =
                            base.add_object(new Line(oline, gui_outline_thickness));
                        }
                    }
                }
                
                //! nothing to draw? continue
                if(positions.empty())
                    continue;
                
                const auto is_end = seg->contains(Frame_t(_frame->index()));
                float percent = saturate((float(_frame->index().get()) - float(seg->end().get())) / float(displayed_range.get()), 0.f, 1.f);
                auto alpha = saturate(200.f * (1 - percent), 0, 255);
                
                base.line(positions, 1, color.alpha(alpha));
                base.text(Str{Meta::toStr(code.best_id) + " (" + dec<2>(code.p).toStr() + ")"}, Loc(positions.back() + Vec2(10, 0)), TextClr{color.alpha(alpha)}, is_end ? Font(0.5, Style::Bold) : Font(0.5), scale);
            }
        }
        
        Loc pos(_grabber.average().cols + 100, 120);
        for (auto& [k, tup] : speeds) {
            //auto w =
            base.text(Str{Meta::toStr(k) + ":"}, pos, TextClr(150, 150, 150, 255), Font(0.5, Style::Bold), scale);//->local_bounds().width;
            base.text(Str{Meta::toStr(std::get<0>(tup) / std::get<1>(tup)) + "cm/s"}, Loc(pos + Vec2(70, 0)), TextClr{White}, Font(0.5), scale);
            pos += Vec2(0, 50);
        }
        
        //----
        /*for (auto& fish : individuals) {
            if (fish->has(tracker->end_frame())) {
                std::vector<std::vector<Vec2>> positions{{}};
                std::vector<std::tuple<Vec2, int64_t, float>> tags;
                Frame_t prev;

                fish->iterate_frames(Range<Frame_t>(tracker->end_frame() - 100_f, tracker->end_frame()),
                    [&](Frame_t frame,
                        const std::shared_ptr<SegmentInformation>& ,
                        const BasicStuff* basic,
                        const PostureStuff* posture)
                    -> bool
                {
                    if (basic) {
                        auto bounds = basic->blob.calculate_bounds();

                        if (prev.valid() && prev != frame - 1_f) {
                            positions.push_back({});
                        }
                        prev = frame;

                        auto p = bounds.pos() + bounds.size() * 0.5;
                        positions.back().push_back(p);

                        if (frame == tracker->end_frame()) {
                            base.circle(p, 10, fish->identity().color());
                            //auto cache = fish->cache_for_frame(frame, tracker->properties(frame)->time);
                            //base.text(Meta::toStr(fish->probability(cache, frame, basic->blob).p), positions.back() - Vec2(0, -100));

                            if (posture) {
                                std::vector<Vertex> oline;
                                auto &_cached_outline = posture->outline;
                                auto &_cached_midline = posture->cached_pp_midline;
                                auto &clr = fish->identity().color();
                                auto max_color = 255;
                                auto points = _cached_outline->uncompress();

                                // check if we actually have a tail index
                                if (gui_show_midline && _cached_midline && _cached_midline->tail_index() != -1) {
                                    base.circle(points.at(_cached_midline->tail_index()) + bounds.pos(), 5, Blue.alpha(max_color * 0.3));
                                    if (_cached_midline->head_index() != -1)
                                        base.circle(points.at(_cached_midline->head_index()) + bounds.pos(), 5, Red.alpha(max_color * 0.3));
                                }

                                //float right_side = outline->tail_index() + 1;
                                //float left_side = points.size() - outline->tail_index();

                                for (size_t i = 0; i < points.size(); i++) {
                                    auto pt = points[i] + bounds.pos();
                                    Color c = clr.alpha(max_color);
                                    oline.push_back(Vertex(pt, c));
                                }
                                oline.push_back(Vertex(points.front() + bounds.pos(), clr.alpha(0.04 * max_color)));
                                //auto line =
                                base.add_object(new Line(oline, gui_outline_thickness));
                            }
                        }
                    }
                    return true;
                });

                if (!tags_recognize) {
                    for (auto& v : positions)
                        base.line(v, 2, fish->identity().color());
                }
            }
        }*/
    });
}

std::string GUI::info_text() const {
    std::stringstream ss;
    auto frame = _grabber.last_index().load();
    if(frame)
        ss << "frame "+std::to_string(frame);
    if(_grabber.video()) {
        ss << "/" << _grabber.video()->length().toStr();
    }
    ss << " " << std::fixed << std::setprecision(1) << _grabber.fps() << "fps";
    //ss << " compratio:" << std::fixed << std::setprecision(2) << _grabber.processed().compression_ratio()*100 << "%";
    //ss << " network: " << std::fixed << std::setprecision(2) << float(_gui_bytes_per_second/1024.f) << "kb/s";
    return ss.str();
}

void GUI::static_event(DrawStructure&, const gui::Event& e) {
    _instance->event(e);
}

void GUI::event(const gui::Event &event) {
    if (event.type == gui::KEY) {
        key_event(event);
        
    } else if(event.type == gui::MMOVE) {
        set_redraw();
    } else if(event.type == gui::WINDOW_RESIZED) {
        using namespace gui;
        Size2 size(event.size.width, event.size.height);
        
        float scale = min(size.width / float(_cropped_size.width),
                          size.height / float(_cropped_size.height));
        //_gui.set_scale(scale * gui::interface_scale()); // SETTING(cam_scale).value<float>());
        gui().set_size(size);
        gui().set_dirty(NULL);
        //_gui.event(event);
        
        Vec2 real_size(_cropped_size.width * scale,
                       _cropped_size.height * scale);
        
        set_redraw();
    }
}

void GUI::key_event(const gui::Event &event) {
    auto &key = event.key;
    if(key.pressed)
        return;
    
    using namespace gui;
    
    if (key.code == Codes::Escape)
        VALUE(terminate) = true;
    else if(key.code == Codes::Add || key.code == Codes::Subtract || key.code == Codes::RBracket || key.code == Codes::Slash) {
        auto code = key.code == Codes::RBracket ? Codes::Add : Codes::Subtract;
        
        SETTING(threshold) = SETTING(threshold).value<int>() + (code == Codes::Add ? 1 : -1);
        print("Threshold ", SETTING(threshold).value<int>());
    }
    else if(key.code == Codes::R) {
        SETTING(recording) = !SETTING(recording);
    }
    else if(key.code == Codes::K) {
        print("Killing...");
        CrashProgram::do_crash = true;
    }
    else if(key.code == Codes::F5) {
        SETTING(reset_average) = true;
    }
#ifndef NDEBUG
    else if(key.code == Codes::Unknown)
        print("Unknown key ",key.code);
#endif
    
    set_redraw();
}

void GUI::set_redraw() {
    std::lock_guard<std::mutex> guard(_display_queue_lock);
    _redraw = true;
}

bool GUI::terminated() const {
    return VALUE(terminate);
}
}
