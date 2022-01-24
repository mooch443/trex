#include "DrawFish.h"
#include <gui/DrawSFBase.h>
#include <misc/OutputLibrary.h>
#include <cmath>
#include <tracking/Individual.h>
#include <tracking/VisualField.h>
#include <gui.h>
#include <misc/CircularGraph.h>
#include <misc/create_struct.h>
#include <gui/DrawMenu.h>
#include <gui/Label.h>
#include <tracking/Recognition.h>
#include <tracking/Categorize.h>
#include <gui/IMGUIBase.h>
#include <gui/DrawBase.h>
#include <tracking/DetectTag.h>

using namespace track;

namespace gui {

CREATE_STRUCT(CachedGUIOptions,
    (bool, gui_show_outline),
    (bool, gui_show_midline),
    (gui::Color, gui_single_identity_color),
    (std::string, gui_fish_color),
    (bool, gui_show_boundary_crossings),
    (uchar, gui_faded_brightness),
    (bool, gui_show_probabilities),
    (bool, gui_show_shadows),
    (bool, gui_show_selections),
    (bool, gui_show_paths),
    (size_t, gui_outline_thickness),
    (bool, gui_show_texts),
    (float, gui_max_path_time),
    (int, panic_button),
    (bool, gui_happy_mode),
    (bool, gui_highlight_categories),
    (bool, gui_show_cliques),
    (bool, gui_show_match_modes)
)

#define GUIOPTION(NAME) CachedGUIOptions::copy < CachedGUIOptions :: NAME > ()

    Fish::Fish(Individual& obj)
        : gui::DrawableCollection(obj.identity().name()),
        _obj(obj),
        _idx(-1),
        _graph(Bounds(0, 0, 300, 300), "Recent direction histogram"),
        _info(&_obj, OptionsList<Output::Modifiers>{})
    {
        CachedGUIOptions::init();
        
        assert(_obj.identity().ID().valid());
        auto ID = _obj.identity().ID();
        set_clickable(true);
        _circle.set_clickable(true);
        on_hover([](auto e) {
            if(!GUI::instance() || !e.hover.hovered)
                return;
            GUI::cache().set_tracking_dirty();
        });
        on_click([ID, this](auto) {
            std::vector<Idx_t> selections = SETTING(gui_focus_group);
            auto it = std::find(selections.begin(), selections.end(), ID);
            
            if(stage() && !(stage()->is_key_pressed(gui::LShift) || stage()->is_key_pressed(gui::RShift))) {
                if(it != selections.end())
                    GUI::cache().deselect_all();
                else
                    GUI::cache().deselect_all_select(ID);
                
            } else {
                if(it != selections.end())
                    GUI::cache().deselect(ID);
                else
                    GUI::cache().do_select(ID);
            }
            
            
            //SETTING(gui_focus_group) = selections;
            this->set_dirty();
        });
        
        _posture.set_origin(Vec2(0.5));
    }
    
    void Fish::set_data(long_t frameIndex, double time, const PPFrame &frame, const EventAnalysis::EventMap *events)
    {
        _safe_idx = _obj.find_frame(frameIndex)->frame;
        _time = time;
        _frame = &frame;
        _events = events;
        
        if(_idx != frameIndex) {
            _library_y = Graph::invalid();
            _avg_cat = -1;
            _next_frame_cache.valid = false;
            _image = nullptr;
            points.clear();
            
            auto seg = _obj.segment_for(_idx);
            if(seg) {
                _match_mode = (int)_obj.matched_using().at(seg->basic_stuff(_idx)).value();
            } else
                _match_mode = -1;
            
            auto && [basic, posture] = _obj.all_stuff(_safe_idx);
            
            std::shared_ptr<Individual::PostureStuff> current_posture;
            std::shared_ptr<Individual::BasicStuff> current_basic;
            
            if(frameIndex == _safe_idx) {
                current_posture = posture;
                current_basic = basic;
            } else {
                auto && [basic, posture] = _obj.all_stuff(frameIndex);
                current_posture = posture;
                current_basic = basic;
            }
            
            _cached_outline = current_posture ? current_posture->outline : nullptr;
            
            if(GUIOPTION(gui_show_outline) || GUIOPTION(gui_show_midline) || GUIOPTION(gui_happy_mode)) {
                if(current_posture) {
                    _cached_midline = SETTING(output_normalize_midline_data) ? _obj.fixed_midline(frameIndex) : _obj.calculate_midline_for(current_basic, current_posture);
                    assert(!_cached_midline || _cached_midline->is_normalized());
                }
            }
            
            set_dirty();
            
            _blob = _obj.compressed_blob(_safe_idx);
            _blob_bounds = _blob ? _blob->calculate_bounds() : bounds();

            if (_blob) {

                auto p = tags::prettify_blobs(
                    GUI::instance()->cache().processed_frame.blobs(), 
                    GUI::instance()->cache().processed_frame.noise(), 
                    GUI::instance()->cache().processed_frame.original_blobs(), 
                    Tracker::instance()->background()->image());

                for (auto& image : p) {

                    auto t = tags::is_good_image(image, Tracker::instance()->background()->image());
                    if (t.image) {
                        auto detector = cv::SiftFeatureDetector::create();
                        cv::Mat local;
                        cv::equalizeHist(t.image->get(), local);

                        std::vector<cv::KeyPoint> keypoints;
                        detector->detect(t.image->get(), keypoints);

                        cv::Mat k;
                        resize_image(local, k, 15.0);

                        for (auto& k : keypoints) {
                            k.pt *= 15;
                        }

                        cv::Mat output;
                        cv::drawKeypoints(k, keypoints, output);
                        tf::imshow("keypoints", output);
                        tf::imshow("tag", k);
                    }
                }


                /*for (auto& b : GUI::instance()->cache().processed_frame.blobs()) {
                    if (b->blob_id() == _blob->blob_id() || (long_t)b->blob_id() == _blob->parent_id) {
                        auto&& [image_pos, image] = b->difference_image(*Tracker::instance()->background(), 0);
                        tf::imshow("blob", image->get());
                        
                    }
                //}*/
            }
            
            auto [_basic, _posture] = _obj.all_stuff(_safe_idx);
            _basic_stuff = _basic;
            _posture_stuff = _posture;
            
            if(frameIndex == _safe_idx && _basic) {
                auto c = Categorize::DataStore::label_averaged(&_obj, Frame_t(frameIndex));
                if(c)
                    _avg_cat = c->id;
            }
            
            auto color_source = GUIOPTION(gui_fish_color);
            if(color_source != "identity" && _blob) {
                _library_y = Output::Library::get_with_modifiers(color_source, _info, _safe_idx);
                if(!Graph::is_invalid(_library_y)) {
                    if(color_source == "X") _library_y /= float(Tracker::average().cols) * FAST_SETTINGS(cm_per_pixel);
                    else if(color_source == "Y") _library_y /= float(Tracker::average().rows) * FAST_SETTINGS(cm_per_pixel);
                }
            }
        }
        
        _idx = frameIndex;
    }
    
    /*void Fish::draw_occlusion(gui::DrawStructure &window) {
        auto &blob = _obj.pixels(_safe_idx);
        window.image(blob_bounds.pos(), _image);
    }*/
    
    void Fish::update(DrawStructure &window) {
        const int frame_rate = FAST_SETTINGS(frame_rate);
        //const float track_max_reassign_time = FAST_SETTINGS(track_max_reassign_time);
        const auto single_identity = GUIOPTION(gui_single_identity_color);
        //const auto properties = Tracker::properties(_idx);
        //const auto safe_properties = Tracker::properties(_safe_idx);
        auto &cache = GUI::instance()->cache();
        
        set_bounds(_blob_bounds);
        
        const Vec2 offset = -_blob_bounds.pos();
        
        const auto centroid = _basic_stuff ? _basic_stuff->centroid : nullptr;
        const auto head = _posture_stuff ? _posture_stuff->head : nullptr;
        
        _fish_pos = centroid ? centroid->pos(Units::PX_AND_SECONDS) : (_blob_bounds.pos() + _blob_bounds.size() * 0.5);
        
        const bool hovered = this->hovered();
        const bool timeline_visible = GUI::instance() && GUI::instance()->timeline().visible();
        const float max_color = timeline_visible ? 255 : GUIOPTION(gui_faded_brightness);
        
        auto base_color = single_identity != Transparent ? single_identity : _obj.identity().color();
        //auto clr = base_color.alpha(saturate(((cache.is_selected(_obj.identity().ID()) || hovered) ? max_color : max_color * 0.4f) * time_fade_percent));
        auto clr = base_color.alpha(saturate(max_color));// * time_fade_percent));
        _color = clr;
        
        auto current_time = _time;
        auto next_props = Tracker::properties(_idx + 1);
        auto next_time = next_props ? next_props->time : (current_time + 1.f/float(frame_rate));
        
        auto active = GUI::cache().active_ids.find(_obj.identity().ID()) != GUI::cache().active_ids.end();
        bool is_selected = cache.is_selected(_obj.identity().ID());
        std::vector<Vec2> points;
        
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && TREX_METAL_AVAILABLE
        if(GUI_SETTINGS(gui_blur_enabled) && std::is_same<MetalImpl, default_impl_t>::value)
        {
            if(!is_selected) tag(Effects::blur);
            else untag(Effects::blur);
            
            if(is_selected && GUI::instance() && GUI::instance()->base()) {
                ((MetalImpl*)((IMGUIBase*)GUI::instance()->base())->platform().get())->center[0] = global_bounds().x / float(GUI::instance()->base()->window_dimensions().width) / gui::interface_scale() * window.scale().x;
                ((MetalImpl*)((IMGUIBase*)GUI::instance()->base())->platform().get())->center[1] = global_bounds().y / float(GUI::instance()->base()->window_dimensions().height) / gui::interface_scale() * window.scale().y;
            }
        }
#endif
#endif
        
        if(GUIOPTION(gui_show_paths))
            paintPath(window, offset, _safe_idx, cmn::max(_obj.start_frame(), _safe_idx - 1000l), base_color);
        
        if(FAST_SETTINGS(track_max_individuals) > 0 && GUIOPTION(gui_show_boundary_crossings))
            update_recognition_circle(window);
        
        /*if(midline && !midline->is_normalized()) {
            midline = midline->normalize();
        }*/
        
        if(active && _cached_outline) {
            if(GUIOPTION(gui_show_shadows) || GUIOPTION(gui_show_outline)) {
                points = _cached_outline->uncompress();
            }
            
            /*if(GUIOPTION(gui_show_shadows) && _polygon) {
                _polygon->set_vertices(points);
                float size = Tracker::average().bounds().size().length() * 0.0025f;
                Vec2 scaling(SQR(offset.x / float(Tracker::average().cols)),
                             SQR(offset.y / float(Tracker::average().rows)));
                _polygon->set_pos(scaling * size + this->size() * 0.5);
                _polygon->set_scale(scaling * 0.25 + 1);
                
    #ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
    #if defined(__APPLE__) && TREX_METAL_AVAILABLE
                if(GUI_SETTINGS(gui_blur_enabled) && std::is_same<MetalImpl, default_impl_t>::value)
                {
                    if(is_selected)_polygon->tag(Effects::blur);
                    else _polygon->untag(Effects::blur);
                }
    #endif
    #endif
            }*/
        }
        
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && TREX_METAL_AVAILABLE
        auto it = cache.fish_selected_blobs.find(_obj.identity().ID());
        if(it != cache.fish_selected_blobs.end()) {
            for(auto & [b, ptr] : cache.display_blobs) {
                if(b->blob_id() == it->second) {
                    ptr->set_pos(Vec2());
                    if(GUI_SETTINGS(gui_blur_enabled) && std::is_same<MetalImpl, default_impl_t>::value)
                    {
                        ptr->untag(Effects::blur);
                    }
                    window.wrap_object(*ptr);
                    break;
                }
            }
        }
#endif
#endif
        
        // DRAW OUTLINE / MIDLINE ON THE MAIN GRAYSCALE IMAGE
        const double damping_linear = .5;
        Vec2 _force = _v * (-damping_linear);
        Vec2 mp;
        
        const int panic_button = GUIOPTION(panic_button);
        auto section = panic_button ? window.find("fishbowl") : nullptr;
        if(section) {
            Vec2 mouse_position = window.mouse_position();
            mouse_position = (mouse_position - section->pos()).div(section->scale());
            mp = mouse_position - this->pos();
        }
        
        _posture.update([this, panic_button, mp, &_force, max_color, &head, &offset, fish = this, active, &points](Entangled& window) {
            if(panic_button) {
                if(float(rand()) / float(RAND_MAX) > 0.75) {
                    _color = _wheel.next();
                }
                
                float r = float(rand()) / float(RAND_MAX);
                _plus_angle += sinf(0.5f * (r - 0.5f) * 2 * float(M_PI) * GUI::cache().dt());
                window.set_rotation(_plus_angle);
                
                //r = float(rand()) / float(RAND_MAX);
                //_position += Vec2(r - 0.5, r - 0.5) * 2 * 10 * GUI::cache().dt();
                
                
                Vec2 distance = _position - (panic_button == 1 ? mp : Vec2());
                auto CL = distance.length();
                if(std::isnan(CL))
                    CL = 0.0001f;
                
                const float stiffness = 50, spring_L = panic_button == 1 ? 2 : 0, spring_damping = 20;
                
                Vec2 f = distance / CL;
                f *= - (stiffness * (CL - spring_L)
                     + spring_damping * _v.dot(distance) / CL);
                _force += f;
                
                float _mass = 5;
                _v += (_force / _mass + Vec2(0, 9.81 / 0.0025)) * GUI::cache().dt();
                
                _position += _v * GUI::cache().dt();
                
                window.set_pos(_position);
                window.set_size(this->size());
            } else {
                _position.x = _position.y = 0;
                window.set_pos(_position);
                window.set_rotation(0);
                window.set_size(Size2());
            }
            
            if(active && _cached_outline && GUIOPTION(gui_show_outline) ){
                std::vector<Vertex> oline;
                points = _cached_outline->uncompress();
                
                if(GUIOPTION(gui_show_shadows)) {
                    if(!_polygon) {
                        _polygon = std::make_shared<Polygon>(std::make_shared<std::vector<Vec2>>());
                        _polygon->set_fill_clr(Black.alpha(25));
                        _polygon->set_origin(Vec2(0.5));
                    }
                    _polygon->set_vertices(points);
                    float size = Tracker::average().bounds().size().length() * 0.0025f;
                    Vec2 scaling(SQR(offset.x / float(Tracker::average().cols)),
                                 SQR(offset.y / float(Tracker::average().rows)));
                    _polygon->set_pos(-offset + scaling * size + fish->size() * 0.5);
                    _polygon->set_scale(scaling * 0.25 + 1);
                    _polygon->set_fill_clr(Black.alpha(25));
                    
                    //window.advance_wrap(*_polygon);
                }
                
                // check if we actually have a tail index
                if(GUIOPTION(gui_show_midline) && _cached_midline && _cached_midline->tail_index() != -1)
                    window.advance(new Circle(points.at(_cached_midline->tail_index()), 2, Blue.alpha(max_color * 0.3f)));
                
                //float right_side = outline->tail_index() + 1;
                //float left_side = points.size() - outline->tail_index();
                
                for(size_t i=0; i<points.size(); i++) {
                    auto &pt = points[i];
                    Color c = _color.alpha(max_color);
                    /*if(outline->tail_index() != -1) {
                        float d = cmn::abs(float(i) - float(outline->tail_index())) / ((long_t)i > outline->tail_index() ? left_side : right_side) * 0.4 + 0.5;
                        c = Color(clr.r, clr.g, clr.b, max_color * d);
                    }*/
                    oline.push_back(Vertex(pt, c));
                }
                oline.push_back(Vertex(points.front(), _color.alpha(0.04 * max_color)));
                //auto line =
                window.advance(new Line(oline, GUIOPTION(gui_outline_thickness)));
                //if(line)
                //    window.text(Meta::toStr(line->points().size()) + "/" + Meta::toStr(oline.size()), Vec2(), White);
                //window.vertices(oline);
                
            }
            if(active && _cached_midline && GUIOPTION(gui_show_midline)) {
                std::vector<MidlineSegment> midline_points;
                //Midline _midline(*_cached_midline);
                //float len = _obj.midline_length();
                
                //if(len > 0)
                //    _midline.fix_length(len);
                
                auto& _midline = *_cached_midline;
                midline_points = _midline.segments();
                
                std::vector<Vertex> line;
                auto tf = _cached_midline->transform(default_config::recognition_normalization_t::none, true);
                
                for (auto segment : midline_points) {
                    //Vec2 pt = segment.pos;
                    //float angle = _cached_midline->angle() + M_PI;
                    //float x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
                    //float y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));
                    
                    //pt = Vec2(x, y) + _cached_midline->offset();
                    line.push_back(Vertex(tf.transformPoint(segment.pos), _color));
                }
                
                window.advance(new Line(line, GUIOPTION(gui_outline_thickness)));
                //window.vertices(line);
                
                if(head) {
                    window.advance(new Circle(head->pos(Units::PX_AND_SECONDS) + offset, 3, Red.alpha(max_color)));
                }
            }
        });
        
        if(panic_button) {
            window.line(_posture.pos(), mp, White.alpha(50));
            GUI::cache().set_animating(this, true);
        } else
            GUI::cache().set_animating(this, false);
        
        window.wrap_object(_posture);
        
        // DISPLAY LABEL AND POSITION
        auto c_pos = centroid->pos(Units::PX_AND_SECONDS) + offset;
        if(c_pos.x > Tracker::average().cols || c_pos.y > Tracker::average().rows)
            return;
        
        auto v = 255 - int(Tracker::average().at(c_pos.y, c_pos.x));
        if(v >= 100)
            v = 220;
        else
            v = 50;
        
        float angle = -centroid->angle();
        if (head) {
            angle = -head->angle();
        }
        
        auto ML = _obj.midline_length();
        auto radius = (FAST_SETTINGS(calculate_posture) && ML != Graph::invalid() ? ML : _blob_bounds.size().max()) * 0.6;
        if(GUIOPTION(gui_show_texts)) {
            // DISPLAY NEXT POSITION (estimated position in _idx + 1)
            //if(cache.processed_frame.cached_individuals.count(_obj.identity().ID())) {
            if(!_next_frame_cache.valid)
                _next_frame_cache = _obj.cache_for_frame(_idx + 1, next_time);
            auto estimated = _next_frame_cache.estimated_px + offset;
            
            window.circle(c_pos, 2, White.alpha(max_color));
                //auto &fcache = cache.processed_frame.cached_individuals.at(_obj.identity().ID());
                //auto estimated = cache.estimated_px + offset;
                //float tdelta = next_time - current_time;
                //float tdelta = fcache.local_tdelta;
                
                window.line(c_pos, estimated, clr);
                window.circle(estimated, 2, Transparent, clr);
                
                //const float max_d = FAST_SETTINGS(track_max_speed) * tdelta / FAST_SETTINGS(cm_per_pixel);
                //window.circle(estimated, max_d * 0.5, Red.alpha(100));
            //}
            
            //window.circle(estimated, FAST_SETTINGS(track_max_speed) * tdelta, clr);
        }
        
        if(GUIOPTION(gui_happy_mode) && _cached_midline && _cached_outline && _posture_stuff && _posture_stuff->head) {
            struct Physics {
                Vec2 direction = Vec2();
                Vec2 v = Vec2();
                long_t frame = -1;
                double blink = 0;
                bool blinking = false;
                double blink_limit = 10;
            };
            
            constexpr double damping_linear = .5;
            constexpr float stiffness = 100, spring_L = 0, spring_damping = 1;
            static std::unordered_map<const Individual*, Physics> _current_angle;
            auto &ph = _current_angle[&_obj];
            double dt = GUI::cache().dt();
            if(dt > 0.1)
                dt = 0.1;
            
            Vec2 force = ph.v * (-damping_linear);
            
            if(ph.frame != _idx) {
                ph.direction += 0; // rad/s
                ph.frame = _idx;
            }
            
            auto alpha = _posture_stuff->head->angle();
            Vec2 movement = Vec2(cos(alpha), sin(alpha));
            Vec2 distance = ph.direction - movement;
            double CL = distance.length();
            if(std::isnan(CL))
                CL = 0.0001;
            
            if(CL != 0) {
                Vec2 f = distance / CL;
                f *= - (stiffness * (CL - spring_L) + spring_damping * ph.v.dot(distance) / CL);
                force += f;
            }
            
            double mass = 1;
            ph.v += force / mass * dt;
            ph.direction += ph.v * dt;
            
            auto &&[eyes, off] = VisualField::generate_eyes(&_obj, _basic_stuff, points, _cached_midline, alpha);
            
            auto d = ph.direction;
            auto L = d.length();
            if(L > 0) d /= L;
            if(L > 1) L = 1;
            d *= L;
            
            double h = ph.blink / 0.1;
            
            if(h > ph.blink_limit && !ph.blinking) {
                ph.blinking = true;
                ph.blink = 0;
                h = 0;
                ph.blink_limit = rand() / double(RAND_MAX) * 30;
            }
            
            if(h > 1 && ph.blinking) {
                ph.blinking = false;
            }
            
            ph.blink += dt;
            
            auto sun_direction = (offset - Vec2(0)).normalize();
            auto eye_scale = max(0.5, _obj.midline_length() / 90);
            for(auto &eye : eyes) {
                eye.pos += ph.direction;
                window.circle(eye.pos + offset, 5 * eye_scale, Black.alpha(200), White.alpha(125));
                auto c = window.circle(eye.pos + Vec2(2.5).mul(d * eye_scale) + offset, 3 * eye_scale, Transparent, Black.alpha(200));
                c->set_scale(Vec2(1, ph.blinking ? h : 1));
                c->set_rotation(atan2(ph.direction) + RADIANS(90));//posture->head->angle() + RADIANS(90));
                window.circle(eye.pos + Vec2(2.5).mul(d * eye_scale) + Vec2(2 * eye_scale).mul(sun_direction) + offset, sqrt(eye_scale), Transparent, White.alpha(200 * c->scale().min()));
            }
        }
        
        
            auto color_source = GUIOPTION(gui_fish_color);
            if(color_source == "viridis") {
                for(auto &b : GUI::instance()->cache().processed_frame.blobs()) {
                    if(b->blob_id() == _blob->blob_id() || (long_t)b->blob_id() == _blob->parent_id) {
                        auto && [dpos, difference] = b->difference_image(*Tracker::instance()->background(), 0);
                        auto rgba = Image::Make(difference->rows, difference->cols, 4);
                        
                        uchar maximum_grey = 255, minimum_grey = 0;//std::numeric_limits<uchar>::max();
                        /*for(size_t i=0; i<difference->size(); ++i) {
                            auto c = difference->data()[i];
                            maximum_grey = max(maximum_grey, c);
                            
                            if(difference->data()[i] > 0)
                                minimum_grey = min(minimum_grey, c);
                        }
                        for(size_t i=0; i<difference->size(); ++i)
                            difference->data()[i] = (uchar)min(255, float(difference->data()[i]) / maximum_grey * 255);
                        
                        if(minimum_grey == maximum_grey)
                            minimum_grey = maximum_grey - 1;*/
                        
                        auto ptr = rgba->data();
                        auto m = difference->data();
                        for(; ptr < rgba->data() + rgba->size(); ptr += rgba->dims, ++m) {
                            auto c = Viridis::value((float(*m) - minimum_grey) / (maximum_grey - minimum_grey));
                            *(ptr) = c.r;
                            *(ptr+1) = c.g;
                            *(ptr+2) = c.b;
                            *(ptr+3) = *m;
                        }
                        
                        window.image(dpos + offset, std::move(rgba), Vec2(1));
                        
                        break;
                    }
                }
                
            } else if(!Graph::is_invalid(_library_y)) {
                auto percent = min(1.f, cmn::abs(_library_y));
                Color clr = /*Color(225, 255, 0, 255)*/ base_color * percent + Color(50, 50, 50, 255) * (1 - percent);
                
                for(auto &b : GUI::instance()->cache().processed_frame.blobs()) {
                    if(b->blob_id() == _blob->blob_id() || (long_t)b->blob_id() == _blob->parent_id) {
                        auto && [image_pos, image] = b->binary_image(*Tracker::instance()->background(), FAST_SETTINGS(track_threshold));
                        auto && [dpos, difference] = b->difference_image(*Tracker::instance()->background(), 0);
                        
                        auto rgba = Image::Make(image->rows, image->cols, 4);
                        
                        uchar maximum = 0;
                        for(size_t i=0; i<difference->size(); ++i) {
                            maximum = max(maximum, difference->data()[i]);
                        }
                        for(size_t i=0; i<difference->size(); ++i)
                            difference->data()[i] = (uchar)min(255, float(difference->data()[i]) / maximum * 255);
                        
                        rgba->set_channels(image->data(), {0, 1, 2});
                        rgba->set_channel(3, difference->data());
                        
                        window.image(image_pos + offset, std::move(rgba), Vec2(1), clr);
                        
                        break;
                    }
                }
            }
        
            
        if(is_selected && GUIOPTION(gui_show_probabilities)) {
            if(!_image) {
                auto probability = Image::Make(Tracker::average().rows, Tracker::average().cols, 4);
                
                auto mat = probability->get();
                mat.setTo(0);
                auto c = cache.processed_frame.cached(_obj.identity().ID());
                if(c) {
                    Vec2 start = c->estimated_px;
                    int32_t radius = 0;
                    float sum;
                    //const float norm = exp(M_PI) - exp(0);
                    
                    auto plot = [&](int x, int y){
                        if(x < 0 || x >= mat.cols)
                            return;
                        if(y < 0 || y >= mat.rows)
                            return;
                        
                        auto p = _obj.probability(-1, *c, _idx, Vec2(x, y) + 1 * 0.5, 1);
                        if(p.p < FAST_SETTINGS(matching_probability_threshold))
                            return;
                        
                        auto clr = DarkCyan.alpha(min(255, 255.f * p.p));//(1 - 1 / (1 + (exp(p.p * M_PI) - exp(0)) / norm))));
                        auto ptr = mat.ptr(y, x);
                        *(ptr + 0) = clr.r;
                        *(ptr + 1) = clr.g;
                        *(ptr + 2) = clr.b;
                        *(ptr + 3) = clr.a;
                        //cv::rectangle(mat, Vec2(x, y), Vec2(x, y) + 1, DarkCyan.alpha(255 * p.p * p.p), -1);
                        
                        sum += p.p;
                    };
                    
                    do {
                        sum = 0;
                        int sx = start.x - radius;
                        int sy = start.y - radius;
                        
                        for (int x=sx; x <= sx+radius*2 && x<(int)probability->cols; ++x) {
                            plot(x, sy);
                            plot(x, sy + radius * 2);
                        }
                        
                        for (int y=sy + 1; y <= sy+radius*2 - 1 && y<(int)probability->rows; ++y) {
                            plot(sx, y);
                            plot(sx + radius * 2, y);
                        }
                        
                        ++radius;
                        
                    } while(sum > 0 || radius < 10);
                    
                    /*for (int x=0; x<_probability.cols; x+=5) {
                        for (int y=0; y<_probability.rows; y+=5) {
                            //auto ptr = mat.ptr(y, x);
                            auto p = _obj.probability(it->second, _idx, Vec2(x, y) + 5 * 0.5, 1);
                            cv::rectangle(mat, Vec2(x, y), Vec2(x, y) + 5, DarkCyan.alpha(255 * p.p), -1);
                            
                            *(ptr + 0) = p.p * 255;
                            *(ptr + 1) = 0;
                            *(ptr + 2) = p.p * 255;
                            *(ptr + 3) = p.p * 255;
                        }
                    }*/
                }
                
                _image = std::make_unique<ExternalImage>(std::move(probability), offset);
            }
            
            window.wrap_object(*_image);
        }
        
        if ((hovered || is_selected) && GUIOPTION(gui_show_selections)) {
            Color circle_clr = Color(v).alpha(saturate(max_color * (hovered ? 1.7 : 1)));
            if(cache.primary_selection() != &_obj)
                circle_clr = Gray.alpha(circle_clr.a);
            
            // draw circle around the fish
            
            
            _circle.set_pos(c_pos);
            _circle.set_radius(radius);
            _circle.set_line_clr(circle_clr);
            _circle.set_fill_clr(hovered ? White.alpha(circle_clr.a * 0.1) : Transparent);
            window.wrap_object(_circle);
            
            //window.circle(c_pos, radius, circle_clr, hovered ? White.alpha(circle_clr.a * 0.1) : Transparent);
            
            // draw unit circle showing the angle of the fish
            Vec2 pos(cmn::cos(angle), -cmn::sin(angle));
            pos = pos * radius + c_pos;
            
            window.circle(pos, 3, circle_clr);
            window.line(c_pos, pos, circle_clr);
            
            if(FAST_SETTINGS(posture_direction_smoothing)) {
                std::map<long_t, float> angles;
                std::map<long_t, float> dangle, ddangle, interp;
                
                float previous = FLT_MAX;
                bool hit = false;
                float value = 0;
                size_t count_ones = 0;
                
                for (long_t frame = _idx - (long_t)FAST_SETTINGS(posture_direction_smoothing); frame <= _idx + (long_t)FAST_SETTINGS(posture_direction_smoothing); ++frame)
                {
                    auto midline = _obj.pp_midline(frame);
                    if(midline) {
                        auto angle = midline->original_angle();
                        angles[frame] = angle;
                        
                        if(previous != FLT_MAX) {
                            auto val = abs((Vec2(cos(previous), sin(previous)).dot(Vec2(cos(angle), sin(angle))) - 1) * 0.5);
                            
                            if(!dangle.empty()) {
                                ddangle[frame] = val - dangle.rbegin()->second;
                                if(ddangle[frame] <= -0.75) {
                                    if(!hit) {
                                        hit = true;
                                        value = 1;
                                    } else {
                                        hit = false;
                                        value = 0;
                                    }
                                }
                            }
                            
                            interp[frame] = value;
                            if(hit) {
                                ++count_ones;
                            }
                            
                            dangle[frame] = val;
                        }
                        previous = angle;
                    }
                }
                
                /*if(count_ones >= interp.size() * 0.5) {
                    for(auto & [frame, n] : interp) {
                        n = n ? 0 : 1;
                    }
                }*/
                
                for(auto && [frame, in] : interp) {
                    if(frame == _idx) {
                        _graph.set_title(Meta::toStr(ddangle.count(frame) ? ddangle.at(frame) : FLT_MAX) + " " +Meta::toStr(in));
                        //Debug("%d: %f (%f)", frame, dangle.back(), ddangle.empty() ? FLT_MAX : ddangle.back());
                    }
                }
                
                _graph.clear();
                _graph.set_pos(c_pos + Vec2(radius, radius));
                
                auto first_frame = interp.empty() ? 0 : interp.begin()->first;
                auto last_frame = interp.empty() ? 0 : interp.rbegin()->first;
                _graph.set_ranges(Rangef(first_frame, last_frame), Rangef(-1, 1));
                
                std::vector<Vec2> points;
                for(auto && [frame, a] : dangle) {
                    points.push_back(Vec2(frame, a));
                }
                _graph.add_points("angle'", points);
                
                points.clear();
                for(auto && [frame, a] : ddangle) {
                    points.push_back(Vec2(frame, a));
                }
                _graph.add_points("angle''", points);
                _graph.set_zero(_idx);
                
                window.wrap_object(_graph);
            }
        }
    }
    
    void Fish::paintPath(DrawStructure& window, const Vec2& offset, long_t to, long_t from, const Color& base_color) {
        if (to == -1)
            to = _obj.end_frame();
        if (from == -1)
            from = _obj.start_frame();
        
        from = _obj.start_frame();
        to = min(_obj.end_frame(), _idx);
        
        float color_start = max(0, round(_idx - FAST_SETTINGS(frame_rate) * GUIOPTION(gui_max_path_time)));
        float color_end = max(color_start + 1, _idx);
        
        from = max(color_start, from);
        
        if(_prev_frame_range.start != _obj.start_frame() || _prev_frame_range.end > _obj.end_frame()) {
            frame_vertices.clear();
        }
        
        _prev_frame_range = Rangel(_obj.start_frame(), _obj.end_frame());
        
        vertices.clear();
        const float max_speed = FAST_SETTINGS(track_max_speed);
        const float thickness = GUIOPTION(gui_outline_thickness);
        
        long_t first = frame_vertices.empty() ? -1 : frame_vertices.begin()->frame;
        
        if(first != -1 && first < from && !frame_vertices.empty()) {
            auto it = frame_vertices.begin();
            while (it != frame_vertices.end() && it->frame < from)
                ++it;
            
            //auto end = it != frame_vertices.begin() ? it-1 : it;
            //Debug("(%d) #1 Erasing from %d to %d (%d-%d, %d-%d) %d-%d", _obj.identity().ID(), frame_vertices.begin()->frame, end->frame, from, to, first, last, startf, endf);
            
            frame_vertices.erase(frame_vertices.begin(), it);
            first = frame_vertices.empty() ? -1 : frame_vertices.begin()->frame;
        }
        
        if(first > from) {
            long_t i=(first != -1 ? first-1 : from);
            auto fit = _obj.iterator_for(i);
            auto end = _obj.frame_segments().end();
            auto begin = _obj.frame_segments().begin();
            //auto seg = _obj.segment_for(i);
            
            for (; i>=from; --i) {
                if(fit == end || (*fit)->start() > i) {
                    while(fit != begin && (*fit)->start() > i) {
                        --fit;
                    }
                }
                
                if(fit != end && (*fit)->contains(i)) {
                    auto id = (*fit)->basic_stuff(i);
                    if(id != -1) {
                        auto stuff = _obj.basic_stuff().at(id);
                        frame_vertices.push_front(FrameVertex{i, Vertex(stuff->centroid->pos(Units::PX_AND_SECONDS)), min(1, stuff->centroid->speed(Units::CM_AND_SECONDS) / max_speed)});
                    }
                }
            }
            
            first = frame_vertices.empty() ? -1 : frame_vertices.begin()->frame;
        }
        
        long_t last = frame_vertices.empty() ? -1 : frame_vertices.rbegin()->frame;
        if(last == -1)
            last = from;
        
        if(last > to && !frame_vertices.empty()) {
            auto it = --frame_vertices.end();
            while(it->frame > to && it != frame_vertices.begin())
                --it;
            
            //Debug("(%d) #2 Erasing from %d to %d (%d-%d, %d-%d)", _obj.identity().ID(), it->frame, frame_vertices.rbegin()->frame, from, to, first, last);
            
            frame_vertices.erase(it, frame_vertices.end());
        }
        
        last = frame_vertices.empty() ? -1 : frame_vertices.rbegin()->frame;
        
        if(last < to) {
            //Debug("(%d) searching from %d to %d", _obj.identity().ID(), max(from, last), to);
            long_t i=max(from, last);
            auto fit = _obj.iterator_for(i);
            auto end = _obj.frame_segments().end();
            
            for (; i<=to; ++i) {
                if(fit == end || (*fit)->end() < i) {
                    //seg = _obj.segment_for(i);
                    while(fit != end && (*fit)->end() < i)
                        ++fit;
                }
                
                if(fit != end && (*fit)->contains(i)) {
                    auto id = (*fit)->basic_stuff(i);
                    if(id != -1) {
                        auto stuff = _obj.basic_stuff().at(id);
                        frame_vertices.push_back(FrameVertex{i, Vertex(stuff->centroid->pos(Units::PX_AND_SECONDS)), min(1, stuff->centroid->speed(Units::CM_AND_SECONDS) / max_speed)});
                    }
                }
            }
            
            last = frame_vertices.empty() ? -1 : frame_vertices.rbegin()->frame;
        }
        
        auto clr = base_color.alpha(255);
        if(!Graph::is_invalid(_library_y)) {
            const auto single_identity = GUIOPTION(gui_single_identity_color);
            auto percent = min(1.f, cmn::abs(_library_y));
            if(single_identity.a != 0) {
                clr = single_identity;
            }
            
            clr = clr.alpha(255) * percent + Color(50, 50, 50, 255) * (1 - percent);
        }
        auto inactive_clr = clr.saturation(0.5);
        Color use = clr;
        
        const float max_distance = Individual::weird_distance() * 0.1 / FAST_SETTINGS(cm_per_pixel);
        
        auto prev = frame_vertices.empty() ? -1 : frame_vertices.begin()->frame;
        Vec2 prev_pos = frame_vertices.empty() ? Vec2(-1) : frame_vertices.begin()->vertex.position();
        for(auto & fv : frame_vertices) {
            float percent = (fv.speed_percentage * 0.15 + 0.85) * (float(fv.frame - color_start) / float(color_end - color_start));
            percent = percent * percent;
            
            if(fv.frame - prev > 1 || (prev != -1 && euclidean_distance(prev_pos, fv.vertex.position()) >= max_distance)) {
                use = inactive_clr;
                if(vertices.size() > 1) {
                    auto v = new Vertices(vertices, PrimitiveType::LineStrip, Vertices::TRANSPORT);
                    v->set_thickness(thickness);
                    window.add_object(v);
                }
                    //window.vertices(vertices, PrimitiveType::LineStrip);
                    //window.line(vertices, 2.0f);
                vertices.clear();
                
                //window.circle(fv.vertex.position() + offset, 1, White.alpha(percent * 255));
            } else
                use = clr;
            prev = fv.frame;
            prev_pos = fv.vertex.position();
            
            vertices.push_back(Vertex(fv.vertex.position() + offset, use.alpha(percent * 255)));
        }
        
       // Debug("(%d) ending up with %d vertices", _obj.identity().ID(), vertices.size());
        
        if(vertices.size() > 1) {
            auto v = new Vertices(vertices, PrimitiveType::LineStrip, Vertices::TRANSPORT);
            v->set_thickness(thickness);
            window.add_object(v);
            //window.vertices(vertices, PrimitiveType::LineStrip);
        }
            //window.line(vertices, 2.0f);
        
        /*auto last = _obj.find_frame(to)->centroid->pos(Units::PX_AND_SECONDS) + offset;
        int count = 0;
        
        const float max_speed = SETTING(track_max_speed);
        vertices.clear();
        
        long_t lastframe = to;
        from = max(to-max_frames, from);
        
        std::vector<Drawable*> events;
        
        const bool timeline_visible = GUI::instance() && GUI::instance()->timeline().visible();
        const float max_color = timeline_visible ? 255 : SETTING(gui_faded_brightness).value<uchar>();
        const Font font(1 / (1 - ((1 - GUI::instance()->cache().zoom_level()) * 0.5)) * 0.7, Align::Center);
        
        // make it so that max_frames is actually the number of frames available
        // even if more could be displayed.
        if(max_frames != -1)
            max_frames = min(max_frames, max(1, to - from + 1));
        
        PhysicalProperties* prev_centroid = NULL;
        for (long_t i=to; i>=from; i--) {
            // draw lines between previous locations
            auto c = _obj.centroid_weighted(i);
            if (c) {
                auto pos = c->pos(Units::PX_AND_SECONDS) + offset;
                auto clr = _obj.identity().color().alpha(255);
                
                float percent = c->speed(Units::CM_AND_SECONDS) / max_speed * 7;
                percent = min(1.f, percent) * 0.8;
                
                //if (cmn::abs(lastframe - i) > 1)
                //    clr = Transparent;
                //else
                    clr = clr * (1.0 - percent) + White * percent;
                
                float fade = (i - from) + 1;
                if(max_frames != -1) {
                    fade = min(float(max_frames), fade) / float(max_frames);
                } else {
                    fade /= float(max(1, to - from + 1));
                }
                
                if(_events) {
                    auto it = _events->events.find(i);
                    if(it != _events->events.end()) {
                        events.push_back(new Circle(pos, 1, White.alpha(0.7 * max_color * fade)));
                    }
                }
                
                if(_obj.is_manual_match(i) && SETTING(gui_show_manual_matches)) {
                    auto blob = _obj.compressed_blob(i);
                    events.push_back(new Circle(pos, 3, Transparent, _obj.identity().color().alpha(fade * max_color)));
                    events.push_back(new Text(Meta::toStr(_obj.identity().ID())+"="+Meta::toStr(blob ? blob->blob_id() : -1), pos, _obj.identity().color().alpha(fade * max_color), font, window.scale().reciprocal()));
                }
                
                //float distance = euclidean_distance(pos * FAST_SETTINGS(cm_per_pixel), vertices.back().position() * FAST_SETTINGS(cm_per_pixel));
                if(!vertices.empty() && (!prev_centroid || prev_centroid->time() - c->time() >= FAST_SETTINGS(track_max_reassign_time) * 0.5))
                {
                    if(vertices.size() > 1)
                        window.line(vertices, 2.0f);
                    vertices.clear();
                }
                
                Color clr0 = clr.alpha(fade * max_color);
                vertices.push_back(Vertex(pos, clr0));
                //cv::circle(target, pos, 1, clr * (1.0 - percent) + original * percent, -1);
                //cv::line(target, last, pos, clr * (1.0 - percent) + original * percent);
                last = pos;
                
                lastframe = i;
                count++;
                
                prev_centroid = c;
            }
            
            if (max_frames != -1 && count >= max_frames) {
                break;
            }
        }
        
        if(vertices.size() > 1)
            window.line(vertices, 2.0f);
        
        for(auto c : events) {
            if(c->type() == Type::CIRCLE)
                window.add_object((Circle*)c);
            else if(c->type() == Type::TEXT)
                window.add_object((Text*)c);
            else
                U_EXCEPTION("Unknown type.");
        }*/
    }
    
    void Fish::update_recognition_circle(DrawStructure& base) {
        if(Tracker::instance()->border().in_recognition_bounds(_fish_pos)) {
            if(!_recognition_circle) {
                // is inside bounds, but we didnt know that yet! start animation
                _recognition_circle = std::make_shared<Circle>(Vec2(), 1, Transparent, Cyan.alpha(50));
            }
            
            auto ts = GUI::cache().dt();
            float target_radius = 100;
            float percent = min(1, _recognition_circle->radius() / target_radius);
            
            if(percent < 0.99) {
                percent *= percent;
                
                _recognition_circle->set_pos(_fish_pos - pos());
                _recognition_circle->set_radius(_recognition_circle->radius() + ts * (1 - percent) * target_radius * 2);
                _recognition_circle->set_fill_clr(Cyan.alpha(50 * (1-percent)));
                GUI::cache().set_animating(this, true);
                
                base.wrap_object(*_recognition_circle);
                
            } else {
                GUI::cache().set_animating(this, false);
            }
            
        } else if(_recognition_circle) {
            _recognition_circle = nullptr;
        }
    }

void Fish::label(DrawStructure &base) {
    if(GUIOPTION(gui_highlight_categories)) {
        if(_avg_cat != -1) {
            base.circle(pos() + size() * 0.5, size().length(), Transparent, ColorWheel(_avg_cat).next().alpha(75));
        } else {
            base.circle(pos() + size() * 0.5, size().length(), Transparent, Purple.alpha(15));
        }
    }
    
    if(GUIOPTION(gui_show_match_modes)) {
        base.circle(pos() + size() * 0.5, size().length(), Transparent, ColorWheel(_match_mode).next().alpha(50));
    }
    
    //auto bdx = blob->blob_id();
    if(GUIOPTION(gui_show_cliques)) {
        uint32_t i=0;
        for(auto &clique : GUI::cache()._cliques) {
            if(contains(clique.fishs, _obj.identity().ID())) {
                base.circle(pos() + size() * 0.5, size().length(), Transparent, ColorWheel(i).next().alpha(50));
                break;
            }
            ++i;
        }
    }
    
    if (!GUIOPTION(gui_show_texts))
        return;
    
    auto blob = _obj.compressed_blob(_idx);
    if(!blob)
        return;
    
    std::string color = "";
    std::stringstream text;
    std::string secondary_text;

    text << _obj.identity().raw_name() << " ";
    
    if (DrawMenu::matching_list_open() && blob) {
        secondary_text = "blob" + Meta::toStr(blob->blob_id());
    }
    else if (GUI_SETTINGS(gui_show_recognition_bounds)) {
        auto&& [valid, segment] = _obj.has_processed_segment(_idx);
        if (valid) {
            auto&& [samples, map] = _obj.processed_recognition(segment.start());
            auto it = std::max_element(map.begin(), map.end(), [](const std::pair<long_t, float>& a, const std::pair<long_t, float>& b) {
                return a.second < b.second;
            });

            if (it == map.end() || it->first != _obj.identity().ID()) {
                color = "str";
                secondary_text += " avg" + Meta::toStr(it->first);
            }
            else
                color = "nr";
        }
    }
    
    auto raw = Tracker::instance()->recognition()->ps_raw(_idx, blob->blob_id());
    if (!raw.empty()) {
        auto it = std::max_element(raw.begin(), raw.end(), [](const std::pair<long_t, float>& a, const std::pair<long_t, float>& b) {
            return a.second < b.second;
            });

        if (it != raw.end()) {
            secondary_text += " loc" + Meta::toStr(it->first) + " (" + Meta::toStr(it->second) + ")";
        }
    }
    //auto raw_cat = Categorize::DataStore::label(Frame_t(_idx), blob);
    //auto cat = Categorize::DataStore::label_interpolated(_obj.identity().ID(), Frame_t(_idx));

    auto c = GUI::cache().processed_frame.cached(_obj.identity().ID());
    if(c) {
        auto cat = c->current_category;
        if(cat != -1) {
            auto l = Categorize::DataStore::label(cat);
            if(l)
                secondary_text += "<key>"+l->name+"</key>";
        }
    }
    
    auto cat = Categorize::DataStore::_label_unsafe(Frame_t(_idx), blob->blob_id());
    if (cat != -1) {
        secondary_text += std::string(" ") + (cat ? "<b>" : "") + "<i>" + Categorize::DataStore::label(cat)->name + "</i>" + (cat ? "</b>" : "");
    }
    
    if(_avg_cat != -1) {
        auto c = Categorize::DataStore::label(_avg_cat);
        if(c)
            secondary_text += (_avg_cat != -1 ? std::string(" ") : std::string()) + "<nr>" + c->name + "</nr>";
    }
    
    auto label = (Label*)custom_data("label");
    auto label_text = (color.empty() ? text.str() : ("<"+color+">"+text.str()+"</"+color+">")) + "<a>" + secondary_text + "</a>";
    if (!label) {
        label = new Label(label_text, blob->calculate_bounds(), fish_pos());
        add_custom_data("label", (void*)label, [](void* ptr) {
            delete (Label*)ptr;
        });
    }
    else
        label->set_data(label_text, blob->calculate_bounds(), fish_pos());

    label->update(base, base.active_section(), 1, blob == nullptr);
}

void Fish::shadow(DrawStructure &window) {
    auto active = GUI::cache().active_ids.find(_obj.identity().ID()) != GUI::cache().active_ids.end();
    
    if(GUIOPTION(gui_show_shadows) && active) {
        if(!_polygon) {
            _polygon = std::make_shared<Polygon>(std::make_shared<std::vector<Vec2>>());
            _polygon->set_fill_clr(Black.alpha(125));
            _polygon->set_origin(Vec2(0.5));
        }
        
        window.wrap_object(*_polygon);
    }
}
}
