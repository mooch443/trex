#include "DrawFish.h"
#include <gui/DrawSFBase.h>
#include <misc/OutputLibrary.h>
#include <tracking/Individual.h>
#include <tracking/VisualField.h>
#include <misc/CircularGraph.h>
#include <misc/create_struct.h>
#include <gui/Label.h>
#include <tracking/Categorize.h>
#include <gui/IMGUIBase.h>
#include <gui/DrawBase.h>
#include <tracking/DetectTag.h>
#include <gui/GUICache.h>
//#include <gui.h>
#include <misc/IdentifiedTag.h>
#include <gui/Skelett.h>

#if defined(__APPLE__) && defined(TREX_ENABLE_EXPERIMENTAL_BLUR)
//#include <gui.h>
#endif

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
    (uint8_t, gui_outline_thickness),
    (bool, gui_show_texts),
    (float, gui_max_path_time),
    (int, panic_button),
    (bool, gui_happy_mode),
    (bool, gui_highlight_categories),
    (bool, gui_show_cliques),
    (bool, gui_show_match_modes)
)

#define GUIOPTION(NAME) CachedGUIOptions::copy < CachedGUIOptions :: NAME > ()
Fish::~Fish() {
    if(GUICache::exists())
        GUICache::instance().set_animating(circle_animator, false);
    
    if (_label) {
        delete _label;
    }
}

    Fish::Fish(Individual& obj)
        :   _id(obj.identity()),
            _info(&obj, Output::Options_t{}),
            _graph(Bounds(0, 0, 300, 300), "Recent direction histogram")
    {
        _previous_color = obj.identity().color();
        CachedGUIOptions::init();
        
        assert(_id.ID().valid());
        auto ID = _id.ID();
        _view.set_clickable(true);
        _circle.set_clickable(true);
        _view.on_hover([this](auto e) {
            _path_dirty = true;
            if(!GUICache::exists() || !e.hover.hovered)
                return;
            GUICache::instance().set_tracking_dirty();
        });
        _view.on_click([ID, this](auto) {
            std::vector<Idx_t> selections = SETTING(gui_focus_group);
            auto it = std::find(selections.begin(), selections.end(), ID);
            
            if(_view.stage() && !(_view.stage()->is_key_pressed(gui::LShift) || _view.stage()->is_key_pressed(gui::RShift))) {
                if(it != selections.end())
                    GUICache::instance().deselect_all();
                else
                    GUICache::instance().deselect_all_select(ID);
                
            } else {
                if(it != selections.end())
                    GUICache::instance().deselect(ID);
                else
                    GUICache::instance().do_select(ID);
            }
            
            
            //SETTING(gui_focus_group) = selections;
            _view.set_dirty();
        });
        
        _posture.set_origin(Vec2(0));
    }

    void Fish::check_tags() {
        //if (_blob)
        {
            std::vector<pv::BlobPtr> blobs;
            std::vector<pv::BlobPtr> noise;
            
            GUICache::instance().processed_frame().transform_blobs([&](const pv::Blob &blob) {
                blobs.emplace_back(pv::Blob::Make(blob));
            });
            GUICache::instance().processed_frame().transform_noise([&](const pv::Blob &blob) {
                noise.emplace_back(pv::Blob::Make(blob));
            });
            
            auto p = tags::prettify_blobs(blobs, noise, {},
                //GUICache::instance().processed_frame().original_blobs(),
                Tracker::instance()->background()->image());

            for (auto& image : p) {

                auto t = tags::is_good_image(image);
                if (t.image) {
                    cv::Mat local;
                    cv::equalizeHist(t.image->get(), local);
                    /*auto detector = cv::SiftFeatureDetector::create();



                    std::vector<cv::KeyPoint> keypoints;
                    detector->detect(t.image->get(), keypoints);*/

                    cv::Mat k;
                    resize_image(local, k, 15.0);

                    /*for (auto& k : keypoints) {
                        k.pt *= 15;
                    }

                    cv::Mat output;
                    cv::drawKeypoints(k, keypoints, output);
                    tf::imshow("keypoints", output);*/
                    tf::imshow("tag", k);
                }
            }


            /*for (auto& b : GUI::instance()->cache().processed_frame().blobs()) {
                if (b->blob_id() == _blob->blob_id() || (long_t)b->blob_id() == _blob->parent_id) {
                    auto&& [image_pos, image] = b->difference_image(*Tracker::instance()->background(), 0);
                    tf::imshow("blob", image->get());

                }
            //}*/
        }
    }
    
    void Fish::set_data(Individual& obj, Frame_t frameIndex, double time, const EventAnalysis::EventMap *events)
    {
        _safe_frame = obj.find_frame(frameIndex)->frame;
        _time = time;
        _events = events;
        
        if(_frame == frameIndex)
            return;
        
        _path_dirty = true;
        _frame = frameIndex;
        
        _library_y = Graph::invalid();
        _avg_cat = -1;
        _next_frame_cache.valid = false;
        if (_image.source())
            _image.unsafe_get_source().set_index(-1);
        points.clear();
        
        auto seg = _frame.valid() ? obj.segment_for(_frame) : nullptr;
        if(seg) {
            _match_mode = (int)obj.matched_using().at(seg->basic_stuff(_frame)).value();
        } else
            _match_mode = -1;
        
        auto && [basic, posture] = obj.all_stuff(_safe_frame);
        
        const PostureStuff* current_posture;
        const BasicStuff* current_basic;
        
        if(frameIndex == _safe_frame) {
            current_posture = posture;
            current_basic = basic;
        } else {
            auto && [basic, posture] = obj.all_stuff(frameIndex);
            current_posture = posture;
            current_basic = basic;
        }
        
        _cached_outline = current_posture ? current_posture->outline : nullptr;
        _ML = obj.midline_length();
        
        if(GUIOPTION(gui_show_outline) || GUIOPTION(gui_show_midline) || GUIOPTION(gui_happy_mode)) {
            if(current_posture) {
                _cached_midline = SETTING(output_normalize_midline_data) ? obj.fixed_midline(frameIndex) : obj.calculate_midline_for(*current_basic, *current_posture);
                assert(!_cached_midline || _cached_midline->is_normalized());
            }
        }
        
        _view.set_dirty();
        
        auto blob = obj.compressed_blob(_safe_frame);
        _blob_bounds = blob ? blob->calculate_bounds() : _view.bounds();

        //check_tags();
        
        _average_pose = obj.pose_window(frameIndex.try_sub(5_f), frameIndex + 5_f);
        if (not _skelett)
            _skelett = std::make_unique<Skelett>();
        _skelett->set_pose(_average_pose);
        _skelett->set_skeleton(GUI_SETTINGS(meta_skeleton));

        auto [_basic, _posture] = obj.all_stuff(_safe_frame);
        if(_basic)
            _basic_stuff = *_basic;
        else
            _basic_stuff.reset();
        
        if(_posture)
            _posture_stuff = *_posture;
        else
            _posture_stuff.reset();
        
#if !COMMONS_NO_PYTHON
        if(frameIndex == _safe_frame && _basic) {
            auto c = Categorize::DataStore::_label_averaged_unsafe(&obj, Frame_t(frameIndex));
            if(c) {
                _avg_cat = c->id;
                _avg_cat_name = c->name;
            }
        }
        
        auto bdx = _basic_stuff.has_value() ? _basic_stuff->blob.blob_id() : pv::bid();
        _cat = Categorize::DataStore::_label_unsafe(Frame_t(_frame), bdx);
        if (_cat != -1 && _cat != _avg_cat) {
            _cat_name = Categorize::DataStore::label(_cat)->name;
        }
#endif
        
        auto color_source = GUIOPTION(gui_fish_color);
        if(color_source != "identity" && blob) {
            _library_y = Output::Library::get_with_modifiers(color_source, _info, _safe_frame);
            if(!Graph::is_invalid(_library_y)) {
                if(color_source == "X") _library_y /= float(Tracker::average().cols) * FAST_SETTING(cm_per_pixel);
                else if(color_source == "Y") _library_y /= float(Tracker::average().rows) * FAST_SETTING(cm_per_pixel);
            }
        }
        
        _range = Range<Frame_t>(obj.start_frame(), obj.end_frame());
        _empty = obj.empty();
        
        _has_processed_segment = obj.has_processed_segment(_frame);
        if(std::get<0>(_has_processed_segment)) {
            processed_segment = obj.processed_recognition(_frame);
        } else
            processed_segment = {};
        
        _segment = obj.segment_for(_frame);
        if(_segment) {
            _segment = std::make_shared<SegmentInformation>(*_segment);
            _qr_code = obj.qrcode_at(_segment->start());
        } else
            _qr_code = {};
        
        auto pred = Tracker::instance()->find_prediction(_frame, _basic_stuff->blob.blob_id());
        if(pred)
            _pred = *pred;
        else
            _pred.clear();
        
        const auto frame_rate = slow::frame_rate;
        auto current_time = _time;
        auto next_props = Tracker::properties(_frame + 1_f);
        auto next_time = next_props ? next_props->time : (current_time + 1.f/float(frame_rate));
        
        //if(!_next_frame_cache.valid)
        {
            auto result = obj.cache_for_frame(Tracker::properties(_frame), _frame + 1_f, next_time);
            if(result) {
                _next_frame_cache = std::move(result.value());
            } else {
                FormatWarning("Cannot create cache_for_frame of ", _id, " for frame ", _frame + 1_f, " because: ", result.error());
            }
        }
        
        _color = get_color(&_basic_stuff.value());
        updatePath(obj, _safe_frame, cmn::max(obj.start_frame(), _safe_frame.try_sub(1000_f)));
    }
    
    /*void Fish::draw_occlusion(gui::DrawStructure &window) {
        auto &blob = _obj.pixels(_safe_idx);
        window.image(blob_bounds.pos(), _image);
    }*/
    
    void Fish::update(const FindCoord& coord, Entangled& parent, DrawStructure &graph) {
        const auto frame_rate = slow::frame_rate;//FAST_SETTING(frame_rate);
        //const float track_max_reassign_time = FAST_SETTING(track_max_reassign_time);
        const auto single_identity = GUIOPTION(gui_single_identity_color);
        //const auto properties = Tracker::properties(_idx);
        //const auto safe_properties = Tracker::properties(_safe_idx);
        auto &cache = GUICache::instance();
        
        _view.set_bounds(_blob_bounds);
        _label_parent.set_bounds(Bounds(Vec2(0), parent.size()));
        
        const Vec2 offset = -_blob_bounds.pos();
        
        const auto centroid = _basic_stuff.has_value() ? &_basic_stuff->centroid : nullptr;
        const auto head = _posture_stuff.has_value() ? _posture_stuff->head : nullptr;
        
        _fish_pos = centroid ? centroid->pos<Units::PX_AND_SECONDS>() : (_blob_bounds.pos() + _blob_bounds.size() * 0.5);
        
        const bool hovered = _view.hovered();
        const bool timeline_visible = true;//GUICache::exists() && Timeline::visible(); //TODO: timeline_visible
        //const float max_color = timeline_visible ? 255 : GUIOPTION(gui_faded_brightness);
        
        auto base_color = single_identity != Transparent ? single_identity : _id.color();
        //auto clr = base_color.alpha(saturate(((cache.is_selected(_obj.identity().ID()) || hovered) ? max_color : max_color * 0.4f) * time_fade_percent));
        //auto clr = base_color.alpha(saturate(max_color));// * time_fade_percent));
        //_color = clr;
        
        
        auto active = GUICache::instance().active_ids.find(_id.ID()) != GUICache::instance().active_ids.end();
        bool is_selected = cache.is_selected(_id.ID());
        std::vector<Vec2> points;



#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE && false
        if (GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
        {
            if (!is_selected) _view.tag(Effects::blur);
            else _view.untag(Effects::blur);

            if (is_selected && GUI::instance() && GUI::instance()->base()) {
                ((MetalImpl*)((IMGUIBase*)GUI::instance()->base())->platform().get())->center[0] = _view.global_bounds().x / float(GUI::instance()->base()->window_dimensions().width) / gui::interface_scale() * graph.scale().x;
                ((MetalImpl*)((IMGUIBase*)GUI::instance()->base())->platform().get())->center[1] = _view.global_bounds().y / float(GUI::instance()->base()->window_dimensions().height) / gui::interface_scale() * graph.scale().y;
            }
        }
#endif
#endif

        /*if(midline && !midline->is_normalized()) {
            midline = midline->normalize();
        }*/

        if (active && _cached_outline) {
            if (GUIOPTION(gui_show_shadows) || GUIOPTION(gui_show_outline)) {
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
    #if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                if(GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
                {
                    if(is_selected)_polygon->tag(Effects::blur);
                    else _polygon->untag(Effects::blur);
                }
    #endif
    #endif
            }*/
        }

#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
        if constexpr(std::is_same<MetalImpl, default_impl_t>::value) {
            if(GUI_SETTINGS(gui_macos_blur) && is_selected) {
                auto it = cache.fish_selected_blobs.find(_id.ID());
                if (it != cache.fish_selected_blobs.end()) {
                    auto dp = cache.display_blobs.find(it->second);
                    if(dp != cache.display_blobs.end()) {
                        dp->second->ptr->untag(Effects::blur);
                    }
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
        auto section = panic_button ? graph.find("fishbowl") : nullptr;
        if (section) {
            Vec2 mouse_position = graph.mouse_position();
            mouse_position = (mouse_position - section->pos()).div(section->scale());
            mp = mouse_position - _view.pos();
        }

        _posture.update([this, panic_button, mp, &_force, &head, &offset, active, &points](Entangled& window) {
            if (panic_button) {
                if (float(rand()) / float(RAND_MAX) > 0.75) {
                    _color = _wheel.next();
                }

                float r = float(rand()) / float(RAND_MAX);
                _plus_angle += sinf(0.5f * (r - 0.5f) * 2 * float(M_PI) * GUICache::instance().dt());
                window.set_rotation(_plus_angle);

                //r = float(rand()) / float(RAND_MAX);
                //_position += Vec2(r - 0.5, r - 0.5) * 2 * 10 * GUI::cache().dt();


                Vec2 distance = _position - (panic_button == 1 ? mp : Vec2());
                auto CL = distance.length();
                if (std::isnan(CL))
                    CL = 0.0001f;

                const float stiffness = 50, spring_L = panic_button == 1 ? 2 : 0, spring_damping = 20;

                Vec2 f = distance / CL;
                f *= -(stiffness * (CL - spring_L)
                    + spring_damping * _v.dot(distance) / CL);
                _force += f;

                float _mass = 5;
                _v += (_force / _mass + Vec2(0, 9.81 / 0.0025)) * GUICache::instance().dt();

                _position += _v * GUICache::instance().dt();

                window.set_pos(_position);
                window.set_size(_view.size());
            }
            else {
                _position.x = _position.y = 0;
                window.set_pos(_position);
                window.set_rotation(0);
                window.set_size(Size2());
            }

            if (active && _cached_outline && GUIOPTION(gui_show_outline)) {
                std::vector<Vertex> oline;
                points = _cached_outline->uncompress();

                if (GUIOPTION(gui_show_shadows)) {
                    if (!_polygon) {
                        _polygon = std::make_shared<Polygon>(std::make_shared<std::vector<Vec2>>());
                        _polygon->set_fill_clr(Black.alpha(25));
                        _polygon->set_origin(Vec2(0.5));
                    }
                    _polygon->set_vertices(points);
                    float size = Tracker::average().bounds().size().length() * 0.0025f;
                    Vec2 scaling(SQR(offset.x / float(Tracker::average().cols)),
                        SQR(offset.y / float(Tracker::average().rows)));
                    _polygon->set_pos(-offset + scaling * size + _view.size() * 0.5);
                    _polygon->set_scale(scaling * 0.25 + 1);
                    _polygon->set_fill_clr(Black.alpha(25));

                    //window.advance_wrap(*_polygon);
                }

                // check if we actually have a tail index
                if (GUIOPTION(gui_show_midline) && _cached_midline && _cached_midline->tail_index() != -1)
                    window.add<Circle>(Loc(points.at(_cached_midline->tail_index())), Radius{2}, LineClr{Blue.alpha(255 * 0.3f)});

                //float right_side = outline->tail_index() + 1;
                //float left_side = points.size() - outline->tail_index();

                for (size_t i = 0; i < points.size(); i++) {
                    auto& pt = points[i];
                    //Color c = _color;
                    /*if(outline->tail_index() != -1) {
                        float d = cmn::abs(float(i) - float(outline->tail_index())) / ((long_t)i > outline->tail_index() ? left_side : right_side) * 0.4 + 0.5;
                        c = Color(clr.r, clr.g, clr.b, max_color * d);
                    }*/
                    oline.push_back(Vertex(pt, _color));
                }
                oline.push_back(Vertex(points.front(), _color.alpha(255 * 0.04)));
                //auto line =
                window.add<Line>(oline, GUIOPTION(gui_outline_thickness));
                //if(line)
                //    window.text(Meta::toStr(line->points().size()) + "/" + Meta::toStr(oline.size()), Vec2(), White);
                //window.vertices(oline);

            }
            if (active && _cached_midline && GUIOPTION(gui_show_midline)) {
                std::vector<MidlineSegment> midline_points;
                //Midline _midline(*_cached_midline);
                //float len = _obj.midline_length();

                //if(len > 0)
                //    _midline.fix_length(len);

                auto& _midline = *_cached_midline;
                midline_points = _midline.segments();

                std::vector<Vertex> line;
                auto tf = _cached_midline->transform(default_config::individual_image_normalization_t::none, true);

                for (auto &segment : midline_points) {
                    //Vec2 pt = segment.pos;
                    //float angle = _cached_midline->angle() + M_PI;
                    //float x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
                    //float y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));

                    //pt = Vec2(x, y) + _cached_midline->offset();
                    line.push_back(Vertex(tf.transformPoint(segment.pos), _color));
                }

                window.add<Line>(line, GUIOPTION(gui_outline_thickness));
                //window.vertices(line);

                if (head) {
                    window.add<Circle>(Loc(head->pos<Units::PX_AND_SECONDS>() + offset), Radius{3}, LineClr{Red.alpha(255)});
                }
            }
            });
        
        
        //! this does not work since update() is called within before_draw :S
        //! this function is not supposed to do what its doing apparently
        //! need to overwrite update() method?
        //if(_view.content_changed())
        //if(_path_dirty)
        _view.update([&](Entangled&) {
            if(not _basic_stuff.has_value())
				return;

            _path_dirty = false;
            
            if (GUIOPTION(gui_show_paths)) {
                for(auto& p : _paths)
                    _view.advance_wrap(*p);
                //paintPath(offset);
            }

            if (FAST_SETTING(track_max_individuals) > 0 && GUIOPTION(gui_show_boundary_crossings))
                update_recognition_circle();

            constexpr const char* animator = "panic-button-animation";
            if(panic_button) {
                _view.add<Line>(_posture.pos(), mp, White.alpha(50));
                GUICache::instance().set_animating(animator, true);
            } else
                GUICache::instance().set_animating(animator, false);
        
            _view.advance_wrap(_posture);
        
            // DISPLAY LABEL AND POSITION
            auto c_pos = centroid->pos<Units::PX_AND_SECONDS>() + offset;
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
        
            auto radius = (slow::calculate_posture && _ML != Graph::invalid() ? _ML : _blob_bounds.size().max()) * 0.6;
            if(GUIOPTION(gui_show_texts)) {
                if(_next_frame_cache.valid) {
                    auto estimated = _next_frame_cache.estimated_px + offset;
                    
                    _view.add<Circle>(Loc(c_pos), Radius{2}, LineClr{White.alpha(255)});
                    _view.add<Line>(c_pos, estimated, _color);
                    _view.add<Circle>(Loc(estimated), Radius{2}, LineClr{Transparent}, FillClr{_color});
                }
            }
        
            if(GUIOPTION(gui_happy_mode) && _cached_midline && _cached_outline && _posture_stuff && _posture_stuff->head) {
                struct Physics {
                    Vec2 direction = Vec2();
                    Vec2 v = Vec2();
                    Frame_t frame = {};
                    double blink = 0;
                    bool blinking = false;
                    double blink_limit = 10;
                };
            
                constexpr double damping_linear = .5;
                constexpr float stiffness = 100, spring_L = 0, spring_damping = 1;
                static std::unordered_map<Idx_t, Physics> _current_angle;
                auto &ph = _current_angle[_id.ID()];
                double dt = GUICache::instance().dt();
                if(dt > 0.1)
                    dt = 0.1;
            
                Vec2 force = ph.v * (-damping_linear);
            
                if(ph.frame != _frame) {
                    ph.direction += 0; // rad/s
                    ph.frame = _frame;
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
            
                if (_basic_stuff.has_value()) {
                    auto&& [eyes, off] = VisualField::generate_eyes(_frame, _id.ID(), *_basic_stuff, points, _cached_midline, alpha);

                    auto d = ph.direction;
                    auto L = d.length();
                    if (L > 0) d /= L;
                    if (L > 1) L = 1;
                    d *= L;

                    double h = ph.blink / 0.1;

                    if (h > ph.blink_limit && !ph.blinking) {
                        ph.blinking = true;
                        ph.blink = 0;
                        h = 0;
                        ph.blink_limit = rand() / double(RAND_MAX) * 30;
                    }

                    if (h > 1 && ph.blinking) {
                        ph.blinking = false;
                    }

                    ph.blink += dt;

                    auto sun_direction = (offset - Vec2(0)).normalize();
                    auto eye_scale = max(0.5, _ML / 90);
                    for (auto& eye : eyes) {
                        eye.pos += ph.direction;
                        _view.add<Circle>(Loc(eye.pos + offset), Radius(5 * eye_scale), LineClr{Black.alpha(200)}, FillClr{White.alpha(125)});
                        auto c = _view.add<Circle>(Loc(eye.pos + Vec2(2.5).mul(d * eye_scale) + offset), Radius(3 * eye_scale), LineClr{Transparent}, FillClr{Black.alpha(200)});
                        c->set_scale(Vec2(1, ph.blinking ? h : 1));
                        c->set_rotation(atan2(ph.direction) + RADIANS(90));//posture->head->angle() + RADIANS(90));
                        _view.add<Circle>(Loc(eye.pos + Vec2(2.5).mul(d * eye_scale) + Vec2(2 * eye_scale).mul(sun_direction) + offset), Radius(sqrt(eye_scale)), LineClr{Transparent}, FillClr{White.alpha(200 * c->scale().min())});
                    }
                }
            }

            auto color_source = GUIOPTION(gui_fish_color);
            if(color_source == "viridis") {
                GUICache::instance().processed_frame().transform_blobs([&, bdx = _basic_stuff->blob.blob_id(), pdx = _basic_stuff->blob.parent_id](const pv::Blob& b)
                {
                    if(!is_in(b.blob_id(), bdx, pdx))
                        return true;
                    
                    auto && [dpos, difference] = b.difference_image(*Tracker::instance()->background(), 0);
                    auto rgba = Image::Make(difference->rows, difference->cols, 4);
                
                    uchar maximum_grey = 255, minimum_grey = 0;
                    
                    auto ptr = rgba->data();
                    auto m = difference->data();
                    for(; ptr < rgba->data() + rgba->size(); ptr += rgba->dims, ++m) {
                        auto c = Viridis::value((float(*m) - minimum_grey) / (maximum_grey - minimum_grey));
                        *(ptr) = c.r;
                        *(ptr+1) = c.g;
                        *(ptr+2) = c.b;
                        *(ptr+3) = *m;
                    }
                
                    _view.add<ExternalImage>(std::move(rgba), dpos + offset);
                    
                    return false;
                });
            
            } else if(not Graph::is_invalid(_library_y)) {
                auto percent = min(1.f, cmn::abs(_library_y));
                Color clr = /*Color(225, 255, 0, 255)*/ base_color * percent + Color(50, 50, 50, 255) * (1 - percent);
            
                GUICache::instance().processed_frame().transform_blobs([&, bdx = _basic_stuff->blob.blob_id(), pdx = _basic_stuff->blob.parent_id](const pv::Blob& b)
                {
                    if(!is_in(b.blob_id(), bdx, pdx))
                        return true;
                    
                    auto && [image_pos, image] = b.binary_image(*Tracker::instance()->background(), FAST_SETTING(track_threshold));
                    auto && [dpos, difference] = b.difference_image(*Tracker::instance()->background(), 0);
                
                    auto rgba = Image::Make(image->rows, image->cols, 4);
                
                    uchar maximum = 0;
                    for(size_t i=0; i<difference->size(); ++i) {
                        maximum = max(maximum, difference->data()[i]);
                    }
                    for(size_t i=0; i<difference->size(); ++i)
                        difference->data()[i] = (uchar)min(255, float(difference->data()[i]) / maximum * 255);
                
                    rgba->set_channels(image->data(), {0, 1, 2});
                    rgba->set_channel(3, difference->data());
                
                    _view.add<ExternalImage>(std::move(rgba), image_pos + offset, Vec2(1), clr);
                
                    return false;
                });
            }
        
            
            if(is_selected && GUIOPTION(gui_show_probabilities)) {
                auto c = cache.processed_frame().cached(_id.ID());
                if (c) {
                    auto &mat = _image.unsafe_get_source();
                    if(mat.index() != _frame.get()) {
                        mat.create(coord.video_size().height, coord.video_size().width, 4);
                        mat.set_index(_frame.get());

                        if(_probability_radius < 10 || _probability_center.empty())
                            mat.set_to(0);
                        else {
                            for (int y = max(0, _probability_center.y - _probability_radius - 1); y < min((float)mat.rows - 1, _probability_center.y + _probability_radius); ++y) {
                                auto ptr = mat.ptr(y, max(0, _probability_center.x - _probability_radius - 1));
                                auto end = mat.ptr(y, min((float)mat.cols - 1, _probability_center.x + _probability_radius + 1));
                                if (end > mat.data() + mat.size())
                                    throw U_EXCEPTION("Mat end ", mat.size(), " end: ", uint64_t(end - mat.data()));
                                std::fill(ptr, end, 0);
                            }
                        }
                        
                        _probability_center = c->estimated_px;
                        float sum;
                        _probability_radius = 0;

                        LockGuard guard(ro_t{}, "Fish::update::probs");
                        
                        auto plot = [&](int x, int y) {
                            Vec2 pos = _probability_center + Vec2(x, y);
                            if (pos.x < 0 || pos.x >= mat.cols)
                                return;
                            if (pos.y < 0 || pos.y >= mat.rows)
                                return;
                            if (_frame <= _range.start)
                                return;

                            auto ptr = mat.ptr(pos.y, pos.x);
                            auto p = 0;//_obj.probability(-1, *c, _frame, pos + 1 * 0.5, 1); //TODO: add probabilities
                            if (p/*.p*/ < FAST_SETTING(matching_probability_threshold))
                                return;

                            auto clr = Viridis::value(p).alpha(uint8_t(min(255, 255.f * p)));
                            *(ptr + 0) = clr.r;
                            *(ptr + 1) = clr.g;
                            *(ptr + 2) = clr.b;
                            *(ptr + 3) = clr.a;
                            sum += p/*.p*/;
                        };
                        
                        do {
                            sum = 0;
                            int r = _probability_radius;
                            
                            for (int y = 0; y < _probability_radius; ++y) {
                                int x = std::sqrt(r * r - y * y);
                                plot(x, y);
                                plot(-x, y);
                                
                                plot(x,  -y);
                                plot(-x, -y);
                            }
                            
                            ++_probability_radius;
                            
                        } while (sum > 0 || _probability_radius < 10);

                        for (int y = max(0, _probability_center.y - _probability_radius - 1); y < min((float)mat.rows - 1, _probability_center.y + _probability_radius); ++y) {
                            int x = max(0, _probability_center.x - _probability_radius - 1);
                            auto ptr = mat.ptr(y, x);
                            auto end = mat.ptr(y, min((float)mat.cols - 1, _probability_center.x + _probability_radius + 1));
                            
                            for (; ptr != end; ++ptr, ++x) {
                                //if (*(ptr) <= 5)
                                    plot(x - _probability_center.x, y - _probability_center.y);
                            }
                        }

                        _image.updated_source();
                    }
                }

                _image.set_pos(offset);
                _view.advance_wrap(_image);
            }
            else if(!_image.source()->empty()) {
                _image.set_source(std::make_unique<Image>());
            }
        
            if ((hovered || is_selected) && GUIOPTION(gui_show_selections)) {
                Color circle_clr = Color(v).alpha(saturate(255 * (hovered ? 1.7 : 1)));
                if(cache.primary_selected_id() != _id.ID())
                    circle_clr = Gray.alpha(circle_clr.a);
            
                // draw circle around the fish
            
            
                _circle.set_pos(c_pos);
                _circle.set_radius(radius);
                _circle.set_origin(Vec2(0.5));
                _circle.set_line_clr(circle_clr);
                _circle.set_fill_clr(hovered ? White.alpha(circle_clr.a * 0.1) : Transparent);
                //_circle.set_scale(Vec2(1));
                _view.advance_wrap(_circle);
                
                auto &gb = _circle.global_bounds();
                if(gb.width < 15) {
                    _circle.set_scale(_circle.scale().mul(15 / gb.width));
                }
            
                //window.circle(c_pos, radius, circle_clr, hovered ? White.alpha(circle_clr.a * 0.1) : Transparent);
            
                // draw unit circle showing the angle of the fish
                Loc pos(cmn::cos(angle), -cmn::sin(angle));
                pos = pos * radius + c_pos;
            
                _view.add<Circle>(pos, Radius{3}, LineClr{circle_clr});
                _view.add<Line>(c_pos, pos, circle_clr);
            
                if(FAST_SETTING(posture_direction_smoothing)) {
                    std::map<Frame_t, float> angles;
                    std::map<Frame_t, float> dangle, ddangle, interp;
                
                    float previous = FLT_MAX;
                    bool hit = false;
                    float value = 0;
                    size_t count_ones = 0;
                
                    for (auto frame = _frame - Frame_t(FAST_SETTING(posture_direction_smoothing)); frame <= _frame + Frame_t(FAST_SETTING(posture_direction_smoothing)); ++frame)
                    {
                        auto midline = _pp_midline;
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
                        if(frame == _frame) {
                            _graph.set_title(Meta::toStr(ddangle.count(frame) ? ddangle.at(frame) : FLT_MAX) + " " +Meta::toStr(in));
                        }
                    }
                
                    _graph.clear();
                    _graph.set_pos(c_pos + Vec2(radius, radius));
                
                    auto first_frame = interp.empty() ? 0_f : interp.begin()->first;
                    auto last_frame = interp.empty() ? 0_f : interp.rbegin()->first;
                    _graph.set_ranges(Rangef(first_frame.get(), last_frame.get()), Rangef(-1, 1));
                
                    std::vector<Vec2> points;
                    for(auto && [frame, a] : dangle) {
                        points.push_back(Vec2(frame.get(), a));
                    }
                    _graph.add_points("angle'", points);
                
                    points.clear();
                    for(auto && [frame, a] : ddangle) {
                        points.push_back(Vec2(frame.get(), a));
                    }
                    _graph.add_points("angle''", points);
                    
                    points.clear();
                    for(auto && [frame, a] : angles) {
                        points.push_back(Vec2(frame.get(), a));
                    }
                    _graph.add_points("angle", points);
                    
                    _graph.set_zero(_frame.get());
                
                    _view.advance_wrap(_graph);
                }
            }
            
        });

        
        parent.advance_wrap(_view);
        
        if(_basic_stuff.has_value() && _basic_stuff->blob.pred.pose.size() > 0) {
            auto & pred =  _basic_stuff->blob.pred;
            for(size_t i=0; i<pred.pose.size(); ++i) {
                auto bone = pred.pose.bone(i);
                if((bone.A.x != 0 || bone.A.y != 0)
                    && (bone.B.x != 0 || bone.B.y != 0)) 
                {
                    parent.add<Circle>(Loc{bone.A}, FillClr{Gray.alpha(25)}, LineClr{Gray.alpha(25)}, Radius{5});
                    parent.add<Line>(Loc{bone.A}, Loc{bone.B}, LineClr{Gray.alpha(50)});
                }
            }

            ColorWheel wheel;
            for(size_t i=0; i<_average_pose.size(); ++i) {
                auto bone = _average_pose.bone(i);
                if ((bone.A.x != 0 || bone.A.y != 0)
                    && (bone.B.x != 0 || bone.B.y != 0))
                {
                    auto c = Color::blend(_color.alpha(125), wheel.next().alpha(130));
                    parent.add<Circle>(Loc{bone.A}, FillClr{c.alpha(75)}, LineClr{c.alpha(150)}, Radius{5});
                    parent.add<Line>(Loc{bone.A}, Loc{bone.B}, LineClr{c.alpha(75)},2);
                }
            }

            if(_skelett)
                parent.advance_wrap(*_skelett);
        }

        parent.advance_wrap(_label_parent);
        _label_parent.update([&](auto&){
            label(coord, _label_parent);
        });
        
        //static auto change = parent.children();
        /*if(parent.children().size() != change.size()) {
            print("_view:");
            for(auto c : parent.children()) {
                auto name = c->toStr();
                print("\t", name);
            }
            print("--");
            change = parent.children();
        }*/
    }

Color Fish::get_color(const BasicStuff * basic) const {
    if(not basic)
        return Transparent;
    
    const auto single_identity = GUIOPTION(gui_single_identity_color);
    auto base_color = single_identity != Transparent ? single_identity : _id.color();
    
    auto color_source = GUIOPTION(gui_fish_color);
    auto clr = base_color.alpha(255);
    if(single_identity.a != 0) {
        clr = single_identity;
    }
    
    if(color_source != "identity") {
        auto y = Output::Library::get_with_modifiers(color_source, _info, _safe_frame);
        
        if(not Graph::is_invalid(y)) {
            if(color_source == "X")
                y /= float(Tracker::average().cols) * slow::cm_per_pixel; //FAST_SETTING(cm_per_pixel);
            else if(color_source == "Y")
                y /= float(Tracker::average().rows) * slow::cm_per_pixel; //FAST_SETTING(cm_per_pixel);
            
            auto percent = saturate(cmn::abs(y), 0.f, 1.f);
            return clr.alpha(255) * percent + Color(50, 50, 50, 255) * (1 - percent);
        }
    } else
        return base_color;
    
    return Transparent;
}

void Fish::updatePath(Individual& obj, Frame_t to, Frame_t from) {
    if (!to.valid())
        to = _range.end;
    if (!from.valid())
        from = _range.start;
    
    from = _empty ? _frame : _range.start;
    to = _empty ? _frame : min(_range.end, _frame);
        
    _color_start = max(sign_cast<int64_t>(_range.start.get()), round(sign_cast<int64_t>(_frame.get()) - FAST_SETTING(frame_rate) * GUIOPTION(gui_max_path_time)));
    _color_end = max(_color_start + 1, (float)_frame.get());
    
    from = max(Frame_t(sign_cast<Frame_t::number_t>(_color_start)), from);
    
    if(_prev_frame_range.start != _range.start
       || _prev_frame_range.end > _range.end)
    {
        frame_vertices.clear();
    }
    
    _prev_frame_range = _range;
    
    const float max_speed = FAST_SETTING(track_max_speed);
    const float thickness = GUIOPTION(gui_outline_thickness);
    
    auto first = frame_vertices.empty() ? Frame_t() : frame_vertices.begin()->frame;
    
    if(first.valid() && first < from && !frame_vertices.empty()) {
        auto it = frame_vertices.begin();
        while (it != frame_vertices.end() && it->frame < from)
            ++it;
        
        //auto end = it != frame_vertices.begin() ? it-1 : it;
        
        frame_vertices.erase(frame_vertices.begin(), it);
        first = frame_vertices.empty() ? Frame_t() : frame_vertices.begin()->frame;
    }
    
    
    if(not first.valid()
       || first > from)
    {
        auto i = (first.valid() ? first - 1_f : from);
        auto fit = obj.iterator_for(i);
        auto end = obj.frame_segments().end();
        auto begin = obj.frame_segments().begin();
        //auto seg = _obj.segment_for(i);
        
        for (; i.valid() && i>=from; --i) {
            if(fit == end || (*fit)->start() > i) {
                while(fit != begin && (fit == end || (*fit)->start() > i))
                {
                    --fit;
                }
            }
            
            if(fit != end && (*fit)->contains(i)) {
                auto id = (*fit)->basic_stuff(i);
                if(id != -1) {
                    auto &stuff = obj.basic_stuff()[id];
                    frame_vertices.push_front(FrameVertex{
                        .frame = i,
                        .vertex = Vertex(stuff->centroid.pos<Units::PX_AND_SECONDS>(), get_color(stuff.get())),
                        .speed_percentage = min(1, stuff->centroid.speed<Units::CM_AND_SECONDS>() / max_speed)
                    });
                }
            }
        }
        
        first = frame_vertices.empty() ? Frame_t() : frame_vertices.begin()->frame;
    }
    
    auto last = frame_vertices.empty() ? Frame_t() : frame_vertices.rbegin()->frame;
    if(!last.valid())
        last = from;
    
    if(last > to && !frame_vertices.empty()) {
        auto it = --frame_vertices.end();
        while(it->frame > to && it != frame_vertices.begin())
            --it;
        
        
        frame_vertices.erase(it, frame_vertices.end());
    }
    
    last = frame_vertices.empty() ? Frame_t() : frame_vertices.rbegin()->frame;
    
    if(not last.valid()
       || last < to)
    {
        auto i = last.valid() ? max(from, last) : from;
        auto fit = obj.iterator_for(i);
        auto end = obj.frame_segments().end();
        
        for (; i<=to; ++i) {
            if(fit == end || (*fit)->end() < i) {
                //seg = _obj.segment_for(i);
                while(fit != end && (*fit)->end() < i)
                    ++fit;
            }
            
            if(fit != end && (*fit)->contains(i)) {
                auto id = (*fit)->basic_stuff(i);
                if(id != -1) {
                    auto &stuff = obj.basic_stuff()[id];
                    frame_vertices.push_back(FrameVertex{
                        .frame = i,
                        .vertex = Vertex(stuff->centroid.pos<Units::PX_AND_SECONDS>(), get_color(stuff.get())),
                        .speed_percentage = min(1, stuff->centroid.speed<Units::CM_AND_SECONDS>() / max_speed)
                    });
                }
            }
        }
        
        last = frame_vertices.empty() ? Frame_t() : frame_vertices.rbegin()->frame;
    }
    
    const Vec2 offset = -_blob_bounds.pos();
    paintPath(offset);
}
    
    void Fish::paintPath(const Vec2& offset) {
        //const float max_speed = FAST_SETTING(track_max_speed);
        const float thickness = GUIOPTION(gui_outline_thickness);
        
        /*auto clr = base_color.alpha(255);
        if(!Graph::is_invalid(_library_y)) {
            const auto single_identity = GUIOPTION(gui_single_identity_color);
            auto percent = min(1.f, cmn::abs(_library_y));
            if(single_identity.a != 0) {
                clr = single_identity;
            }
            
            clr = clr.alpha(255) * percent + Color(50, 50, 50, 255) * (1 - percent);
            _previous_color = clr;
        } else {
            clr = _previous_color;
        }
        auto inactive_clr = clr.saturation(0.5);
        Color use = clr;*/
        
        ///TODO: could try to replace vertices 1by1 and get "did change" for free, before we even
        ///      try to update the object.
        const float max_distance = SQR(Individual::weird_distance() * 0.1 / slow::cm_per_pixel);
        size_t paths_index = 0;
        _vertices.clear();
        _vertices.reserve(frame_vertices.size());

        auto prev = frame_vertices.empty() ? Frame_t() : frame_vertices.begin()->frame;
        Vec2 prev_pos = frame_vertices.empty() ? Vec2(-1) : frame_vertices.begin()->vertex.position();
        for(auto & fv : frame_vertices) {
            float percent = (fv.speed_percentage * 0.15 + 0.85) * (float(fv.frame.get() - _color_start) / float(_color_end - _color_start));
            
            assert(fv.speed_percentage >= 0 && fv.speed_percentage <= 1);
            assert(fv.frame.get() >= _color_start);
            assert(_color_end - _color_start > 0);
            assert(percent >= 0 && percent <= 1);
            percent = percent * percent;
            
            if(fv.frame - prev > 1_f || (prev.valid() && sqdistance(prev_pos, fv.vertex.position()) >= max_distance)) {
                //use = inactive_clr;
                if(_vertices.size() > 1) {
                    if (_paths.size() <= paths_index) {
                        _paths.emplace_back(std::make_unique<Vertices>(_vertices, PrimitiveType::LineStrip, Vertices::COPY));
                        _paths[paths_index]->set_thickness(thickness);
                        //_view.advance_wrap(*_paths[paths_index]);
                    } else {
                        auto& v = _paths[paths_index];
                        if(v->change_points() != _vertices) {
                            std::swap(v->change_points(), _vertices);
                            v->confirm_points();
                        }
                        //_view.advance_wrap(*v);
                    }

                    ++paths_index;
                }
                    //window.vertivertices, 2.0f);
                _vertices.clear();
                
                //window.circle(fv.vertex.position() + offset, 1, White.alpha(percent * 255));
            } //else
               // use = clr;
            prev = fv.frame;
            prev_pos = fv.vertex.position();
            _vertices.emplace_back(fv.vertex.position() + offset, fv.vertex.clr().alpha(percent * 255));
        }
        
        
        if (_paths.size() <= paths_index) {
            _paths.emplace_back(std::make_unique<Vertices>(_vertices, PrimitiveType::LineStrip, Vertices::COPY));
            _paths[paths_index]->set_thickness(thickness);
            //_view.advance_wrap(*_paths[paths_index]);
        }
        else {
            auto& v = _paths[paths_index];
            if(v->change_points() != _vertices) {
                std::swap(v->change_points(), _vertices);
                v->confirm_points();
                //_view.advance_wrap(*v);
            }
        }
        if(paths_index + 1 < _paths.size())
            _paths.resize(paths_index + 1);

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
        
        MotionRecord* prev_centroid = NULL;
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
                
                //float distance = euclidean_distance(pos * FAST_SETTING(cm_per_pixel), vertices.back().position() * FAST_SETTING(cm_per_pixel));
                if(!vertices.empty() && (!prev_centroid || prev_centroid->time() - c->time() >= FAST_SETTING(track_max_reassign_time) * 0.5))
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
                throw U_EXCEPTION("Unknown type.");
        }*/
    }
    
    void Fish::update_recognition_circle() {
        if(Tracker::instance()->border().in_recognition_bounds(_fish_pos)) {
            if(!_recognition_circle) {
                // is inside bounds, but we didnt know that yet! start animation
                _recognition_circle = std::make_shared<Circle>(Radius{1}, LineClr{Transparent}, FillClr{Cyan.alpha(50)});
            }
            
            auto ts = GUICache::instance().dt();
            float target_radius = 100;
            float percent = min(1, _recognition_circle->radius() / target_radius);
            
            if(percent < 1.0) {
                percent *= percent;
                
                _recognition_circle->set_pos(_fish_pos - _view.pos());
                _recognition_circle->set_radius(_recognition_circle->radius() + ts * (1 - percent) * target_radius * 2);
                _recognition_circle->set_fill_clr(Cyan.alpha(50 * (1-percent)));
                GUICache::instance().set_animating(circle_animator, true);
                
                _view.advance_wrap(*_recognition_circle);
                
            } else {
                GUICache::instance().set_animating(circle_animator, false);
            }
            
        } else if(_recognition_circle) {
            _recognition_circle = nullptr;
        }
    }

void Fish::label(const FindCoord& coord, Entangled &e) {
    if(GUIOPTION(gui_highlight_categories)) {
        if(_avg_cat != -1) {
            e.add<Circle>(Loc(_view.pos() + _view.size() * 0.5),
                          Radius{_view.size().length()},
                          LineClr{Transparent},
                          FillClr{ColorWheel(_avg_cat).next().alpha(75)});
        } else {
            /*e.add<Circle>(Loc(_view.pos() + _view.size() * 0.5),
                          Radius{_view.size().length()},
                          LineClr{Transparent},
                          FillClr{Purple.alpha(15)});*/
        }
    }
    
    if(GUIOPTION(gui_show_match_modes)) {
        e.add<Circle>(Loc(_view.pos() + _view.size() * 0.5),
                      Radius{_view.size().length()},
                      LineClr{Transparent},
                      FillClr{ColorWheel(_match_mode).next().alpha(50)});
    }
    
    //auto bdx = blob->blob_id();
    if(GUIOPTION(gui_show_cliques)) {
        uint32_t i=0;
        for(auto &clique : GUICache::instance()._cliques) {
            if(clique.fishs.contains(_id.ID())) {
                e.add<Circle>(Loc(_view.pos() + _view.size() * 0.5),
                              Radius{_view.size().length()},
                              LineClr{Transparent},
                              FillClr{ColorWheel(i).next().alpha(50)});
                break;
            }
            ++i;
        }
    }
    
    if (!GUIOPTION(gui_show_texts))
        return;
    
    if(!_basic_stuff.has_value())
        return;
    
    std::string color = "";
    std::stringstream text;
    std::string secondary_text;

    text << _id.raw_name() << " ";
    
    /*if (DrawMenu::matching_list_open() && blob) {
        secondary_text = "blob" + Meta::toStr(blob->blob_id());
    }
    else*/ if (GUI_SETTINGS(gui_show_recognition_bounds)) {
        auto& [valid, segment] = _has_processed_segment;
        if (valid) {
            auto& [samples, map] = processed_segment;
            auto it = std::max_element(map.begin(), map.end(), [](const std::pair<Idx_t, float>& a, const std::pair<Idx_t, float>& b) {
                return a.second < b.second;
            });

            if (it == map.end() || it->first != _id.ID()) {
                color = "str";
                secondary_text += " avg" + Meta::toStr(it->first);
            }
            else
                color = "nr";
        } else {
            if(not _pred.empty()) {
                auto map = Tracker::prediction2map(_pred);
                auto it = std::max_element(map.begin(), map.end(), [](const std::pair<Idx_t, float>& a, const std::pair<Idx_t, float>& b) {
                        return a.second < b.second;
                    });

                if (it != map.end()) {
                    secondary_text += " loc" + Meta::toStr(it->first) + " (" + dec<2>(it->second).toStr() + ")";
                }
            }
        }
    }
    //auto raw_cat = Categorize::DataStore::label(Frame_t(_idx), blob);
    //auto cat = Categorize::DataStore::label_interpolated(_obj.identity().ID(), Frame_t(_idx));

#if !COMMONS_NO_PYTHON
    auto detection = tags::find(_frame, _basic_stuff->blob.blob_id());
    if (detection.valid()) {
        secondary_text += "<a>tag:" + Meta::toStr(detection.id) + " (" + dec<2>(detection.p).toStr() + ")</a>";
    }
    
    if(_segment) {
        auto [id, p, n] = _qr_code;
        if(id >= 0 && p > 0) {
            secondary_text += std::string(" ") + "<a><i>QR:"+Meta::toStr(id)+" ("+dec<2>(p).toStr() + ")</i></a>";
        }
    }
    
    auto c = GUICache::instance().processed_frame().cached(_id.ID());
    if(c) {
        auto cat = c->current_category;
        if(cat != -1 && cat != _avg_cat) {
            secondary_text += std::string(" ") + "<key>"+_cat_name+"</key>";
        }
    }
    
    auto bdx = _basic_stuff.has_value() ? _basic_stuff->blob.blob_id() : pv::bid();
    if (_cat != -1 && _cat != _avg_cat) {
        secondary_text += std::string(" ") + (_cat ? "<b>" : "") + "<i>" + _cat_name + "</i>" + (_cat ? "</b>" : "");
    }
    
    if(_avg_cat != -1) {
        secondary_text += (_avg_cat != -1 ? std::string(" ") : std::string()) + "<nr>" + _avg_cat_name + "</nr>";
    }
#endif
    
    auto label_text = (color.empty() ? text.str() : ("<"+color+">"+text.str()+"</"+color+">")) + "<a>" + secondary_text + "</a>";
    if(_basic_stuff.has_value()) {
        if (!_label) {
            _label = new Label(label_text, _basic_stuff->blob.calculate_bounds(), fish_pos());
        }
        else
            _label->set_data(this->frame(), label_text, _basic_stuff->blob.calculate_bounds(), fish_pos());
    }

    e.advance_wrap(*_label);
    _label->update(coord, 1, 0, not _basic_stuff.has_value(), GUICache::instance().dt());
}

Drawable* Fish::shadow() {
    auto active = GUICache::instance().active_ids.find(_id.ID()) != GUICache::instance().active_ids.end();
    
    if(GUIOPTION(gui_show_shadows) && active) {
        if(!_polygon) {
            _polygon = std::make_shared<Polygon>(std::make_shared<std::vector<Vec2>>());
            _polygon->set_fill_clr(Black.alpha(125));
            _polygon->set_origin(Vec2(0.5));
        }
        
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
        if (GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
        {
            auto is_selected = GUICache::instance().is_selected(_id.ID());
            if (!is_selected) _polygon->tag(Effects::blur);
            else _polygon->untag(Effects::blur);
        }
#endif
#endif
        return _polygon.get();//window.wrap_object(*_polygon);
    }
    return nullptr;
}
}
