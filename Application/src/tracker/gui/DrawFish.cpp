#include "DrawFish.h"
#include <gui/DrawSFBase.h>
#include <tracking/OutputLibrary.h>
#include <tracking/Individual.h>
#include <tracking/VisualField.h>
#include <misc/CircularGraph.h>
#include <misc/create_struct.h>
#include <gui/Label.h>
#include <ml/Categorize.h>
#include <gui/IMGUIBase.h>
#include <gui/DrawBase.h>
#include <tracking/DetectTag.h>
#include <gui/GUICache.h>
//#include <gui.h>
#include <misc/IdentifiedTag.h>
#include <gui/Skelett.h>
#include <tracking/Individual.h>
#include <processing/Background.h>
#include <misc/CircularGraph.h>
#include <gui/DynamicGUI.h>
#include <gui/dyn/Action.h>
#include <gui/dyn/Context.h>
#include <gui/dyn/ParseText.h>

#if defined(__APPLE__) && defined(TREX_ENABLE_EXPERIMENTAL_BLUR)
//#include <gui.h>
#endif

using namespace track;

namespace cmn::gui {

struct Fish::Data {
    dyn::Context context;
    std::string label_text;
};

Fish::~Fish() {
    if (_label) {
        delete _label;
    }
}

    Fish::Fish(Individual& obj)
        :   _id(obj.identity()),
            _info(&obj, Output::Options_t{})
    {
        _data = std::make_unique<Data>();
        _data->context = [&](){
            using namespace dyn;
            Context context;
            context.actions = {
            };
/// {name}{if:{not:{has_pred}}:' {max_pred}':''}
            context.variables = {
                VarFunc("help", [this](const VarProps&) -> std::string {
                    return "The following variables are available:\n"+Meta::toStr(extract_keys(_data->context.variables));
                }),
                VarFunc("hovered", [this](const VarProps&) -> bool {
                    return _tight_selection.hovered();
                }),
                VarFunc("window_size", [](const VarProps&) -> Vec2 {
                    return FindCoord::get().screen_size();
                }),
                VarFunc("has_pred", [this](const VarProps&) {
                    return _raw_preds.has_value();
                }),
                VarFunc("max_pred", [this](const VarProps&) -> std::pair<Idx_t, float>{
                    //Print("max_pred: ", no_quotes(_raw_preds ? Meta::toStr(_raw_preds.value()) : "null"));
                    if(_raw_preds.has_value()) {
                        auto m = max_element(_raw_preds.value());
                        if(m)
                            return *m;
                    }
                    
                    return {};
                }),
                VarFunc("predictions", [this](const VarProps&) -> std::vector<float> {
                    std::vector<float> r;
                    for(auto &[k, v] : _raw_preds.value()) {
                        r.push_back(v);
                    }
                    return r;
                }),
                VarFunc("name", [this](const VarProps&) {
                    return _id.raw_name();
                }),
                VarFunc("id", [this](const VarProps&) {
                    return _id.ID();
                }),
                VarFunc("bdx", [this](const VarProps&) {
                    return _basic_stuff ? _basic_stuff->blob.blob_id() : pv::bid();
                }),
                VarFunc("tag", [this](const VarProps&) -> sprite::Map {
                    if(not _basic_stuff)
                        return {};
                        //throw InvalidArgumentException("Invalid frame, no data available.");
                    auto bdx = _basic_stuff->blob.blob_id();
                    auto detection = tags::find(_frame, bdx);
                    if(not detection.valid())
                        return {};
                    
                    sprite::Map map;
                    map["id"] = detection.id;
                    map["p"] = detection.p;
                    return map;
                }),
                VarFunc("qr", [this](const VarProps&) -> sprite::Map {
                    if(not _tracklet) {
                        return {};//throw InvalidArgumentException("No tracklet set to retrieve QRCode from.");
                    }
                    
                    auto [id, p, n] = _qr_code;
                    sprite::Map map;
                    map["id"] = id;
                    map["p"] = p;
                    map["n"] = n;
                    return map;
                }),
                VarFunc("category", [this](const VarProps&) {
                    return _cat_name;
                }),
                
                VarFunc("average_category", [this](const VarProps&) {
                    return _avg_cat_name;
                })
            };

            return context;
        }();
        
        _previous_color = obj.identity().color();
        
        assert(_id.ID().valid());
        //auto ID = _id.ID();
        //_view.set_clickable(true);
        //_circle.set_clickable(true);
        _posture.set_origin(Vec2(0));
        _view.set_name(_id.name());
        
        _tight_selection.set_clickable(true);
        _tight_selection.on_hover([this](Event e) {
            selection_hovered(e);
        });
        _tight_selection.on_click([this](Event e) {
            selection_clicked(e);
        });
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
                GUICache::instance().background()->image());

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
                    auto&& [image_pos, image] = b->difference_image(*Tracker::background(), 0);
                    tf::imshow("blob", image->get());

                }
            //}*/
        }
    }

#define OPTION(NAME) _options. NAME
    
    void Fish::set_data(const UpdateSettings& options, Individual& obj, Frame_t frameIndex, double time, const EventAnalysis::EventMap *events)
    {
        _options = options;
        _safe_frame = frameIndex;
        _time = time;
        _events = events;
        
        //if(_frame == frameIndex)
        //    return;
        
        _path_dirty = true;
        if(_frame != frameIndex) {
            _frame_change.reset();
        }
        _frame = frameIndex;
        
        _library_y = GlobalSettings::invalid();
        _avg_cat = std::nullopt;
        _next_frame_cache = std::nullopt;
        if (_image.source())
            _image.unsafe_get_source().set_index(-1);
        points.clear();
        
        auto seg = _frame.valid() ? GUICache::instance().tracklet_cache(_id.ID()) : nullptr;
        _match_mode = std::nullopt;
        
        if(seg && seg->contains(_frame)) {
            auto matched_using = obj.matched_using(seg->basic_stuff(_frame));
            if(matched_using.has_value())
                _match_mode = matched_using.value();
        }
        
        if(auto it = GUICache::instance().fish_selected_blobs.find(_id.ID());
           it != GUICache::instance().fish_selected_blobs.end())
        {
            _basic_stuff = it->second.basic_stuff;
            if(it->second.posture_stuff.has_value())
                _posture_stuff = it->second.posture_stuff->clone();
            else
                _posture_stuff.reset();
        
            if(it->second.pred)
                _pred = it->second.pred.value();
            else
                _pred.clear();
            
        } else {
            _basic_stuff.reset();
            _posture_stuff.reset();
            _pred.clear();
            
            if(not obj.empty()) {
                auto current_basic = obj.find_frame(frameIndex);
                if(not current_basic)
                    throw U_EXCEPTION(_id," should have a safe frame given that its not empty.");
                
                _basic_stuff = *current_basic;
                _safe_frame = _basic_stuff->frame;
                
                auto posture = obj.posture_stuff(_safe_frame);
                if(posture)
                    _posture_stuff = posture->clone();
                else
                    _posture_stuff.reset();
            }
        }
        
        /// need to update this as well so there are no old values
        _cached_outline = _posture_stuff && _posture_stuff->outline
                            ? &_posture_stuff->outline
                            : nullptr;
        
        _ML = obj.midline_length();
        _pp_midline = nullptr;
        
        if(   OPTION(gui_show_outline)
           || OPTION(gui_show_midline)
           || OPTION(gui_happy_mode))
        {
            if(_posture_stuff) {
                _cached_midline = SETTING(output_normalize_midline_data) ? obj.fixed_midline(frameIndex) : obj.calculate_midline_for(*_posture_stuff);
                _pp_midline = obj.pp_midline(frameIndex);
                assert(!_cached_midline || _cached_midline->is_normalized());
            }
        }
        
        if(FAST_SETTING(posture_direction_smoothing)) {
            std::map<Frame_t, Float2_t> interp;
            _previous_midline_angles.clear();
            _previous_midline_angles_d.clear();
            _previous_midline_angles_dd.clear();
        
            auto movement = obj.calculate_previous_vector(_frame);
            if(_pp_midline) {
                midline_direction = _pp_midline->midline_direction();
            } else midline_direction = Vec2();
            
            posture_direction_ = movement.direction;
            _posture_directions = movement.directions;
        }
        
        _view.set_dirty();
        
        _blob_bounds = _basic_stuff
                            ? _basic_stuff->blob.calculate_bounds()
                            : _view.bounds();
        
#if !COMMONS_NO_PYTHON
        if(frameIndex == _safe_frame && _basic_stuff) {
            auto c = Categorize::DataStore::_label_averaged_unsafe(&obj, Frame_t(frameIndex));
            if(c) {
                _avg_cat = c->id;
                _avg_cat_name = c->name;
            }
        }
        
        auto bdx = _basic_stuff.has_value() ? _basic_stuff->blob.blob_id() : pv::bid();
        _cat = Categorize::DataStore::_label_unsafe(Frame_t(_frame), bdx);
        if (_cat.has_value() && _cat != _avg_cat) {
            _cat_name = Categorize::DataStore::label(_cat)->name;
        }
#endif
        
        auto color_source = OPTION(gui_fish_color);
        if(_basic_stuff
           && color_source != "identity")
        {
            _library_y = Output::Library::get_with_modifiers(color_source, _info, _safe_frame);
            if(!GlobalSettings::is_invalid(_library_y)) {
                auto video_size = FindCoord::get().video_size();
                if(color_source == "X") 
                    _library_y /= video_size.width * FAST_SETTING(cm_per_pixel);
                else if(color_source == "Y")
                    _library_y /= video_size.height * FAST_SETTING(cm_per_pixel);
            }
        }
        
        _range = Range<Frame_t>(obj.start_frame(), obj.end_frame());
        _empty = obj.empty();
        
        _has_processed_tracklet = GUICache::instance().processed_tracklet_cache(_id.ID()); //obj.has_processed_tracklet(_frame);
        if(std::get<0>(_has_processed_tracklet)) {
            processed_tracklet = obj.processed_recognition(std::get<1>(_has_processed_tracklet).start());
        } else
            processed_tracklet = {};
        
        _tracklet = GUICache::instance().tracklet_cache(_id.ID());//obj.tracklet_for(_frame);
        if(_tracklet) {
            _qr_code = obj.qrcode_at(_tracklet->start());
        } else
            _qr_code = {};
        
        //const auto frame_rate = slow::frame_rate;
        //auto current_time = _time;
        //auto next_props = GUICache::instance()._next_props ? &GUICache::instance()._next_props.value() : nullptr;
        //auto next_props = Tracker::properties(_frame + 1_f);
        //auto next_time = next_props ? next_props->time : (current_time + 1.f/float(frame_rate));
        
        //if(!_next_frame_cache.valid)
        if(GUICache::instance()._props) {
            auto result = GUICache::instance().next_frame_cache(_id.ID());
            //auto result = obj.cache_for_frame(&GUICache::instance()._props.value(), _frame + 1_f, next_time);
            if(result.has_value()) {
                _next_frame_cache = *result.value();
            } else {
                //FormatWarning("Cannot create cache_for_frame of ", _id, " for frame ", _frame + 1_f, " because: ", result.error());
                _next_frame_cache = std::nullopt;
            }
        }
        
        /**
         * ML Predictions
         */
        if(_basic_stuff) {
            auto && [valid, tracklet] = obj.has_processed_tracklet(frameIndex);
            
            std::string title = "recognition";
            
            if(valid) {
                auto && [n, values] = obj.processed_recognition(tracklet.start());
                title = "average n:"+Meta::toStr(n);
                _raw_preds = values;
                
            } else {
                auto pred = GUICache::instance().find_prediction(_basic_stuff->blob.blob_id());
                if(pred)
                    _raw_preds = track::prediction2map(*pred);
                else
                    _raw_preds = std::nullopt;
            }
            
            _recognition_tracklet = tracklet;
            _recognition_str = title;
        }
        
        _color = get_color(&_basic_stuff.value());

        //if(OPTION(gui_pose_smoothing) > 0_f)
        //auto gui_pose_smoothing = OPTION(gui_pose_smoothing);
        //_average_pose = obj.pose_window(frameIndex.try_sub(gui_pose_smoothing), frameIndex + gui_pose_smoothing, frameIndex);
        //else
        //    _average_pose = obj.pos
        if (not _skelett) {
            _skelett = std::make_unique<Skelett>();
        }
        if(frameIndex == _safe_frame)
            _skelett->set_color(_color.alpha(150));
        else
            _skelett->set_color(Gray.alpha(100));
        _skelett->set_name(Meta::toStr(_id.color()));
        
        if(_basic_stuff && _basic_stuff->blob.pred.valid()
           && not _basic_stuff->blob.pred.pose.empty())
        {
            _skelett->set_pose(_basic_stuff->blob.pred.pose);
        }
        _skelett->set_skeleton(GUI_SETTINGS(detect_skeleton));
        if(_skelett->show_text())
            _skelett->set(GUIOPTION(detect_keypoint_names));

        updatePath(obj, _safe_frame, cmn::max(obj.start_frame(), _safe_frame.try_sub(1000_f)));
        
        auto &cache = GUICache::instance();
        if(OPTION(gui_show_probabilities)
           && cache.is_selected(_id.ID()))
        {
            auto c = cache.processed_frame().cached(_id.ID());
            if (c) {
                auto &mat = _image.unsafe_get_source();
                //if(mat.index() != narrow_cast<long_t>(_frame.get()))
                {
                    _probability_radius = saturate(FAST_SETTING(track_max_speed) / FAST_SETTING(cm_per_pixel) * 0.5_F, 1_F, 5000_F);
                    //auto coord = FindCoord::get();
                    
                    auto res_factor = max(1.0, _probability_radius * 2 / max(512, _probability_radius * 2 / 4));
                    const auto res = _probability_radius * 2 / res_factor;
                    if(mat.cols != res || mat.rows != res || mat.dims != 4)
                    {
                        mat.create(_probability_radius * 2 / res_factor, _probability_radius * 2 / res_factor, 4);
                        mat.set_to(0);
                    } //coord.video_size().height, coord.video_size().width, 4);
                    mat.set_index(_frame.get());
                    //mat.set_to(0);
                    
                    /*if(_probability_radius < 10 || _probability_center.empty())
                        mat.set_to(0);
                    else {*/
                        /*for (int y = 0; y < _probability_radius * 2; ++y) {
                            auto ptr = mat.ptr(y, 0);
                            auto end = mat.ptr(y, _probability_radius);
                            if (end > mat.data() + mat.size())
                                throw U_EXCEPTION("Mat end ", mat.size(), " end: ", uint64_t(end - mat.data()));
                            std::fill(ptr, end, 0);
                        }*/
                    //}
                    
                    _probability_center = c->estimated_px;
                    float sum;
                    
                    auto plot = [&](int x, int y) {
                        auto pos = Vec2(x, y);
                        if (pos.x < 0 || pos.x >= mat.cols)
                            return;
                        if (pos.y < 0 || pos.y >= mat.rows)
                            return;
                        if (_frame <= _range.start)
                            return;
                        
                        auto ptr = mat.ptr(pos.y, pos.x);
                        
                        auto p = obj.probability(MaybeLabel{}, *c, _frame, _probability_center - _probability_radius + pos * res_factor + 0.5, 1); //TODO: add probabilities
                        if (p < FAST_SETTING(match_min_probability))
                            return;
                        
                        auto clr = Viridis::value(p).alpha(uint8_t(min(255, 255.f * p)));
                        *(ptr + 0) = clr.r;
                        *(ptr + 1) = clr.g;
                        *(ptr + 2) = clr.b;
                        *(ptr + 3) = clr.a;
                        sum += p;
                    };
                    
                    /*do {
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
                        
                    } while (sum > 0 || _probability_radius < 10);*/
                    
                    distribute_indexes([&](auto, auto it, auto nex, auto){
                        //Print("y ", it, " - ", nex);
                        for(auto y = it; y < nex; ++y) {
                            int x = 0;
                            auto ptr = mat.ptr(y, 0);
                            auto end = mat.ptr(y, mat.cols);
                            
                            for (; ptr != end; ++ptr, ++x) {
                                //if (*(ptr) <= 5)
                                plot(x, y);
                            }
                        }
                    }, cache.pool(), int(0), int(mat.rows));
                    
                    /*for (int y = 0; y < _probability_radius * 2; ++y) {
                     int x = 0;
                     auto ptr = mat.ptr(y, x);
                     auto end = mat.ptr(y, _probability_radius * 2);
                     
                     for (; ptr != end; ++ptr, ++x) {
                         //if (*(ptr) <= 5)
                         plot(x, y);
                     }
                    }*/
                    
                    _image.set_pos(_probability_center - _probability_radius - _blob_bounds.pos());
                    _image.set_scale(res_factor);
                    _image.updated_source();
                    //tf::imshow("image", mat.get());
                }
            }
        }
        
        {
            const auto centroid = _posture_stuff.has_value() && _posture_stuff->centroid_posture
                    ? _posture_stuff->centroid_posture.get()
                    : (_basic_stuff.has_value()
                        ? &_basic_stuff->centroid
                        : nullptr);
            const Vec2 offset = -_blob_bounds.pos();
            auto c_pos = centroid
                ? (centroid->pos<Units::PX_AND_SECONDS>() + offset)
                : Vec2(_blob_bounds.size() * 0.5_F);
            
            auto angle = centroid ? -centroid->angle() : 0_F;
            if (_posture_stuff.has_value()
                && _posture_stuff->head)
            {
                angle = -_posture_stuff->head->angle();
            }
            
            if(setup_rotated_bbx(FindCoord::get(), offset, c_pos, angle)) {
                _view.set_animating(false);
            }
        }
        
        try {
            dyn::State state;
            _data->label_text = dyn::parse_text(OPTION(gui_fish_label), _data->context, state);
        } catch(const std::exception& ex) {
#ifndef NDEBUG
            FormatWarning("Caught exception when parsing text: ", ex.what());
#endif
            _data->label_text = "[<red>ERROR</red>] <lightgray>gui_fish_label</lightgray>: <red>"+std::string(ex.what())+"</red>";
        }
    }
    
    /*void Fish::draw_occlusion(gui::DrawStructure &window) {
        auto &blob = _obj.pixels(_safe_idx);
        window.image(blob_bounds.pos(), _image);
    }*/

// Insert points uniformly into a vector until its size matches the target size
void insertUniformPoints(std::vector<Vec2>& vec, size_t targetSize) {
    size_t currentSize = vec.size();
    if (currentSize >= targetSize) return;

    // Step 1: Calculate the total length of the polyline formed by the current points
    float totalLength = 0.0f;
    std::vector<float> segmentLengths(currentSize - 1);
    for (size_t i = 0; i < currentSize - 1; ++i) {
        segmentLengths[i] = euclidean_distance(vec[i], vec[i + 1]);
        totalLength += segmentLengths[i];
    }

    // Step 2: Calculate the required distance between each point in the final vector
    float desiredSpacing = totalLength / (targetSize - 1);

    // Step 3: Walk along the original segments and insert new points
    std::vector<Vec2> newVec;
    newVec.push_back(vec[0]);  // Start with the first point

    float accumulatedDistance = 0.0f;
    size_t currentSegment = 0;

    // Loop until we have inserted all required points
    for (size_t i = 1; i < targetSize - 1; ++i) {
        float targetDistance = i * desiredSpacing;

        // Advance along the segments until we reach the desired distance
        while (accumulatedDistance + segmentLengths[currentSegment] < targetDistance) {
            accumulatedDistance += segmentLengths[currentSegment];
            currentSegment++;
        }

        // Calculate the exact position of the new point within the current segment
        float remainingDistance = targetDistance - accumulatedDistance;
        float t = remainingDistance / segmentLengths[currentSegment];  // Fraction along the current segment

        // Interpolate to find the new point
        Vec2 newPoint = lerp(vec[currentSegment], vec[currentSegment + 1], t);
        newVec.push_back(newPoint);
    }

    // Add the last point
    newVec.push_back(vec.back());

    vec = newVec;  // Update the original vector
}

// Find the optimal rotation that minimizes the total distance between source and target
size_t findOptimalRotation(const std::vector<Vec2>& source, const std::vector<Vec2>& target) {
    size_t bestRotation = 0;
    Float2_t minDistance = std::numeric_limits<Float2_t>::max();
    size_t N = source.size();

    // Ensure source and target are of the same size
    if (N != target.size() || N == 0) {
        throw std::invalid_argument("Source and target vectors must be of the same non-zero size.");
    }

    // Precompute distances for all possible shifts
    for (size_t shift = 0; shift < N; ++shift) {
        Float2_t totalDistance = 0;

        for (size_t i = 0; i < N; ++i) {
            size_t sourceIndex = (i + shift) % N;
            totalDistance += sqdistance(source[sourceIndex], target[i]);

            // Early exit if totalDistance exceeds minDistance
            if (totalDistance >= minDistance) {
                break;
            }
        }

        if (totalDistance < minDistance) {
            minDistance = totalDistance;
            bestRotation = shift;
        }
    }

    return bestRotation;
}

// Function to calculate the center (centroid) of a set of points
Vec2 calculateCenter(const std::vector<Vec2>& points) {
    Vec2 center(0_F, 0_F);
    for (const auto& p : points)
        center += p;
    center /= static_cast<Float2_t>(points.size());
    return center;
}

// Function to adjust points relative to a center point
void adjustPointsRelativeToCenter(std::vector<Vec2>& points, const Vec2& center, bool subtract) {
    for (auto& p : points) {
        if (subtract) {
            p.x -= center.x;
            p.y -= center.y;
        } else {
            p.x += center.x;
            p.y += center.y;
        }
    }
}

Float2_t easeInOutQuad(Float2_t t) {
    if (t < 0.5_F) {
        return 2._F * t * t;
    } else {
        return -1._F + (4._F - 2._F * t) * t;
    }
}

Float2_t easeOutCubic(Float2_t t) {
    t -= 1._F;
    return t * t * t + 1._F;
}


bool morphVectorsWithRotation(std::vector<Vec2>& source, std::vector<Vec2>& target, Float2_t dt, Float2_t threshold) {
    // Ensure both source and target are not empty
    if (source.empty()) {
        source = target;
        return true;
    }

    //Print("Morphing vectors with dt =", dt, ", threshold =", threshold);

    // Step 1: Equalize the number of points
    size_t sourceSize = source.size();
    size_t targetSize = target.size();
    
    // we are using sqdistances
    threshold *= threshold;

    if (sourceSize < targetSize) {
        insertUniformPoints(source, targetSize);
    } else if (targetSize < sourceSize) {
        insertUniformPoints(target, sourceSize);
    }

    //Print("After equalizing, number of points =", source.size());

    // Step 2: Calculate centers
    /*Vec2 sourceCenter = calculateCenter(source);
    Vec2 targetCenter = calculateCenter(target);

    // Step 3: Adjust points relative to their centers
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] -= sourceCenter;
        target[i] -= targetCenter;
    }*/

    // Step 4: Find optimal rotation
    size_t optimalRotation = findOptimalRotation(source, target);
    //Print("Optimal rotation =", optimalRotation);

    // Step 5: Rotate source for optimal alignment
    std::rotate(source.begin(), source.begin() + optimalRotation, source.end());

    // Step 5: Perform the synchronous morphing
    size_t size = source.size();

    // Step 5.1: Calculate maximum distance
    Float2_t maxDistance = 0._F;
    std::vector<Float2_t> distances(size);
    for (size_t i = 0; i < size; ++i) {
        distances[i] = sqdistance(source[i], target[i]);
        if (distances[i] > maxDistance) {
            maxDistance = distances[i];
        }
    }

    // Step 5.2: Morph points using non-linear interpolation
    Float2_t interpolationSpeed = 5._F;  // Adjust as needed
    for (size_t i = 0; i < size; ++i) {
        // Normalize the movement based on distance
        Float2_t normalizedT = dt * interpolationSpeed;
        if (maxDistance > 0._F) {
            normalizedT *= (distances[i] / maxDistance);
        }
        
        normalizedT = sqrt(normalizedT);
        normalizedT = std::min(normalizedT, 1._F);  // Clamp to 1.0

        // Apply easing function to normalizedT
        Float2_t easedT = easeOutCubic(normalizedT);

        // Morph the source point towards the corresponding target point
        source[i] = lerp(source[i], target[i], easedT);

        // If the source point is close enough to the target point, snap it to the target
        if (sqdistance(source[i], target[i]) <= threshold) {
            source[i] = target[i];
        }

        // Debug print for each point
        //Print("Point", i, ": normalizedT =", normalizedT, ", easedT =", easedT, ", distance =", distances[i]);
    }

    // Step 7: Adjust source points back to absolute positions using target center
    /*for (size_t i = 0; i < size; ++i) {
        source[i] += targetCenter;
    }*/

    // Step 8: Check if all points have reached their targets
    bool allPointsMatch = true;
    for (size_t i = 0; i < size; ++i) {
        if (sqdistance(source[i], target[i] /*+ targetCenter*/) > threshold) {
            allPointsMatch = false;
            break;
        }
    }

    //Print("All points match =", allPointsMatch);

    // If all points match, resize source if needed
    if (allPointsMatch) {
        source.resize(target.size());
        return true;
    } else
        return false;
}

std::vector<Vec2> generateCircleVertices(const Vec2& center, Float2_t radius, int N) {
    std::vector<Vec2> vertices;
    vertices.reserve(N);  // Reserve space for efficiency

    // Calculate the angle between each point
    Float2_t angleIncrement = (2.0_F * M_PI) / static_cast<Float2_t>(N);

    // Generate the vertices
    for (int i = 0; i < N; ++i) {
        Float2_t angle = angleIncrement * static_cast<Float2_t>(i);
        Float2_t x = center.x + radius * cos(angle);
        Float2_t y = center.y + radius * sin(angle);
        vertices.emplace_back(x, y);
    }

    return vertices;
}

template<typename Points>
std::vector<Vec2> find_rbb(const FindCoord& coords, Vec2 offset, Float2_t angle, Vec2 pt_center, Points& points)
{
    Vec2 mi(FLT_MAX, FLT_MAX);
    Vec2 ma(0, 0);
    
    using Point_t = Points::value_type;
    
    Transform transform;
    transform.translate(pt_center);
    transform.rotate(DEGREE(angle));
    transform.translate(-pt_center);
    
    for(auto &_pt : points) {
        if constexpr(std::same_as<Point_t, blob::Pose::Point>) {
            if(not _pt.valid())
                continue;
        }
        
        Vec2 pt = transform.transformPoint(_pt);
        assert(not std::isnan(pt.x));
        mi.x = min(mi.x, (Float2_t)pt.x);
        mi.y = min(mi.y, (Float2_t)pt.y);
        ma.x = max(ma.x, (Float2_t)pt.x);
        ma.y = max(ma.y, (Float2_t)pt.y);
    }
    
    assert(not std::isnan(coords.bowl_scale().x));
    auto scaled_w = abs(coords.bowl_scale().x * (mi.x - ma.x));
    if(scaled_w < 20) {
        auto d = 20 - scaled_w;
        mi.x -= d * 0.5;
        ma.x += d * 0.5;
    }
    
    auto scaled_h = abs(coords.bowl_scale().y * (mi.y - ma.y));
    if(scaled_h < 20) {
        auto d = 20 - scaled_h;
        mi.y -= d * 0.5;
        ma.y += d * 0.5;
    }
    
    //Print("scaled_w = ", scaled_w, " scaled_h = ", scaled_h);
    
    auto bds = Bounds{mi, Size2(ma - mi)};
    auto inverted = transform.getInverse();
    
    Transform t;
    t.translate(offset);
    inverted = t.combine(inverted);
    
    std::vector<Vec2> vertices {
        inverted.transformPoint(mi),
        inverted.transformPoint(Vec2(ma.x, mi.y)),
        inverted.transformPoint(ma),
        inverted.transformPoint(Vec2(mi.x, ma.y)),
    };
    
    mi = Vec2(FLT_MAX, FLT_MAX);
    ma = Vec2();
    
    for(auto &p0 : vertices) {
        assert(not std::isnan(p0.x));
        
        mi.x = min(mi.x, p0.x);
        mi.y = min(mi.y, p0.y);
        ma.x = max(ma.x, p0.x);
        ma.y = max(ma.y, p0.y);
    }
    
    auto c = (ma - mi) * 0.5;
    for(auto &p0 : vertices)
        p0 += c;
    
    return vertices;
}

bool Fish::setup_rotated_bbx(const FindCoord& coords, const Vec2& offset, const Vec2&, Float2_t angle)
{
    bool corners_changed{false};
    
    if(_basic_stuff
       && _basic_stuff->blob.pred.valid()
       && not _basic_stuff->blob.pred.pose.empty())
    {
        auto &pose = _basic_stuff->blob.pred.pose;
        Vec2 pose_center;
        size_t count{0};
        
        for(auto &_pt : pose.points) {
            if(_pt.valid()) {
                pose_center += (Vec2)_pt;
                ++count;
            }
        }
        
        if(count > 0)
            pose_center /= count;
        
        auto corners = find_rbb(coords, offset, angle, pose_center, pose.points);
        if(_current_corners != corners)
        {
            _current_corners = std::move(corners);
            corners_changed = true;
        }
        
    } else if(_posture_stuff) {
        auto example_points = _posture_stuff->outline.uncompress();
        
        Vec2 c;
        for(auto &pt : example_points) {
            pt += _blob_bounds.pos();
            c += pt;
        }
        if(not example_points.empty())
            c /= example_points.size();
        
        auto corners = find_rbb(coords, offset, angle, c, example_points);
        if(_current_corners != corners)
        {
            _current_corners = std::move(corners);
            corners_changed = true;
        }
        
    } else if(auto ptr = _basic_stuff ? _basic_stuff->blob.unpack() : nullptr;
       ptr != nullptr)
    {
        std::vector<Vec2> example_points;
        example_points.reserve(ptr->lines()->size() * 2);
        
        for(auto &line : *ptr->lines()) {
            example_points.emplace_back(line.x0, line.y);
            example_points.emplace_back(line.x1, line.y);
        }
        
        auto corners = find_rbb(coords, offset, angle, _blob_bounds.center(), example_points);
        if(_current_corners != corners)
        {
            _current_corners = std::move(corners);
            corners_changed = true;
        }
        
        /*auto mi = Vec2(FLT_MAX, FLT_MAX);
        auto ma = Vec2();
        
        for(auto &p0 : *ellipse) {
            mi.x = min(mi.x, p0.x);
            mi.y = min(mi.y, p0.y);
            ma.x = max(ma.x, p0.x);
            ma.y = max(ma.y, p0.y);
        }
        
        auto c = (ma - mi) * 0.5;
        for(auto &p0 : *ellipse)
            p0 -= c;
        
        {
            auto coeffs = periodic::eft(*ellipse, 7);
            if(coeffs) {
                auto pts = std::move(periodic::ieft(*coeffs, coeffs->size(), 50, center, false, 1_F).back());
                ellipse = std::move(pts);
            }
        }*/
        
       //ellipse = std::make_unique<periodic::points_t::element_type>(vertices);
    }
    
    const auto init = [this](auto& o) {
        /*if constexpr(std::is_base_of_v<Drawable, std::remove_cvref_t<decltype(o)>>) {
            //Print("Registering ", o);
            o.on_hover([this](Event e) {
                selection_hovered(e);
            });
            o.on_click([this](Event e) {
                selection_clicked(e);
            });
            o.set_clickable(true);
        } else {
            Print("Unknown object.");
        }*/
    };
    
    //Print(_id, ": corners_changed = ", corners_changed);
    
    /// upsample points if we have them
    if(corners_changed || _frame_change.elapsed() < 0.4_F) {
        auto corners = _current_corners;
        insertUniformPoints(corners, 50);
        
        //Print("Calculating area for ", _current_corners, " of ", _id);
        auto poly_area = polygon_area(_current_corners);
        //auto radius = (slow::calculate_posture && _ML != GlobalSettings::invalid() ? _ML : _blob_bounds.size().max()) * 0.5;
        if(corners_changed) {
            _tight_selection.set_vertices(_current_corners);
            _tight_selection.set_origin(Vec2{0.5_F});
            _tight_selection.set_fill_clr(Transparent);
            _cached_circle.clear();
            _cached_points.clear();
        }
        
        auto c = calculateCenter(corners);
        
        Float2_t max_d{0};
        for(auto &pt : corners) {
            auto d = sqdistance(c, pt);
            if(d > max_d)
                max_d = d;
        }
        max_d = sqrt(max_d);
        
        {
            auto radius = max_d * 0.85;
            if(_radius <= 1)
                _radius = radius;
            else {
                _radius = _radius + (radius - _radius) * GUICache::instance().dt() * 5_F;
                _cached_circle.clear();
            }
        }
        
        auto circle_area = M_PI * SQR(_radius);
        
        if(_frame_change.elapsed() >= 0.15_F && (poly_area < circle_area * 0.75 || not _basic_stuff.has_value() || _basic_stuff->frame != _frame ))
        {
            // ok
        } else {
            // make circle
            if(_cached_circle.empty())
                _cached_circle = generateCircleVertices(Vec2(), _radius, 100);
            
            corners.clear();
            for(auto &pt : _cached_circle)
                corners.push_back(pt + _blob_bounds.size() * 0.5_F + _radius);
        }
        
        _cached_points = std::move(corners);
    }
    
    /// if we dont have data, dont do anything
    if(_cached_points.empty()) {
        //Print("done with ", _id);
        return true;
    }
    
    if(not std::holds_alternative<Polygon>(_selection)) {
        _selection.emplace<Polygon>();
        std::visit(init, _selection);
    }
    
    auto &poly = std::get<Polygon>(_selection);
    
    auto finished = morphVectorsWithRotation(_current_points, _cached_points, GUICache::instance().dt() * 1.25_F, 1_F);
    if(finished && _frame_change.elapsed() >= 0.5_F) {
        //Print("done with ",_id,".");
        poly.set_animating(false);
    } else {
        //Print("not done with ", _id);
        poly.set_animating(true);
        poly.set_dirty();
    }
    
    poly.set_vertices(_current_points);
    //poly.set_vertices(*ellipse);
    poly.set_origin(Vec2(0.5));
    poly.set_name(_id.name()+"-poly");
    return finished;
}

void Fish::selection_hovered(Event e) {
    //Print("Hovering ", e.hover.x, ",", e.hover.y);
    _path_dirty = true;
    if(!GUICache::exists() || !e.hover.hovered)
        return;
    GUICache::instance().set_tracking_dirty();
}

void Fish::selection_clicked(Event) {
    auto ID = _id.ID();
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
}
    
    void Fish::update(const FindCoord& coord, Entangled& parent, DrawStructure &graph) {
        //const auto frame_rate = slow::frame_rate;//FAST_SETTING(frame_rate);
        //const float track_max_reassign_time = FAST_SETTING(track_max_reassign_time);
        const auto single_identity = OPTION(gui_single_identity_color);
        //const auto properties = Tracker::properties(_idx);
        //const auto safe_properties = Tracker::properties(_safe_idx);
        auto &cache = GUICache::instance();
        
        _view.set_bounds(_blob_bounds);
        _label_parent.set_bounds(Bounds(Vec2(0), parent.size()));
        
        const Vec2 offset = -_blob_bounds.pos();
        
        const auto centroid = _basic_stuff.has_value() ? &_basic_stuff->centroid : nullptr;
        const auto head = _posture_stuff.has_value() ? _posture_stuff->head.get() : nullptr;
        const auto pcentroid = _posture_stuff.has_value() ? _posture_stuff->centroid_posture.get() : nullptr;
        
        _fish_pos = centroid
            ? centroid->pos<Units::PX_AND_SECONDS>()
            : (_blob_bounds.pos() + _blob_bounds.size() * 0.5);
        
        const bool hovered = std::visit([this](auto& o) -> bool {
                return _tight_selection.hovered();
            
        }, _selection);
        //const bool timeline_visible = true;//GUICache::exists() && Timeline::visible(); //TODO: timeline_visible
        //const float max_color = timeline_visible ? 255 : OPTION(gui_faded_brightness);
        
        auto base_color = single_identity != Transparent ? single_identity : _id.color();
        //auto clr = base_color.alpha(saturate(((cache.is_selected(_obj.identity().ID()) || hovered) ? max_color : max_color * 0.4f) * time_fade_percent));
        //auto clr = base_color.alpha(saturate(max_color));// * time_fade_percent));
        //_color = clr;
        
        
        auto active = GUICache::instance().active_ids.find(_id.ID()) != GUICache::instance().active_ids.end();
        bool is_selected = cache.is_selected(_id.ID());
        std::vector<Vec2> points;



#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE //&& false
        if (GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
        {
            if (!is_selected) _view.tag(Effects::blur);
            else _view.untag(Effects::blur);

            /*if (is_selected && GUI::instance() && GUI::instance()->base()) {
                ((MetalImpl*)((IMGUIBase*)GUI::instance()->base())->platform().get())->center[0] = _view.global_bounds().x / float(GUI::instance()->base()->window_dimensions().width) / gui::interface_scale() * graph.scale().x;
                ((MetalImpl*)((IMGUIBase*)GUI::instance()->base())->platform().get())->center[1] = _view.global_bounds().y / float(GUI::instance()->base()->window_dimensions().height) / gui::interface_scale() * graph.scale().y;
            }*/
        }
#endif
#endif

        if (active && _cached_outline) {
            if (OPTION(gui_show_shadows) || OPTION(gui_show_outline)) {
                if(points.empty())
                    points = _cached_outline->uncompress();
            }
            
            if (OPTION(gui_show_shadows)) {
                if (!_polygon) {
                    _polygon = std::make_shared<Polygon>(std::make_shared<std::vector<Vec2>>());
                    _polygon->set_fill_clr(Black.alpha(25));
                    _polygon->set_origin(Vec2(0.5));
                }
                _polygon->set_vertices(points);
                
                auto video_size = FindCoord::get().video_size();
                float size = video_size.length() * 0.0025f;
                Vec2 scaling(SQR(offset.x / video_size.width),
                             SQR(offset.y / video_size.height));
                _polygon->set_pos(-offset + scaling * size + _view.size() * 0.5);
                _polygon->set_scale(scaling * 0.25 + 1);
                _polygon->set_fill_clr(Black.alpha(25));

#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                if(GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
                {
                    if(is_selected)_polygon->tag(Effects::blur);
                    else _polygon->untag(Effects::blur);
                }
#endif
#endif
            }
        }

#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
        if constexpr(std::is_same<MetalImpl, default_impl_t>::value) {
            if(GUI_SETTINGS(gui_macos_blur) && is_selected) {
                auto it = cache.fish_selected_blobs.find(_id.ID());
                if (it != cache.fish_selected_blobs.end()) {
                    auto dp = cache.display_blobs.find(it->second.bdx);
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

        const int panic_button = OPTION(panic_button);
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

            if (active && _cached_outline && OPTION(gui_show_outline)) {
                Line::Vertices_t oline;
                if(points.empty())
                    points = _cached_outline->uncompress();

                // check if we actually have a tail index
                if (OPTION(gui_show_midline) && _cached_midline && _cached_midline->tail_index() != -1)
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
                window.add<Line>(oline, Line::Thickness_t{OPTION(gui_outline_thickness)});
                //if(line)
                //    window.text(Meta::toStr(line->points().size()) + "/" + Meta::toStr(oline.size()), Vec2(), White);
                //window.vertices(oline);
                
                if(_basic_stuff
                   && _basic_stuff->blob.pred.valid()
                   && _basic_stuff->blob.pred.outlines.has_holes())
                {
                    auto &lines = _basic_stuff->blob.pred.outlines.lines;
                    for(size_t i = 0; i<lines.size(); ++i) {
                        Line::Vertices_t gline;
                        for(auto &pt : (std::vector<Vec2>)lines.at(i)) {
                            gline.emplace_back(pt + offset, _color.saturation(0.25));
                        }
                        window.add<Line>(gline, Line::Thickness_t{OPTION(gui_outline_thickness)});
                    }
                }

            }
            if (active && _cached_midline && OPTION(gui_show_midline)) {
                std::vector<MidlineSegment> midline_points;
                //Midline _midline(*_cached_midline);
                //float len = _obj.midline_length();

                //if(len > 0)
                //    _midline.fix_length(len);

                auto& _midline = *_cached_midline;
                midline_points = _midline.segments();

                Line::Vertices_t line;
                auto tf = _cached_midline->transform(default_config::individual_image_normalization_t::none, true);

                for (auto &segment : midline_points) {
                    //Vec2 pt = segment.pos;
                    //float angle = _cached_midline->angle() + M_PI;
                    //float x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
                    //float y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));

                    //pt = Vec2(x, y) + _cached_midline->offset();
                    line.push_back(Vertex(tf.transformPoint(segment.pos), _color));
                }

                window.add<Line>(line, Line::Thickness_t{OPTION(gui_outline_thickness)});
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
            if(not _basic_stuff.has_value()) {
                return;
            }

            if(_id.ID() == Idx_t(28)) {
                //Print("Updating ", _id, " animating=",_view.is_animating());
            }
            _path_dirty = false;
            
            if (OPTION(gui_show_paths)) {
                for(auto& p : _paths)
                    _view.advance_wrap(*p);
                //paintPath(offset);
            }

            if (FAST_SETTING(track_max_individuals) > 0 && OPTION(gui_show_boundary_crossings))
                update_recognition_circle();

            if(panic_button) {
                _view.add<Line>(Line::Point_t(_posture.pos()), Line::Point_t(mp), LineClr{ White.alpha(50) });
            }
        
            _view.advance_wrap(_posture);
        
            // DISPLAY LABEL AND POSITION
            auto bg = GUICache::instance().background();
            const auto centroid = _posture_stuff.has_value() && _posture_stuff->centroid_posture
                    ? _posture_stuff->centroid_posture.get()
                    : (_basic_stuff.has_value()
                        ? &_basic_stuff->centroid
                        : nullptr);
            
            auto c_pos = (centroid ? centroid->pos<Units::PX_AND_SECONDS>() + offset : Vec2());
            if(not bg || c_pos.x > bg->image().cols || c_pos.y > bg->image().rows)
                return;
        
            auto v = 255 - int(bg->image().at(c_pos.y, c_pos.x));
            if(v >= 100)
                v = 220;
            else
                v = 50;
        
            float angle = -centroid->angle();
            if (head) {
                angle = -head->angle();
            }
        
            if(OPTION(gui_show_texts)) {
                if(_next_frame_cache.has_value()) {
                    auto estimated = _next_frame_cache->estimated_px + offset;
                    
                    _view.add<Circle>(Loc(c_pos), Radius{2}, LineClr{White.alpha(255)});
                    _view.add<Line>(Line::Point_t(c_pos), Line::Point_t(estimated), LineClr{ _color });
                    _view.add<Circle>(Loc(estimated), Radius{2}, LineClr{Transparent}, FillClr{_color});
                }
            }
        
            if(OPTION(gui_happy_mode)
               && _cached_midline
               && _cached_outline
               && _posture_stuff.has_value()
               && _posture_stuff->head)
            {
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
                        auto epos = Vec2(eye.pos);
                        
                        _view.add<Circle>(Loc(epos + offset), Radius(5 * eye_scale), LineClr{Black.alpha(200)}, FillClr{White.alpha(125)});
                        auto c = _view.add<Circle>(Loc(epos + Vec2(2.5).mul(d * eye_scale) + offset), Radius(3 * eye_scale), LineClr{Transparent}, FillClr{Black.alpha(200)});
                        c->set_scale(Vec2(1, ph.blinking ? h : 1));
                        c->set_rotation(atan2(ph.direction) + RADIANS(90));//posture->head->angle() + RADIANS(90));
                        _view.add<Circle>(Loc(epos + Vec2(2.5).mul(d * eye_scale) + Vec2(2 * eye_scale).mul(sun_direction) + offset), Radius(sqrt(eye_scale)), LineClr{Transparent}, FillClr{White.alpha(200 * c->scale().min())});
                    }
                }
            }

            auto color_source = OPTION(gui_fish_color);
            auto bdx = _basic_stuff.has_value() ? _basic_stuff->blob.blob_id() : pv::bid();
            auto pdx = _basic_stuff.has_value() ? _basic_stuff->blob.parent_id : pv::bid();

            if(color_source == "viridis") {
                GUICache::instance().processed_frame().transform_blobs_by_bid(std::array{bdx, pdx}, [&,bdx,pdx] (const pv::Blob& b)
                {
                    if(!is_in(b.blob_id(), bdx, pdx))
                        return true;
                    
                    auto && [dpos, _difference] = b.difference_image(*GUICache::instance().background(), 0);
                    auto difference = _difference->to_greyscale();
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
            
            } else if(not GlobalSettings::is_invalid(_library_y)) {
                auto percent = min(1.f, cmn::abs(_library_y));
                
                if(percent < 1) {
                    Color clr = /*Color(225, 255, 0, 255)*/ base_color * percent + Color(50, 50, 50, 255) * (1 - percent);
                    
                    GUICache::instance().processed_frame().transform_blobs([&, bdx, pdx](const pv::Blob& b)
                                                                           {
                        if(!is_in(b.blob_id(), bdx, pdx))
                            return true;
                        
                        auto && [image_pos, image] = b.binary_image(*GUICache::instance().background(), FAST_SETTING(track_threshold));
                        auto && [dpos, _difference] = b.difference_image(*GUICache::instance().background(), 0);
                        auto difference = _difference->to_greyscale();
                        
                        auto rgba = Image::Make(image->rows, image->cols, 4);
                        
                        uchar maximum = 0;
                        for(size_t i=0; i<difference->size(); ++i) {
                            maximum = max(maximum, difference->data()[i]);
                        }
                        for(size_t i=0; i<difference->size(); ++i)
                            difference->data()[i] = (uchar)min(255, float(difference->data()[i]) / maximum * 255);
                        
                        rgba->set_channels(image->data(), {0, 1, 2});
                        rgba->set_channel(3, *difference);
                        
                        _view.add<ExternalImage>(std::move(rgba), image_pos + offset, Vec2(1), clr);
                        
                        return false;
                    });
                }
            }
        
            if(is_selected && OPTION(gui_show_probabilities)) {
                //_image.set_pos(offset);
                _view.advance_wrap(_image);
            }
            else if(!_image.source()->empty()) {
                _image.set_source(std::make_unique<Image>());
            }
            
            _view.advance_wrap(_tight_selection);
            
            if ((hovered || is_selected) && OPTION(gui_show_selections)) {
                std::visit([&](auto& o) {
                    if(abs(last_scale - FindCoord::get().bowl_scale().x) > std::numeric_limits<Float2_t>::epsilon()) {
                        if(last_scale == 0)
                            _frame_change.reset();
                        last_scale = FindCoord::get().bowl_scale().x;
                        setup_rotated_bbx(FindCoord::get(), offset, c_pos, angle);
                    }
                    else if constexpr(std::is_base_of_v<Drawable, std::remove_cvref_t<decltype(o)>>) {
                        if(not o.is_animating())
                            return;
                        setup_rotated_bbx(FindCoord::get(), offset, c_pos, angle);
                    }
                }, _selection);
                
                std::visit([this](auto& o) {
                    if constexpr(std::is_base_of_v<Drawable, std::remove_cvref_t<decltype(o)>>) {
                        o.set_name(_id.name());
                        
                        //Print("Highlight ", o);
                        if(_tight_selection.hovered()) {
                            o.set(FillClr{White.alpha(50)});
                            o.set(LineClr{White.alpha(200)});
                        } else {
                            o.set(FillClr{White.alpha(5)});
                            o.set(LineClr{White.alpha(100)});
                        }
                        
                        _view.advance_wrap(o);
                        //if(not std::isinf(_view.global_text_scale().x)
                        //   && _view.global_text_scale().min() > 0)
                        {
                            
                            //auto cbs = FindCoord::get().convert(BowlRect(o.bounds()));
                            /*static Vec2 minimum_scale(1, 1);
                            static int direction = -1;
                            minimum_scale += direction * minimum_scale.x * 0.01;
                            if(minimum_scale.x < 0.5)
                                direction = 1;
                            else if(minimum_scale.x > 1.5)
                                direction = -1;
                                
                            minimum_scale = Vec2(20_F).div(cbs.size());
                            auto scale = Vec2(max(1_F, minimum_scale.y),
                                              max(1_F, minimum_scale.x));
                            //max(28_F / cbs.size().mean(), 1_F);
                            //auto s = FindCoord::get().bowl_scale();
                            //Print("cbs = ", cbs, " scale = ", minimum_scale);
                            o.set_scale(scale);//s.reciprocal());*/
                        }
                        
                        //static float angle = 0;
                        //angle += 0.01;
                        //o.set(Rotation{angle});
                    }
                }, _selection);
            } else {
                std::visit([this](auto& o) {
                    if constexpr(std::is_base_of_v<Drawable, std::remove_cvref_t<decltype(o)>>) {
                        //Print("Transparent ", o);
                        o.set(FillClr{Transparent});
                        o.set(LineClr{Transparent});
                        //o.set_animating(false);
                        _view.advance_wrap(o);
                        //o.set(Rotation{0});
                    }
                    last_scale = 0_F;
                }, _selection);
            }
            
            if ((hovered || is_selected) && OPTION(gui_show_selections)) {
                auto radius = _radius;//(slow::calculate_posture && _ML != GlobalSettings::invalid() ? _ML : _blob_bounds.size().max()) * 0.6;
                
                auto circle_clr = Color((uint8_t)v, (uint8_t)saturate(255 * (hovered ? 1.7 : 1)));
                if(cache.primary_selected_id() != _id.ID())
                    circle_clr = Gray.alpha(circle_clr.a);
                
                // draw unit circle showing the angle of the fish
                Loc pos(cmn::cos(angle), -cmn::sin(angle));
                pos = pos * radius + c_pos;
            
                _view.add<Circle>(pos, Radius{3}, LineClr{circle_clr});
                _view.add<Line>(Line::Point_t(c_pos), Line::Point_t(Vec2(pos)), LineClr{ circle_clr });
            
                if(FAST_SETTING(posture_direction_smoothing)) {
                    size_t i = 0;
                    for(auto d : _posture_directions) {
                        pos = c_pos + d * (radius + i * 5);
                        _view.add<Line>(Line::Point_t(c_pos), Line::Point_t(Vec2(pos)), LineClr{Red.alpha(50)});
                        _view.add<Circle>(pos, Radius{3}, LineClr{Red.alpha(100)});
                        ++i;
                    }
                    
                    auto _needs_invert = !FAST_SETTING(midline_invert);
                    auto direction = _needs_invert ? midline_direction : -midline_direction;
                    
                    auto inverted = acos((-direction).dot(posture_direction_)) < acos(direction.dot(posture_direction_));
                    if(inverted)
                        direction = -direction;
                    
                    
                    pos = c_pos + direction * radius;
                    _view.add<Line>(Line::Point_t(c_pos), Line::Point_t(Vec2(pos)), LineClr{Yellow});
                    
                    pos = c_pos + posture_direction_ * radius;
                    _view.add<Line>(Line::Point_t(c_pos), Line::Point_t(Vec2(pos)), LineClr{Cyan});
                    _view.add<Circle>(pos, Radius{3}, LineClr{inverted ? Yellow : Cyan});
                }
            }
            
        });

        
        parent.advance_wrap(_view);
        
        if(_basic_stuff.has_value() && _basic_stuff->blob.pred.pose.size() > 0) {
            if(_skelett)
                parent.advance_wrap(*_skelett);
        }

        parent.advance_wrap(_label_parent);
        _label_parent.update([&](auto&){
            label(coord, _label_parent);
        });
    }

Color Fish::get_color(const BasicStuff * basic) const {
    if(not basic)
        return Transparent;
    
    const auto single_identity = OPTION(gui_single_identity_color);
    auto base_color = single_identity != Transparent ? single_identity : _id.color();
    
    auto color_source = OPTION(gui_fish_color);
    auto clr = base_color.alpha(255);
    if(single_identity.a != 0) {
        clr = single_identity;
    }
    
    if(color_source != "identity") {
        auto y = Output::Library::get_with_modifiers(color_source, _info, _safe_frame);
        
        if(not GlobalSettings::is_invalid(y)) {
            auto video_size = FindCoord::get().video_size();
            if(color_source == "X")
                y /= video_size.width * slow::cm_per_pixel;
            else if(color_source == "Y")
                y /= video_size.height * slow::cm_per_pixel;
            
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
        
    _color_start = max(sign_cast<int64_t>(_range.start.get()), round(sign_cast<int64_t>(_frame.get()) - FAST_SETTING(frame_rate) * OPTION(gui_max_path_time)));
    _color_end = max(_color_start + 1, (float)_frame.get());
    
    from = max(Frame_t(sign_cast<Frame_t::number_t>(_color_start)), from);
    
    if(_prev_frame_range.start != _range.start
       || _prev_frame_range.end > _range.end)
    {
        frame_vertices.clear();
    }
    
    _prev_frame_range = _range;
    
    const Float2_t max_speed = FAST_SETTING(track_max_speed);
    //const float thickness = OPTION(gui_outline_thickness);
    
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
        auto end = obj.tracklets().end();
        auto begin = obj.tracklets().begin();
        //auto seg = _obj.tracklet_for(i);
        
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
                        .speed_percentage = min(1_F, stuff->centroid.speed<Units::CM_AND_SECONDS>() / max_speed)
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
        auto end = obj.tracklets().end();
        
        for (; i<=to; ++i) {
            if(fit == end || (*fit)->end() < i) {
                //seg = _obj.tracklet_for(i);
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
                        .speed_percentage = min(1_F, stuff->centroid.speed<Units::CM_AND_SECONDS>() / max_speed)
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
        const float thickness = OPTION(gui_outline_thickness);
        
        ///TODO: could try to replace vertices 1by1 and get "did change" for free, before we even
        ///      try to update the object.
        const Float2_t max_distance = SQR(Individual::weird_distance() * 0.1_F / slow::cm_per_pixel);
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
        
        if(_vertices.size() > 1) {
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
            
        } else if(_vertices.size() == 1 && paths_index > 0) {
            --paths_index;
        }
        if(paths_index + 1 < _paths.size())
            _paths.resize(paths_index + 1);
    }
    
    void Fish::update_recognition_circle() {
        if(GUICache::instance().border().in_recognition_bounds(_fish_pos)) {
            if(!_recognition_circle) {
                // is inside bounds, but we didnt know that yet! start animation
                _recognition_circle = std::make_shared<Circle>(Radius{1}, LineClr{Transparent}, FillClr{Cyan.alpha(50)});
            }
            
            auto ts = GUICache::instance().dt();
            float target_radius = 100;
            float percent = min(1, _recognition_circle->radius() / target_radius);
            //Print(_id, " = ", percent, " target:", target_radius);
            
            if(percent < 1.0) {
                percent *= percent;
                
                _recognition_circle->set_pos(_fish_pos - _view.pos());
                _recognition_circle->set_radius(_recognition_circle->radius() + ts * (1 - percent) * target_radius * 2);
                _recognition_circle->set_fill_clr(Cyan.alpha(50 * (1-percent)));
                _view.advance_wrap(*_recognition_circle);
                
            }
            
        } else if(_recognition_circle) {
            _recognition_circle = nullptr;
        }
    }

void Fish::label(const FindCoord& coord, Entangled &e) {
    if(OPTION(gui_highlight_categories)) {
        if(_avg_cat.has_value()) {
            e.add<Circle>(Loc(_view.pos() + _view.size() * 0.5),
                          Radius{_view.size().length()},
                          LineClr{Transparent},
                          FillClr{ColorWheel(_avg_cat.value()).next().alpha(75)});
        } else {
            /*e.add<Circle>(Loc(_view.pos() + _view.size() * 0.5),
                          Radius{_view.size().length()},
                          LineClr{Transparent},
                          FillClr{Purple.alpha(15)});*/
        }
    }
    
    if(OPTION(gui_show_match_modes)) {
        e.add<Circle>(Loc(_view.pos() + _view.size() * 0.5),
                      Radius{_view.size().length()},
                      LineClr{Transparent},
                      FillClr{ColorWheel(_match_mode.has_value() ? (int)_match_mode.value().value() : -1).next().alpha(50)});
    }
    
    //auto bdx = blob->blob_id();
    if(OPTION(gui_show_cliques)) {
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
    
    if (!OPTION(gui_show_texts))
        return;
    
    //if(!_basic_stuff.has_value())
    //    return;
    
    /*std::string color = "";
    std::stringstream text;
    std::string secondary_text;
    

    text << _id.raw_name() << " ";
    
    if (GUI_SETTINGS(gui_show_recognition_bounds)) {
        auto& [valid, tracklet] = _has_processed_tracklet;
        if (valid) {
            auto& [samples, map] = processed_tracklet;
            auto it = std::max_element(map.begin(), map.end(), [](const std::pair<Idx_t, float>& a, const std::pair<Idx_t, float>& b) {
                return a.second < b.second;
            });

            if (it == map.end() || it->first != _id.ID()) {
                color = "str";
                secondary_text += " avg" + Meta::toStr(it->first);
            }
            else
                color = "green";
        } else {
            if(not _pred.empty()) {
                auto map = track::prediction2map(_pred);
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

    auto bdx = _basic_stuff.has_value() ? _basic_stuff->blob.blob_id() : pv::bid();
#if !COMMONS_NO_PYTHON
    auto detection = tags::find(_frame, bdx);
    if (detection.valid()) {
        secondary_text += "<a>tag:" + Meta::toStr(detection.id) + " (" + dec<2>(detection.p).toStr() + ")</a>";
    }
    
    if(_tracklet) {
        auto [id, p, n] = _qr_code;
        if(id >= 0 && p > 0) {
            secondary_text += std::string(" ") + "<a><i>QR:"+Meta::toStr(id)+" ("+dec<2>(p).toStr() + ")</i></a>";
        }
    }
    
    auto c = GUICache::instance().processed_frame().cached(_id.ID());
    if(c) {
        auto cat = c->current_category;
        if(cat.has_value() && cat != _avg_cat) {
            secondary_text += std::string(" ") + "<key>"+_cat_name+"</key>";
        }
    }
    
    if (_cat.has_value() && _cat != _avg_cat) {
        secondary_text += std::string(" ") + (_cat ? "<b>" : "") + "<i>" + _cat_name + "</i>" + (_cat ? "</b>" : "");
    }
    
    if(_avg_cat.has_value()) {
        secondary_text += (_avg_cat.has_value() ? std::string(" ") : std::string()) + "<nr>" + _avg_cat_name + "</nr>";
    }
#endif*/
    /// {if:{not:{has_pred}}:{name}:{if:{equal:{at:0:{max_pred}}:{id}}:<green>{name}</green>:<red>{name}</red> <i>loc</i>[<c><nr>{at:0:{max_pred}}</nr>:<nr>{int:{*:100:{at:1:{max_pred}}}}</nr><i>%</i></c>]}}{if:{tag}:' <a>tag:{tag.id} ({dec:2:{tag.p}})</a>':''}{if:{average_category}:' <nr>{average_category}</nr>':''}{if:{&&:{category}:{not:{equal:{category}:{average_category}}}}:' <b><i>{category}</i></b>':''}
    
    if(not _basic_stuff.has_value())
        return;
    
    auto pos = fish_pos();
    if(_basic_stuff && _basic_stuff->blob.pred.valid()
       && not _basic_stuff->blob.pred.pose.empty())
    {
        pos = _basic_stuff->blob.pred.pose.points.front();
        
        if(auto pose_midline_indexes = SETTING(pose_midline_indexes).value<track::PoseMidlineIndexes>();
           not pose_midline_indexes.indexes.empty())
        {
            if(_basic_stuff->blob.pred.pose.size() > pose_midline_indexes.indexes.front())
            {
                auto point = _basic_stuff->blob.pred.pose.point(pose_midline_indexes.indexes.front());
                pos = point;
            }
        }
    }
    
    if (!_label) {
        _label = new Label(_data->label_text, _basic_stuff->blob.calculate_bounds(), pos);
    }
    else
        _label->set_data(this->frame(), _data->label_text, _basic_stuff->blob.calculate_bounds(), pos);

    //Print("Drawing label for fish ", _id.ID(), " at ", fish_pos(), " with ", _basic_stuff.has_value() ? "blob " + Meta::toStr(_basic_stuff->blob.blob_id()) : "no blob");
    
    e.advance_wrap(*_label);
    
    auto disabled = _frame != _safe_frame;
    _label->set_line_color(disabled ? Gray.alpha(150) : _color);
    _label->update(coord, 1, 0, disabled, GUICache::instance().dt());
}

Drawable* Fish::shadow() {
    auto active = GUICache::instance().active_ids.find(_id.ID()) != GUICache::instance().active_ids.end();
    
    if(OPTION(gui_show_shadows) && active) {
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
