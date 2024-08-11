#include "AnnotationScene.h"
#include <gui/DrawStructure.h>
#include <video/VideoSource.h>
#include <misc/TrackingSettings.h>
#include <misc/Buffers.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/Skelett.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/Action.h>
#include <gui/DynamicGUI.h>
#include <gui/Bowl.h>

namespace cmn::gui {

using namespace dyn;
using namespace track;

// Function to convert BOX annotation to YOLOv8 format
std::string convertBoxToYoloFormat(const Annotation& anno, const Size2& imgSize) {
    // Assertions for programmer errors
    assert(anno.points.size() == 2 && "Box annotation must have exactly 2 points");
    assert(imgSize.width > 0 && imgSize.height > 0 && "Image size must be positive");

    // Exception for user error (invalid data)
    if (anno.points[0].x < 0 || anno.points[0].y < 0 || anno.points[1].x > imgSize.width || anno.points[1].y > imgSize.height) {
        throw InvalidArgumentException("Box coordinates out of image bounds");
    }

    // Normalizing points to range [0, 1]
    float cx = (anno.points[0].x + anno.points[1].x) / 2 / imgSize.width;
    float cy = (anno.points[0].y + anno.points[1].y) / 2 / imgSize.height;
    float width = std::abs(anno.points[1].x - anno.points[0].x) / imgSize.width;
    float height = std::abs(anno.points[1].y - anno.points[0].y) / imgSize.height;

    return Meta::toStr(anno.clid) + " " + Meta::toStr(cx) + " " + Meta::toStr(cy) + " " + Meta::toStr(width) + " " + Meta::toStr(height);
}

// Convert POSE annotation to YOLOv8 format
std::string convertPoseToYoloFormat(const Annotation& anno, const Size2& imgSize) {
    // Assertions for programmer errors
    assert(!anno.points.empty() && "Pose annotation must have at least one point");
    assert(imgSize.width > 0 && imgSize.height > 0 && "Image size must be positive");

    // Calculate bounding box
    uint16_t minX = anno.points[0].x, maxX = anno.points[0].x;
    uint16_t minY = anno.points[0].y, maxY = anno.points[0].y;
    for (const auto& point : anno.points) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);

        // Exception for user error (invalid data)
        if (point.x < 0 || point.x > imgSize.width || point.y < 0 || point.y > imgSize.height) {
            throw InvalidArgumentException("Pose point out of image bounds");
        }
    }

    float cx = (minX + maxX) / 2 / imgSize.width;
    float cy = (minY + maxY) / 2 / imgSize.height;
    float width = (maxX - minX) / imgSize.width;
    float height = (maxY - minY) / imgSize.height;

    std::string output = Meta::toStr(anno.clid) + " " + Meta::toStr(cx) + " " + Meta::toStr(cy) + " " + Meta::toStr(width) + " " + Meta::toStr(height);
    for (const auto& point : anno.points) {
        float normalized_x = point.x / imgSize.width;
        float normalized_y = point.y / imgSize.height;
        output += " " + Meta::toStr(normalized_x) + " " + Meta::toStr(normalized_y);
    }
    return output;
}

// Convert SEGMENTATION annotation to YOLOv8 format
std::string convertSegmentationToYoloFormat(const Annotation& anno, const Size2& imgSize) {
    // Assertions for programmer errors
    assert(anno.points.size() >= 3 && "Segmentation annotation must have at least 3 points for a valid polygon");
    assert(imgSize.width > 0 && imgSize.height > 0 && "Image size must be positive");

    std::string output = Meta::toStr(anno.clid);

    // Iterate through points to construct the bounding polygon
    for (const auto& point : anno.points) {
        // Exception for user error (invalid data)
        if (point.x < 0 || point.x > imgSize.width || point.y < 0 || point.y > imgSize.height) {
            throw InvalidArgumentException("Segmentation point out of image bounds");
        }

        float normalized_x = point.x / imgSize.width;
        float normalized_y = point.y / imgSize.height;
        output += " " + Meta::toStr(normalized_x) + " " + Meta::toStr(normalized_y);
    }

    return output;
}

// Function to determine the most common annotation type
AnnotationType findMostCommonAnnotationType(const std::vector<Annotation>& annotations) {
    std::unordered_map<AnnotationType, int> typeCounts;
    for (const auto& anno : annotations) {
        typeCounts[anno.type]++;
    }

    return std::max_element(typeCounts.begin(), typeCounts.end(),
        [](const std::pair<AnnotationType, int>& a, const std::pair<AnnotationType, int>& b) {
            return a.second < b.second;
        })->first;
}

// Updated exportAnnotationsToYolo function
void exportAnnotationsToYolo(const std::vector<Annotation>& annotations, const Size2& imgSize, const std::string& outputFile, std::optional<AnnotationType> exportType = std::nullopt) {
    // Determine the export type if not specified
    AnnotationType typeToExport = exportType.has_value() ? exportType.value() : findMostCommonAnnotationType(annotations);

    std::ofstream file(outputFile);

    for (const auto& anno : annotations) {
        if (anno.type != typeToExport) {
            Print("Skipping annotation of type ", static_cast<int>(anno.type), ", not matching export type ", static_cast<int>(typeToExport), "\n");
            continue;
        }

        std::string yoloFormatLine;
        switch (anno.type) {
            case AnnotationType::BOX:
                yoloFormatLine = convertBoxToYoloFormat(anno, imgSize);
                break;
            case AnnotationType::POSE:
                yoloFormatLine = convertPoseToYoloFormat(anno, imgSize);
                break;
            case AnnotationType::SEGMENTATION:
                yoloFormatLine = convertSegmentationToYoloFormat(anno, imgSize);
                break;
        }
        file << yoloFormatLine << std::endl;
    }

    file.close();
}

void AnnotationView::set_annotation(Annotation && a) {
    if(a.type == AnnotationType::BOX) {
        _rect = std::make_unique<Rect>(
            Box{
                Vec2(a.points.front()),
                Size2(Vec2(a.points.back()) - Vec2(a.points.front()))
            },
            FillClr{Red.alpha(50)},
            LineClr{Red.alpha(125)});
        
        _circles.clear();
        for(auto &p : a.points) {
            _circles.push_back(std::make_shared<Circle>(Loc{p}, Radius{15}, FillClr{Red.alpha(50)}, LineClr{Red.alpha(125)}));
            _circles.back()->set_draggable();
            _circles.back()->on_hover([ptr = _circles.back().get()](Event e) {
                if(e.hover.hovered)
                    ptr->set(FillClr{Red.exposureHSL(1.5).alpha(125)});
                else
                    ptr->set(FillClr{Red.alpha(50)});
            });
            _circles.back()->on_click([this](Event e) {
                if(not e.mbutton.pressed) {
                    if(_rect) {
                        assert(_circles.size() == 2);
                        //_rect->set(Box{_circles.front()->pos(), Size2(_circles.back()->pos() - _circles.front()->pos())});
                    }
                }
            });
        }
        
    } else {
        _circles.clear();
        for(auto &p : a.points) {
            _circles.push_back(std::make_shared<Circle>(Loc{p}, Radius{15}, FillClr{Red.alpha(50)}, LineClr{Red.alpha(125)}));
            _circles.back()->set_draggable();
        }
    }
    
    _a = std::move(a);
}

void AnnotationView::update() {
    begin();
    if(_rect) {
        _rect->set(Box{_circles.front()->pos(), Size2(_circles.back()->pos() - _circles.front()->pos())});
    }
    
    auto it = _circles.begin();
    auto bds = Bounds::combine_all([&]() -> std::optional<Bounds> {
        if(it == _circles.end())
            return std::nullopt;
        auto b = Bounds((*it)->pos(), Size2(1));
        ++it;
        return b;
    });
    
    add<Rect>(Box{bds}, FillClr{Yellow.alpha(50)}, LineClr{Yellow.alpha(150)});
    for(auto &c : _circles)
        advance_wrap(*c);
    if(_rect)
        advance_wrap(*_rect);
    
    end();
}

void AnnotationView::init() {
    _rect = nullptr;
    _circles.clear();
}

// Constructor implementation
AnnotationScene::AnnotationScene(Base& window)
: 
    Scene(window, "annotation-scene", [this](Scene&, DrawStructure& base){
        _draw(base);
    }),
    currentFrameIndex(0),
    _bowl(nullptr),
    _current_image(std::make_unique<ExternalImage>()),
    _gui(std::make_unique<dyn::DynamicGUI>())
{
}

// Method to retrieve a frame (placeholder, needs actual implementation)
Image::Ptr AnnotationScene::retrieveFrame(Frame_t) {
    // Implement frame retrieval logic
    return nullptr; // Placeholder
}

// Activate method implementation
void AnnotationScene::activate() {
    Scene::activate();
    // Logic to activate the scene, e.g., initializing framePreloader
    auto source = SETTING(source).value<file::PathArray>();
    Print("Loading source = ", source);
    
    std::unique_lock guard(_video_mutex);
    _video = std::make_unique<VideoSource>(source);
    _video->set_colors(ImageMode::RGB);
    
    video_length = _video->length();
    video_size = _video->size();
    
    _skeleton = SETTING(detect_skeleton).value<Pose::Skeleton>();
    _pose_in_progress = {};
    
    SETTING(frame_rate) = Settings::frame_rate_t(_video->framerate() != short(-1) ? _video->framerate() : 25);
    
    assert(not _frame_future.valid()); // should have been cleared upon deactivate
    _frame_future = select_unique_frames();
    
    assert(not _next_frame.valid());
}

// Deactivate method implementation
void AnnotationScene::deactivate() {
    Scene::deactivate();
    
    try {
        if(_frame_future.valid()) {
            _frame_future.get();
        }
        
    } catch(const std::exception& e) {
        std::unique_lock guard(_video_mutex);
        FormatExcept("Error retrieving unique frames from ", _video->source(), ": ", e.what());
    }
    
    // Logic to deactivate the scene
    _gui->clear();
    _bowl = nullptr;
    
    std::unique_lock guard(_video_mutex);
    _video = nullptr;
    _selected_frames.clear();
}

// Custom drawing implementation
void AnnotationScene::_draw(DrawStructure& graph) {
    if(window()) {
        //auto update = FindCoord::set_screen_size(graph, *window()); //.div(graph.scale().reciprocal() * gui::interface_scale());
        //
        FindCoord::set_video(video_size);
        //if(update != window_size)
         //   window_size = update;
    }

    auto coord = FindCoord::get();
    if (not _bowl) {
        _bowl = std::make_unique<Bowl>(nullptr);
        _bowl->set_video_aspect_ratio(coord.video_size().width, coord.video_size().height);
        _bowl->fit_to_screen(coord.screen_size());
    }
    
    if(_frame_future.valid()) {
        if(_frame_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) 
        {
            // the list of viable frames is ready
            try {
                _selected_frames = _frame_future.get();
                _next_frame = retrieve_next_frame();
                
            } catch(const std::exception& e) {
                std::unique_lock guard(_video_mutex);
                FormatExcept("Cannot retrieve unique frames from ", _video->source(),": ", e.what());
            }
        }
    }
    
    if(not _current_image->source()
       && _next_frame.valid())
    {
        if(_next_frame.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            auto image = _next_frame.get();
            if(image) {
                currentFrameIndex = Frame_t(image->index());
                _loaded_frames[currentFrameIndex] = std::move(image);
            }
            _next_frame = retrieve_next_frame();
        }
        
    } else if(not _current_image->source())
        _next_frame = retrieve_next_frame();
    
    if(not *_gui) {
        *_gui = DynamicGUI{
            .gui = nullptr,
            .path = "annotation_layout.json",
            .context = [&](){
                dyn::Context context;
                context.actions = {
                    
                };

                context.variables = {
                    VarFunc("window_size", [](const VarProps&) -> Vec2 {
                        return FindCoord::get().screen_size();
                    }),
                    VarFunc("status", [this](const VarProps&) -> int 
                    {
                        if(_frame_future.valid()) {
                            return 0;
                        }
                        return 1; // frames retrieved
                    }),
                    VarFunc("frame", [this](const VarProps&) {
                        return currentFrameIndex;
                    }),
                    VarFunc("2hud", [](const VarProps& props) {
                        auto coords = FindCoord::get();
                        auto p = Meta::fromStr<Vec2>(props.parameters.front());
                        return coords.convert(BowlCoord(p));
                    }),
                    VarFunc("2bowl", [](const VarProps& props) {
                        auto coords = FindCoord::get();
                        auto p = Meta::fromStr<Vec2>(props.parameters.front());
                        return coords.convert(HUDCoord(p));
                    }),
                    VarFunc("annotations", [this](const VarProps& props) -> std::vector<std::shared_ptr<VarBase_t>>& {
                        if(props.parameters.empty())
                            throw InvalidArgumentException("Not enough arguments for annotations. Need frame.");
                        auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                        auto it = _gui_annotations.find(frame);
                        if(it == _gui_annotations.end()
                           && frame.valid())
                        {
                            // have to create a new one
                            auto a = annotations.find(frame);
                            if(a != annotations.end()) {
                                auto objs = a->second.getAllObjects();
                                auto &data = _gui_data[frame];
                                auto &anns = _gui_annotations[frame];
                                
                                data.clear();
                                
                                for(auto &[_, ann] : objs) {
                                    size_t index = data.size();
                                    data.emplace_back();
                                    auto &map = data.back();
                                    map["index"] = index;
                                    map["type"] = (int)ann.type;
                                    
                                    std::vector<Vec2> pts(ann.points.begin(), ann.points.end());
                                    map["points"] = pts;
                                    if(anns.size() < data.size())
                                        anns.emplace_back(new Variable([index, &data](const VarProps&) -> auto& {
                                            return data.at(index);
                                        }));
                                }
                                
                                if(data.size() < anns.size())
                                    anns.resize(data.size());
                                
                                return anns;
                                
                            } else {
                                return _gui_annotations[frame];
                            }
                            
                        } else
                            return it->second;
                    })
                };

                return context;
            }(),
            .base = window()
        };
        
        _gui->context.custom_elements["pose"] = std::unique_ptr<CustomElement>(new CustomElement {
            .name = "pose",
            .create = [this](LayoutContext& layout) -> Layout::Ptr {
                std::shared_ptr<Skelett> ptr;
                auto points = layout.get<std::vector<Pose::Point>>(std::vector<Pose::Point>{}, "points");
                //auto color = layout.textClr;
                auto line = layout.line;
                auto fill = layout.fill;

                if (not ptr) {
                    ptr = std::make_shared<Skelett>();
                    ptr->set_skeleton(_skeleton);
                }

                //ptr->set(color);
                ptr->set(FillClr{ fill });
                //ptr->set(LineClr{ line });
                ptr->set_color(line);
                
                //Print("Creating new skelett with points ", points);
                
                auto coords = FindCoord::get();
                for(auto &pt : points)
                    pt = coords.convert(BowlCoord(pt));
                
                Pose pose{
                    .points = std::move(points)
                };
                ptr->set_pose(pose);
                //Print("Create new label with text = ", text);
                
                return Layout::Ptr(ptr);
            },
            .update = [](Layout::Ptr& o, const Context& context, State& state, const auto& patterns) -> bool {
                //Print("Updating label with patterns: ", patterns);
                //Print("o = ", o.get());

                /*Idx_t id;
                if (patterns.contains("id"))
                    id = Meta::fromStr<Idx_t>(parse_text(patterns.at("id").original, context, state));
                
                if (id.valid()) {
                    auto it = _labels.find(id);
                    if (it != _labels.end()) {
                        if(it->second.get() != o.get())
                            o = Layout::Ptr(it->second);
                    }
                }

                auto p = o.to<Label>();
                auto source = p->source();
                auto pos = source.pos();
                auto center = p->center();
                auto text = p->text()->text();

                if(patterns.contains("text"))
                    text = parse_text(patterns.at("text").original, context, state);
                if (patterns.contains("pos")) {
                    pos = Meta::fromStr<Vec2>(parse_text(patterns.at("pos").original, context, state));
                }
                if (patterns.contains("size")) {
                    source = Bounds(pos, Meta::fromStr<Size2>(parse_text(patterns.at("size").original, context, state)));
                }
                if (patterns.contains("center")) {
                    center = Meta::fromStr<Vec2>(parse_text(patterns.at("center").original, context, state));
                } else
                    center = source.pos()+ Vec2(source.width, source.height) * 0.5;

                if(patterns.contains("line"))
                    p->set_line_color(Meta::fromStr<Color>(parse_text(patterns.at("line").original, context, state)));
                if (patterns.contains("fill"))
                    p->set_fill_color(Meta::fromStr<Color>(parse_text(patterns.at("fill").original, context, state)));
                if(patterns.contains("color"))
                    p->text()->set(TextClr{ Meta::fromStr<Color>(parse_text(patterns.at("color").original, context, state)) });

                p->set_data(0_f, text, source, center);
                p->update(FindCoord::get(), 1, 1, false, dt, Scale{1});*/
                
                auto p = o.to<Skelett>();
                
                if(patterns.contains("line"))
                    p->set_color(Meta::fromStr<Color>(parse_text(patterns.at("line").original, context, state)));
                if (patterns.contains("fill"))
                    p->set(FillClr{ Meta::fromStr<Color>(parse_text(patterns.at("fill").original, context, state)) });
                if(patterns.contains("points")) {
                    auto points = Meta::fromStr<std::vector<Pose::Point>>(parse_text(patterns.at("points").original, context, state));
                    
                    auto coords = FindCoord::get();
                    for(auto &pt : points)
                        pt = coords.convert(BowlCoord(pt));
                    
                    Print("Setting skelett to ", points);
                    Pose pose{
                        .points = std::move(points)
                    };
                    p->set_pose(pose);
                }
                
                return true;
            }
        });
    }
    
    graph.wrap_object(*_current_image);
    
    graph.wrap_object(*_bowl);
    _bowl->update_scaling(_timer.elapsed());
    _timer.reset();
    
    _bowl->fit_to_screen(coord.screen_size());
    _bowl->set_target_focus({});
    
    auto coords = FindCoord::get();
    _bowl->update(currentFrameIndex, graph, coords);
    
    _current_image->set_scale(_bowl->_current_scale);
    _current_image->set_pos(_bowl->_current_pos);
    
    graph.section("elements", [&](auto&, Section* s) {
        s->set_scale(_bowl->_current_scale);
        s->set_pos(_bowl->_current_pos);
        
        if(_drag_box)
            graph.wrap_object(*_drag_box);
        
        if(not _pose_in_progress.points.empty()) {
            for(auto &pt : _pose_in_progress.points) {
                graph.circle(Loc{pt}, Radius{5}, FillClr{Red.alpha(125)});
            }
        }
        
        if(_view_frame != currentFrameIndex) {
            _views.clear();
        }
        
        if(auto it = annotations.find(currentFrameIndex);
           it != annotations.end())
        {
            auto anns = it->second.getAllObjects();
            for(auto [ID, obj] : anns) {
                if(not _views.contains(ID)) {
                    auto [it, r] = _views.emplace(ID, new AnnotationView);
                    if(not r)
                        throw InvalidArgumentException("Cannot insert ", ID, " into the map.");
                    it->second->set_annotation(std::move(obj));
                }
            }
            
            _view_frame = currentFrameIndex;
        }
        
        for(auto &[ID, o] : _views)
            graph.wrap_object(*o);
    });
    
    graph.section("gui", [this, &graph](DrawStructure &, Section *){
        _gui->update(graph, nullptr);
    });
}

std::future<std::unordered_set<Frame_t>> AnnotationScene::select_unique_frames() {
    return std::async(std::launch::async, [this](){
        std::unordered_set<Frame_t> indexes;
        
        Frame_t L;
        Size2 size;
        std::vector<Image::Ptr> savedImages;
        std::vector<cv::Mat> featureVectors;
        {
            std::unique_lock guard(_video_mutex);
            L = _video->length();
            size = _video->size();
        }
        
        auto to_feature = [](const Image::Ptr& image) {
            auto mat = image->get();
            cv::Mat feature;
            cv::resize(mat, feature, Size2(50));
            feature = feature.reshape(1,1);
            feature.convertTo(feature, CV_32F);
            return feature;
        };
        
        // select N frames uniformly samples
        auto step = max(1u, L.get() / min(100u, max(1u, uint32_t(L.get() * 0.1))));
        Print("Moving through the video at ", step, " images step size (",L.get() / step," images).");
        
        struct ImageMaker {
            Image::Ptr operator()() {
                return Image::Make();
            }
        };
        ImageBuffers<Image::Ptr, ImageMaker> buffers("select_unique_frames", size);
        for (Frame_t i = 0_f; i < L; i += Frame_t(step)) {
            auto output = buffers.get(source_location::current());
            output->create(size.height, size.width, 4);
            
            {
                std::unique_lock guard(_video_mutex);
                _video->frame(i, *output);
            }
            
                //tf::imshow("first", output->get());
            thread_print("Reading ", i, "/", L);
            auto feature = to_feature(output);
            savedImages.emplace_back(std::move(output));
            featureVectors.emplace_back(std::move(feature));
        }
        
        // K-means clustering
        int K = 50; // Number of clusters
        cv::Mat labels, centers;
        cv::Mat samples;
        cv::vconcat(featureVectors, samples); // Concatenate all feature vectors
        cv::kmeans(samples, K, labels,
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
                   3, cv::KMEANS_PP_CENTERS, centers);
        
        // Struct to hold frame index and its distance to the centroid
        struct FrameDistance {
            int index;
            int real_index;
            float distance;
            bool operator<(const FrameDistance& other) const {
                return distance < other.distance;
            }
        };

        // Map each cluster to its frames and their distances
        std::map<int, std::vector<FrameDistance>> clusters;
        for (int i = 0; i < samples.rows; ++i) {
            int clusterIdx = labels.at<int>(i);
            float dist = cv::norm(samples.row(i), centers.row(clusterIdx), cv::NORM_L2);
            clusters[clusterIdx].push_back({i, savedImages.at(i)->index(), dist});
        }

        // Sorting frames within each cluster
        for (auto& [clusterIdx, frameList] : clusters) {
            std::sort(frameList.begin(), frameList.end());
        }

        // Selecting top M frames uniformly from N clusters
        int N = narrow_cast<int>(clusters.size()); // Number of clusters
        int M = 100; // Total number of frames to select
        int framesPerCluster = M / N; // Frames to select from each cluster
        std::vector<Image::Ptr> selectedFrames;
        
        for (const auto& [clusterIdx, frameList] : clusters) {
            int count = 0;
            for (size_t i = 0; i < frameList.size(); i += max(1u, frameList.size() / framesPerCluster)) {
                if (count < framesPerCluster) {
                    int frameIndex = frameList[i].real_index;
                    int index = frameList[i].index;
                    
                    if(not indexes.contains(Frame_t(frameIndex))) {
                        auto image = std::move(savedImages[index]);
                        if(not image)
                            Print("Image not present: ", i);
                        else {
                            if(frameIndex != image->index())
                                Print("Not the same: ", frameIndex, " != ", image->index());
                            indexes.insert(Frame_t(image->index()));
                            _loaded_frames[Frame_t(image->index())] = std::move(image);
                            //selectedFrames.emplace_back(std::move(image));
                            count++;
                        }
                    } else
                        Print("Frame ", frameIndex, " is already in ", indexes);
                }
            }
        }
        
        Print("selected indexes = ", indexes);
        return indexes;
    });
}

// Handling global events for video navigation
bool AnnotationScene::on_global_event(Event event) {
    auto graph = _bowl && _bowl->stage() ? _bowl->stage() : nullptr;
    if(event.type == EventType::MMOVE
       && graph
       && graph->is_mouse_down(0)
       && not graph->selected_object())
    {
        auto p = Vec2(event.move.x, event.move.y);
        
        auto coords = FindCoord::get();
        if(not _drag_box)
            _drag_box = std::make_unique<Rect>(Loc{coords.convert(HUDCoord{p})});
        
        auto pos = _drag_box->pos();
        Print(pos, " => ", coords.convert(HUDCoord(p)) - pos);
        _drag_box->create(Size{coords.convert(HUDCoord(p)) - pos + Vec2(1)}, FillClr{Red.alpha(50)});
    }

    if(event.type == EventType::MBUTTON
       && event.mbutton.button == 0
       && not event.mbutton.pressed)
    {
        auto p = Vec2(event.mbutton.x, event.mbutton.y);
        auto coord = FindCoord::get();
        Print("adding point at ", p, " => ", coord.convert(HUDCoord(p)));
        p = coord.convert(HUDCoord(p));
        
        if(_drag_box
           && _drag_box->size().length() >= 2)
        {
            Annotation a{
                .type = AnnotationType::BOX,
                .points = {
                    _drag_box->pos(),
                    _drag_box->pos() + _drag_box->size()
                }
            };
            
            addAnnotation(currentFrameIndex, std::move(a));
            
        } else {
            if(_pose_in_progress.points.empty()) {
                _pose_in_progress = Annotation{
                    .type = AnnotationType::POSE
                };
            }
            
            _pose_in_progress.points.push_back(Pose::Point(p));
        }
        
        _drag_box = nullptr;
        
    } else if(event.type == EventType::MBUTTON
              && event.mbutton.button == 1
              && event.mbutton.pressed)
    {
        _drag_box = nullptr;
        
        if(not _pose_in_progress.points.empty()) {
            addAnnotation(currentFrameIndex, std::move(_pose_in_progress));
        }
        _pose_in_progress = {};
    }
    
    if(event.type == EventType::KEY
       && event.key.pressed) 
    {
        switch (event.key.code) {
            case Codes::Left:
                // retrieve a frame that is the highest frame before the current frame
                if(currentFrameIndex > 0_f) {
                    Frame_t closestFrame;
                    for (const auto& frame : _selected_frames) {
                        if (frame < currentFrameIndex) {
                            if (not closestFrame.valid() || frame > closestFrame) {
                                closestFrame = frame;
                            }
                        }
                    }

                    if (closestFrame.valid()) {
                        Print("Navigating to frame ", closestFrame);
                        navigateToFrame(closestFrame);
                    }
                }
                break;
                
            case Codes::Right:
                // retrieve a frame that is the lowest frame after the current frame
                if(currentFrameIndex < video_length) { // Assuming MAX_FRAME_INDEX is the upper limit for frame index
                    Frame_t closestFrame;
                    for (const auto& frame : _selected_frames) {
                        if (frame > currentFrameIndex) {
                            if (not closestFrame.valid() || frame < closestFrame) {
                                closestFrame = frame;
                            }
                        }
                    }

                    if (closestFrame.valid()) {
                        Print("Navigating to frame ", closestFrame);
                        navigateToFrame(closestFrame);
                    }
                }
                break;

                
            default:
                break;
        }
    }
    
    return false; // Return true if the event is handled
}

// Methods to manage annotations
AnnotationScene::Manager::ID AnnotationScene::addAnnotation(Frame_t frameNumber, Annotation&& pose) {
    if(auto it = _gui_annotations.find(frameNumber);
       it != _gui_annotations.end())
    {
        _gui_annotations.erase(it);
    }
    return annotations[frameNumber].registerObject(std::move(pose));
}

void AnnotationScene::removeAnnotation(Frame_t frameNumber, Manager::ID id) {
    if(annotations.contains(frameNumber)) {
        if(auto it = _gui_annotations.find(frameNumber);
           it != _gui_annotations.end())
        {
            _gui_annotations.erase(it);
        }
        annotations.at(frameNumber).unregisterObject(id);
    }
}

const Annotation& AnnotationScene::getAnnotation(Frame_t frameNumber, Manager::ID id) const {
    auto it = annotations.find(frameNumber);
    if (it != annotations.end()) {
        return it->second.getObject(id);
    }
    throw InvalidArgumentException("Cannot find id ", id, " in frame ", frameNumber);
}

// Method to handle frame navigation
void AnnotationScene::navigateToFrame(Frame_t frameIndex) {
    currentFrameIndex = frameIndex;
    auto it = _loaded_frames.find(frameIndex);
    if(it != _loaded_frames.end()) {
        _current_image->set_source(Image::Make(*it->second));
    } else {
        /*auto image = retrieveFrame(frameIndex);
        if(image)
            _current_image->set_source(Image::Make(*image));*/
    }
}

std::future<Image::Ptr> AnnotationScene::retrieve_next_frame() {
    Frame_t index;
    if(_selected_frames.empty())
        throw InvalidArgumentException("No frames selected");
    auto it = _selected_frames.begin();
    index = *it;
    _selected_frames.erase(it);
    Print("Preloading frame ", index);
    
    return std::async(std::launch::async, [this](Frame_t index){
        std::unique_lock guard(_video_mutex);
        auto image = std::make_unique<Image>();
        _video->frame(index, *image);
        Print("Frame ", index," loaded.");
        return image;
    }, index);
}

// More methods as required...

} // namespace gui
