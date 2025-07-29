#include "Export.h"

#include <tracking/Tracker.h>
#include <tracking/OutputLibrary.h>
#include <misc/cnpy_wrapper.h>
#include <tracking/MemoryStats.h>
#include <file/DataLocation.h>
#include <gui/IdentityHeatmap.h>
#include <tracking/FilterCache.h>
#include <tracking/VisualIdentification.h>
#include <tracking/IndividualManager.h>
#include <processing/PadImage.h>
#include <gui/DrawGraph.h>
#include <misc/DetectionTypes.h>

#if WIN32
#include <io.h>

#define access(X, Y) _access(X, Y)
#define W_OK 2
#endif

namespace track {

    void temporary_save(file::Path path, std::function<void(file::Path)> fn) {
        /**
         * There is sometimes a problem when people save to network harddrives.
         * The first NPY file will not finish writing / sync completely before the next one starts.
         * This leads to "does not contain a ZIP file" exception and terminates the saving process.
         * Instead, we move the file to a temporary folder first (on our local harddrive) and then
         * move it.
         * (Only if a /tmp/ folder exists though.)
         */

        file::Path final_path = path;
        file::Path tmp_path, use_path;

#ifdef WIN32
        char chPath[MAX_PATH];
        if (GetTempPath(MAX_PATH, chPath))
            tmp_path = chPath;
#else
        // Determine temporary directory: environment variable or fallback
        static const char* env_tmp = std::getenv("TMPDIR");
        if (env_tmp && file::Path(env_tmp).exists() && access(env_tmp, W_OK) == 0) {
            tmp_path = env_tmp;
        } else if (file::Path("/tmp").exists() && access("/tmp", W_OK) == 0) {
            tmp_path = "/tmp";
        } else {
            // no writable temp directory available; proceed without a tmp path
        }
#endif

        if (not tmp_path.empty() && tmp_path.exists()) {
            if (access(tmp_path.c_str(), W_OK) == 0)
                use_path = tmp_path / path.filename();
        }

        try {
            fn(use_path);

            static std::mutex mutex;
            std::lock_guard guard(mutex);
            if (final_path != use_path) {
                if (!use_path.move_to(final_path)) {
                    throw U_EXCEPTION("Cannot move file '",use_path.str(),"' to '",final_path.str(),"'.");
                } //else
            }

        }
        catch (const std::exception& ex) {
            FormatExcept("Problem copying file ",use_path.str()," to ",final_path.str(),": ",ex.what(),"");
            // there will be a utils exception, so its printed out already
        }
    }

namespace hist_utils {
    /**
     * Taken from: https://stackoverflow.com/questions/38910945/pixel-wise-median-of-sequence-of-cvmats
     */
    using namespace std;
    using namespace cv;
    
    struct Hist {
        vector<short> h;
        int count;
        Hist() : h(256, 0), count(0) {};
    };
    
    void addImage(Mat1b& img, vector<vector<Hist>>& M, Mat1b& med)
    {
        assert(img.rows == med.rows);
        assert(img.cols == med.cols);
        
        for (uint r = 0; r < (uint)img.rows; ++r) {
            for (uint c = 0; c < (uint)img.cols; ++c){
                
                // Add pixel to histogram
                Hist& hist = M[r][c];
                ++hist.h[img((int)r, (int)c)];
                ++hist.count;
                
                // Compute median
                uint i;
                int n = hist.count / 2;
                for (i = 0; i < 256 && ((n -= hist.h[i]) >= 0); ++i);
                
                // 'i' is the median value
                med((int)r,(int)c) = uchar(i);
            }
        }
        
        // Add image to my list
        //images.push_back(img.clone());
    }
    
    /*void remImage(vector<Mat1b>& images, int idx, vector<vector<Hist>>& M, Mat1b& med)
    {
        assert(idx >= 0 && idx < images.size());
        
        Mat1b& img = images[idx];
        for (int r = 0; r < img.rows; ++r) {
            for (int c = 0; c < img.cols; ++c){
                
                // Remove pixel from histogram
                Hist& hist = M[r][c];
                --hist.h[img(r, c)];
                --hist.count;
                
                // Compute median
                int i;
                int n = hist.count / 2;
                for (i = 0; i < 256 && ((n -= hist.h[i]) >= 0); ++i);
                
                // 'i' is the median value
                med(r, c) = uchar(i);
            }
        }
        
        // Remove image from list
        images.erase(images.begin() + idx);
    }*/
    
    void init(vector<vector<Hist>>& M, Mat1b& med, int rows, int cols)
    {
        med = Mat1b(rows, cols, uchar(0));
        M.clear();
        M.resize((uint)rows);
        for (uint i = 0; i < (uint)rows; ++i) {
            M[i].resize((uint)cols);
        }
    }
}

void export_data(pv::File& video, Tracker& tracker, Idx_t fdx, const Range<Frame_t>& range, const std::function<void(float, std::string_view)>& progress_callback) {
    using namespace gui;
    using namespace track::image;
    
    GenericThreadPool _blob_thread_pool(cmn::hardware_concurrency(), "export_pool");
    LockGuard guard(ro_t{}, "GUI::export_tracks");
    
    // save old values and remove all calculation/scaling options from output
    auto previous_graphs = SETTING(output_fields).value<default_config::graphs_type>();
    auto previous_options = SETTING(output_default_options).value<default_config::default_options_type>();
    
    Output::Library::remove_calculation_options();
    
    auto previous_output_frame_window = SETTING(output_frame_window).value<uint32_t>();
    auto output_tracklet_images = SETTING(output_tracklet_images).value<bool>();
    auto output_format = SETTING(output_format).value<default_config::output_format_t::Class>();
    auto output_posture_data = SETTING(output_posture_data).value<bool>();
    auto output_min_frames = SETTING(output_min_frames).value<uint16_t>();
    auto no_tracking_data = SETTING(auto_no_tracking_data).value<bool>();
    auto auto_no_memory_stats = SETTING(auto_no_memory_stats).value<bool>();
    
    const auto normalize = default_config::valid_individual_image_normalization();
    
    if(no_tracking_data) {
        FormatWarning("Not saving tracking data because of 'auto_no_tracking_data' flag being set.");
    }
    //auto calculate_posture = SETTING(calculate_posture).value<bool>();
    
    const Size2 output_size = SETTING(individual_image_size);
    const bool do_normalize_tracklets = SETTING(tracklet_normalize).value<bool>();
    const bool do_normalize_output = SETTING(output_normalize_midline_data).value<bool>();
    const uint16_t tracklet_max_images = SETTING(tracklet_max_images);
    
    auto data_prefix = SETTING(data_prefix).value<file::Path>();
    auto fishdata = file::DataLocation::parse("output", data_prefix);
    if(!fishdata.exists())
        if(!fishdata.create_folder())
            throw U_EXCEPTION("Cannot create folder ",fishdata.str()," for saving fishdata.");
    
    file::Path input = SETTING(filename).value<file::Path>().filename();
    if(input.has_extension("pv"))
        input = input.remove_extension();
    
    std::string filename = input.str();
    
    auto posture_path = (fishdata / (filename + "_posture_*.npz")).str();
    auto recognition_path = (fishdata / (filename + "_recognition_*.npz")).str();
    
    if(!range.empty())
        Print("[exporting] Exporting range [", range.start,"-",range.end,"]");
    else
        Print("[exporting] Exporting all frames (", tracker.number_frames(),")");
    auto individual_prefix = FAST_SETTING(individual_prefix);
    Print("[exporting] Writing data from `output_fields` to ",fishdata / (filename+"_"+individual_prefix+"*."+output_format.str()));
    if(output_posture_data)
        Print("[exporting] Writing posture data to ",posture_path);
    Print("[exporting] Writing recognition data to ",recognition_path);
    
    struct ResetOutputFields {
        std::vector<std::pair<std::string, std::vector<std::string>>> original_output_fields;
        ResetOutputFields() {
            original_output_fields = SETTING(output_fields).value<decltype(original_output_fields)>();
        }
        ~ResetOutputFields() {
            SETTING(output_fields) = original_output_fields;
        }
    } reset_output_fields;
    
    if (auto detect_classes = SETTING(detect_classes).value<blob::MaybeObjectClass_t>();
        detect_classes.has_value()
        && not detect_classes->empty())
    {
        auto fields = SETTING(output_fields)
            .value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
        auto initial_fields = fields; // copy
        
        if(BOOL_SETTING(output_auto_pose)) {
            // Generate only the missing ones
            auto new_pose_fields = default_config::add_missing_pose_fields();
            
            // If there’s nothing new to add, we’re done
            if (!new_pose_fields.empty())
            {
                // Insert the new (missing) ones
                fields.insert(fields.end(), new_pose_fields.begin(), new_pose_fields.end());
            }
        }
        
        if(BOOL_SETTING(output_auto_detection_fields)) {
            bool found = false;
            for(auto &[name, v] : fields) {
                if(name == "detection_p") {
                    found = true;
                }
            }
            
            if(not found) {
                fields.push_back({"detection_p", {}});
            }
        }
        
        // Update the setting if required
        if(fields != initial_fields) {
            SETTING(output_fields) = std::move(fields);
        }
    }
    
    Output::Library::Init();
    {
        std::set<std::string> keys;
        for(auto &[key, _] : reset_output_fields.original_output_fields) {
            keys.insert(key);
        }
        Print("[exporting] functions: ", keys);
    }
    DebugHeader("...");
    
    Output::cached_output_fields_t cached_output_fields = Output::Library::get_cached_fields();
    
    try {
        std::map<Idx_t, float> all_percents;
        std::mutex percent_mutex;
        IndividualManager::transform_all([&all_percents](auto fdx, auto) {
            all_percents[fdx] = 0;
        });
        
        struct ImageData {
            pv::BlobPtr blob;
            Idx_t fdx;
            gui::Transform midline_transform;
            Float2_t median_midline_length_px{0_F};
            Range<Frame_t> range;
        };
        std::map<Frame_t, std::map<Idx_t, ImageData>> waiting_pixels;
        std::mutex sync;
        
        std::vector<std::shared_ptr<PropertiesGraph>> fish_graphs;
        std::vector<Output::LibraryCache::Ptr> library_cache;
        float last_percent = -1;
        
        auto work_item = [&](size_t thread_index, Idx_t id, const Individual* fish){
            if(fdx.valid() && fdx != id)
                return;
            
            //if(SETTING(terminate))
            //    return;
            
            std::function<void(float)> callback = [id, &percent_mutex, &all_percents, &last_percent, &fishdata, output_posture_data, &progress_callback](float percent) {
                float overall_percent = 0;
                
                {
                    std::lock_guard<std::mutex> guard(percent_mutex);
                    
                    for(auto && [k, p] : all_percents) {
                        if(k == id)
                            p = percent;
                        overall_percent += p;
                    }
                    //added_frames += print_step_size;
                   // added_frames += counter % print_step_size;
                    
                    //if(GUI::instance())
                    //{
                    overall_percent = overall_percent / all_percents.size() * (output_posture_data ? 0.5f : 1.0f);
                    progress_callback(overall_percent, "");
                        //WorkProgress::set_percent(overall_percent / all_percents.size() * (output_posture_data ? 0.5f : 1.0f));
                        //overall_percent = WorkProgress::percent();
                    //} else
                    //    overall_percent = overall_percent / (float)all_percents.size() * (output_posture_data ? 0.5f : 1.0f);
                }
                
                // synchronize with debug messages
                //std::lock_guard<std::mutex> lock(DEBUG::debug_mutex());
                static std::mutex _mutex;
                if(std::lock_guard guard(_mutex);
                   cmn::abs(last_percent - overall_percent) >= 0.05)
                {
                    last_percent = overall_percent;
                    overall_percent *= 100;
                    
                    size_t i;
                    printf("[");
                    for(i=0; i<overall_percent * 0.5; ++i) {
                        printf("=");
                    }
                    for(; i<100 * 0.5; ++i) {
                        printf(" ");
                    }
                    printf("] %.2f%% exported (to '%s/...)\r", overall_percent, fishdata.str().c_str());
                    fflush(stdout);
                }
            };
            
            if (fish->frame_count() >= output_min_frames) {
                if(!no_tracking_data) {
                    if(!range.empty())
                        fish_graphs.at(thread_index)->setup_graph(cached_output_fields, range.start, range, fish, library_cache.at(thread_index));
                    else
                        fish_graphs.at(thread_index)->setup_graph(
                                                                  cached_output_fields, fish->start_frame(),
                              Range<Frame_t>{
                                  fish->start_frame(),
                                  fish->end_frame()
                              },
                              fish,
                              library_cache.at(thread_index));
                    
                    file::Path path = (filename + "_" + fish->identity().name() + "." + output_format.str());
                    file::Path final_path = fishdata / path;
                    
                    try {
                        if(output_format == default_config::output_format_t::npz) {
                            temporary_save(final_path, [&](file::Path use_path) {
                                fish_graphs.at(thread_index)->graph().save_npz(use_path.str(), &callback, true);
                                
                                std::vector<Frame_t::number_t> tracklets;
                                std::vector<float> vxy;
                                vxy.reserve(fish->frame_count() * 4);
                                
                                for(auto & tracklet : fish->tracklets()) {
                                    tracklets.push_back(tracklet->start().get());
                                    tracklets.push_back(tracklet->end().get());
                                    
                                    for(auto frame = tracklet->start() + 1_f;
                                        frame <= tracklet->end();
                                        frame += 1_f)
                                    {
                                        auto idx = tracklet->basic_stuff(frame);
                                        if(idx < 0)
                                            continue;
                                        
                                        auto centroid = fish->basic_stuff()[(size_t)idx]->centroid;
                                        
                                        auto v = centroid.v<Units::PX_AND_SECONDS>();
                                        auto speed = centroid.speed<Units::PX_AND_SECONDS>();
                                        vxy.push_back(frame.get());
                                        vxy.push_back(v.x);
                                        vxy.push_back(v.y);
                                        vxy.push_back(speed);
                                    }
                                }
                                cnpy::npz_save(use_path.str(), "tracklets", tracklets.data(), std::vector<size_t>{tracklets.size() / 2, 2}, "a");
                                cnpy::npz_save(use_path.str(), "tracklet_vxys", vxy.data(), std::vector<size_t>{vxy.size() / 4, 4}, "a");
                                cnpy::npz_save(use_path.str(), "cm_per_pixel", std::vector<double>{FAST_SETTING(cm_per_pixel)}, "a");
                                cnpy::npz_save(use_path.str(), "id", std::vector<size_t>{fish->identity().ID().get()}, "a");
                                cnpy::npz_save(use_path.str(), "frame_rate", std::vector<double>{(double)FAST_SETTING(frame_rate)}, "a");
                                cnpy::npz_save(use_path.str(), "detect_type", std::vector<uint32_t>{(uint32_t)SETTING(detect_type).value<track::detect::ObjectDetectionType_t>()}, "a");
                                cnpy::npz_save(use_path.str(), "detect_format", std::vector<uint32_t>{(uint32_t)SETTING(detect_format).value<track::detect::ObjectDetectionFormat_t>()}, "a");
                                auto video_size = SETTING(video_size).value<Size2>();
                                cnpy::npz_save(use_path.str(), "video_size", std::vector<double>{
                                    video_size.width,
                                    video_size.height
                                }, "a");
                            });
                            
                        } else
                            fish_graphs.at(thread_index)->graph().export_data(final_path.str(), &callback);
                        
                    } catch(const UtilsException&) {
                        FormatExcept("Failed to save data for individual ",fish->identity()," to location ",final_path,".");
                    }
                }
                
                const file::Path tags_path = FAST_SETTING(tags_path);
                
                for(auto &seg : fish->tracklets()) {
                    //for(auto frameIndex = seg->start(); frameIndex <= seg->end(); ++frameIndex) {
                    auto set = fish->has_tag_images_for(seg->end());
                    if(set && !set->empty()) {
                        std::vector<uchar> arrays;
                        std::vector<long_t> frame_indices;
                        std::vector<pv::bid> blob_ids;
                        
                        std::vector<uchar> image_data;
                        Size2 shape;
                        
                        Print("tags for ", fish->identity().ID(),":");
                        for(auto && [var, bid, ptr, frame] : *set) {
                            shape = Size2(ptr->cols, ptr->rows);
                            // had previous frame, lost in this frame (finalize segment)
                            assert(frame <= seg->end());
                            auto before = arrays.size();
                            arrays.resize(arrays.size() + ptr->size());
                            
                            Print(frame);
                            frame_indices.push_back(frame.get());
                            blob_ids.push_back(bid);
                            std::copy(ptr->data(), ptr->data() + ptr->size(), arrays.begin() + before);
                        }
                        Print();
                        
                        if(arrays.size() > 0) {
                            auto range = fish->get_tracklet(seg->end());
                            
                            if(!fish->has(range.start()))
                                throw U_EXCEPTION("Range starts at ",range.start(),", but frame is not set for fish ",fish->identity().ID(),".");
                            auto start_blob_id = fish->blob(range.start())->blob_id();
                            
                            file::Path path(tags_path / filename / ("frame"+range.start().toStr()+"_blob"+Meta::toStr(start_blob_id)+".npz"));
                            if(!path.remove_filename().exists()) {
                                if(!path.remove_filename().create_folder())
                                    throw U_EXCEPTION("Cannot create folder ",path.remove_filename().str()," please check permissions.");
                            }
                            
                            Print("Writing ", set->size()," images ",path.str());
                            cmn::npz_save(path.str(), "images", arrays.data(), {set->size(), (uint)shape.width, (uint)shape.height});
                            
                            //path = path.remove_filename() / ("fdx_"+path.filename().to_string());
                            cmn::npz_save(path.str(), "frames", frame_indices, "a");
                            cmn::npz_save(path.str(), "blob_ids", blob_ids, "a");
                        }
                    }
                    //}
                }
                
                /**
                 * Output representative images for each tracklet (that is long_t enough).
                 * These are currently median images and will all be saved into one big NPZ file.
                 * TODO: need to check for size issues (>=4GB?) - shouldnt happen too often though
                 */
                if(output_tracklet_images) {
                    Print("Generating tracklet images for fish ",fish->identity().raw_name(),"...");
                    const bool calculate_posture = FAST_SETTING(calculate_posture);
                    const auto individual_image_normalization = default_config::valid_individual_image_normalization();
                    
                    for(auto &range : fish->tracklets()) {
                        // only generate an image if the tracklet is long enough
                        if(range->length().get() >= output_min_frames) {
                            auto filters = constraints::local_midline_length(fish, range->range);
                            // Init data strutctures
                            //size_t image_count = 0;
                            
                            if(not do_normalize_tracklets
                               || not calculate_posture
                               || not is_in(individual_image_normalization, default_config::individual_image_normalization_t::posture)
                               || filters->median_midline_length_px > 0)
                            {
                                std::set<Frame_t> frames(range->iterable().begin(), range->iterable().end());
                                
                                if(tracklet_max_images != 0 && frames.size() > tracklet_max_images) {
                                    auto step_size = frames.size() / tracklet_max_images;
                                    std::set<Frame_t> tmp;
                                    for(auto it = frames.begin(); it != frames.end();) {
                                        tmp.insert(*it);
                                        // stride with step_size
                                        for(size_t j = 0; j < step_size && it != frames.end(); ++j) {
                                            // prefer images that are already set for other fish (fewer calls to preprocess)
                                            std::lock_guard<std::mutex> guard(sync);
                                            if(waiting_pixels.find(*(++it)) != waiting_pixels.end())
                                                break;
                                        }
                                    }
                                    
                                    frames = tmp;
                                }
                                
                                for(auto frame : frames) {
                                    auto midline = fish->midline(frame);
                                    if(not do_normalize_tracklets
                                       || not calculate_posture
                                       || not is_in(individual_image_normalization, default_config::individual_image_normalization_t::posture)
                                       || midline)
                                    {
                                        auto blob = fish->blob(frame);
                                        assert(blob);
                                        
                                        Transform trans;
                                        if(midline)
                                            trans = midline->transform(normalize);
                                        //pv::bid org_id;
                                        
                                        /*ImageData data(ImageData::Blob{
                                            blob->num_pixels(),
                                            pv::CompressedBlob{blob},
                                            org_id,
                                            blob->bounds()
                                        }, frame, *range, fish, fish->identity().ID(), trans);
                                        data.filters = std::make_shared<FilterCache>(*filters);
                                        assert(data.tracklet.contains(frame));*/
                                        
                                        std::lock_guard<std::mutex> guard(sync);
                                        waiting_pixels[frame][id] = ImageData{
                                            .blob = std::move(blob),
                                            .fdx = fish->identity().ID(),
                                            .midline_transform = trans,
                                            .median_midline_length_px = filters ? filters->median_midline_length_px : 0,
                                            .range = range->range
                                        };
                                        //++image_count;
                                    }
                                }
                            } // </median_midline_length>
                            
                        } // </output_min_frames>
                    }
                }
                
                if(BOOL_SETTING(output_visual_fields)) {
                    auto path = fishdata / (filename + "_visual_field_"+fish->identity().name());
                    fish->save_visual_field(path, range, progress_callback, true);
                }
                
                if(BOOL_SETTING(output_recognition_data)) {
                    // output network data
                    file::Path path = (filename + "_recognition_" + fish->identity().name() + ".npz");
                    
                    FrameRange fish_range(range);
                    if(range.empty())
                        fish_range = FrameRange(Range<Frame_t>(fish->start_frame(), fish->end_frame()));
                    
                    namespace py = Python;
                    std::vector<float> probabilities;
                    probabilities.reserve((size_t)fish_range.length().get() * py::VINetwork::number_classes());
                    
                    std::vector<long_t> recognition_frames;
                    
                    for(auto frame : fish_range.iterable()) {
                        auto blob = fish->blob(frame);
                        if(blob) {
                            auto pred = Tracker::instance()->find_prediction(frame, blob->blob_id());
                            if(pred) {
                                auto map = track::prediction2map(*pred);
                                for(auto && [rid, p] : map) {
                                    probabilities.push_back(p);
                                }
                                recognition_frames.push_back(frame.get());
                            }
                        }
                    }
                    
                    file::Path final_path = fishdata / path;
                    temporary_save(fishdata / path, [&](file::Path use_path) {
                        cmn::npz_save(use_path.str(), "frames", recognition_frames);
                        cmn::npz_save(use_path.str(), "probs", probabilities.data(), { recognition_frames.size(), py::VINetwork::number_classes() }, "a");
                    });
                }
                
                if(output_posture_data) {
                    file::Path path = (filename + "_posture_" + fish->identity().name() + ".npz");
                    
                    FrameRange fish_range(range);
                    if(range.empty())
                        fish_range = FrameRange(Range<Frame_t>(fish->start_frame(), fish->end_frame()));
                    
                    
                    std::vector<Vec2> midline_points, outline_points, midline_points_raw, hole_points;
                    std::vector<Vec2> offsets;
                    std::vector<float> midline_angles, midline_cms, areas, midline_offsets;
                    std::vector<size_t> hole_counts;
                    midline_points.reserve((size_t)fish_range.length().get() * 2 * FAST_SETTING(midline_resolution));
                    midline_points_raw.reserve(midline_points.capacity());
                    midline_angles.reserve((size_t)fish_range.length().get());
                    midline_offsets.reserve((size_t)fish_range.length().get());
                    areas.reserve((size_t)fish_range.length().get());
                    //outline_points.reserve(fish_range.length() * 2);
                    
                    size_t num_midline_points = 0, num_outline_points = 0;
                    
                    std::vector<long_t> posture_frames;
                    std::vector<size_t> midline_lengths, outline_lengths;
                    size_t first_midline_length = 0;
                    bool same_midline_length = true;
                    size_t counter = 0;
                    size_t print_step_size = size_t(fish_range.length().get()) / 100u;
                    if(print_step_size == 0)
                        print_step_size = 1;
                    
                    for(auto frame : fish_range.iterable()) {
                        auto outline = fish->outline(frame);
                        auto midline = do_normalize_output ? fish->fixed_midline(frame) : fish->midline(frame);
                        
                        if(outline && midline) {
                            posture_frames.push_back(frame.get());
                            
                            auto blob = fish->blob(frame);
                            offsets.push_back(blob->bounds().pos());
                            
                            /// also save the hole outlines of each blob (even though
                            /// these might be duplicates as soon as we start splitting blobs).
                            /// we will have the *hole_counts* array that contains...
                            ///  [ M, N_pts0, N_pts1, ..., N_ptsM, <next object> ]
                            if(blob->prediction().valid()
                               && blob->prediction().outlines.has_holes())
                            {
                                auto &lines = blob->prediction().outlines.lines;
                                hole_counts.emplace_back(lines.size());
                                
                                for(size_t i = 0; i < lines.size(); ++i) {
                                    auto pts = (std::vector<Vec2>)lines.at(i);
                                    hole_counts.emplace_back(pts.size());
                                    std::copy(pts.begin(), pts.end(), std::back_inserter(hole_points));
                                }
                                
                            } else {
                                hole_counts.emplace_back(0);
                            }
                            
                            midline_angles.push_back(midline->angle());
                            midline_offsets.push_back(atan2(midline->segments().back().pos - midline->segments().front().pos));
                            
                            // transform for transforming coordinates to real-world
                            Transform tf = midline->transform(default_config::individual_image_normalization_t::none, true);
                            
                            auto points = outline->uncompress();
                            outline_points.insert(outline_points.end(), points.begin(), points.end());
                            outline_lengths.push_back(points.size());
                            
                            for(auto & seg : midline->segments()) {
                                midline_points_raw.push_back(seg.pos);
                                midline_points.push_back(tf.transformPoint(seg.pos));
                            }
                            
                            num_outline_points += points.size();
                            
                            if(same_midline_length && first_midline_length && midline->segments().size() != first_midline_length)
                                same_midline_length = false;
                            if(!first_midline_length)
                                first_midline_length = midline->segments().size();
                            
                            num_midline_points += midline->segments().size();
                            
                            midline_lengths.push_back(midline->segments().size());
                            midline_cms.push_back(midline->len() * FAST_SETTING(cm_per_pixel));
                            areas.push_back(polygon_area(points));
                        }
                        
                        ++counter;
                        if(counter % print_step_size == 0) {
                            callback(float(counter) / float(fish_range.length().get()) + 1);
                        }
                    }
                    
                    temporary_save(fishdata / path, [&](file::Path use_path) {
                        cmn::npz_save(use_path.str(), "frames", posture_frames);
                        cmn::npz_save(use_path.str(), "offset", (const Float2_t*)offsets.data(), { posture_frames.size(), 2 }, "a");
                        cmn::npz_save(use_path.str(), "midline_lengths", midline_lengths, "a");
                        cmn::npz_save(use_path.str(), "midline_centimeters", midline_cms, "a");
                        cmn::npz_save(use_path.str(), "midline_offsets", midline_offsets, "a");
                        cmn::npz_save(use_path.str(), "midline_angle", midline_angles, "a");
                        cmn::npz_save(use_path.str(), "posture_area", areas, "a");
                        cmn::npz_save(use_path.str(), "hole_points", (const Float2_t*)hole_points.data(), std::vector<size_t>{ hole_points.size(), 2 }, "a");
                        cmn::npz_save(use_path.str(), "hole_counts", hole_counts, "a");
                        
                        if(same_midline_length) {
                            cmn::npz_save(use_path.str(), "midline_points", (const Float2_t*)midline_points.data(), std::vector<size_t>{ posture_frames.size(), first_midline_length, 2 }, "a");
                            cmn::npz_save(use_path.str(), "midline_points_raw", (const Float2_t*)midline_points_raw.data(), std::vector<size_t>{ posture_frames.size(), first_midline_length, 2 }, "a");
                        }
                        else {
                            cmn::npz_save(use_path.str(), "midline_points", (const Float2_t*)midline_points.data(), std::vector<size_t>{ num_midline_points, 2 }, "a");
                            cmn::npz_save(use_path.str(), "midline_points_raw", (const Float2_t*)midline_points_raw.data(), std::vector<size_t>{ num_midline_points, 2 }, "a");
                        }
                        
                        cmn::npz_save(use_path.str(), "outline_lengths", outline_lengths, "a");
                        cmn::npz_save(use_path.str(), "outline_points", (const Float2_t*)outline_points.data(), std::vector<size_t>{ num_outline_points, 2 }, "a");
                    });
                }
                
            } else {
                FormatWarning("Not exporting individual ",fish->identity().ID()," because it only has ",fish->frame_count(),"/",SETTING(output_min_frames).value<uint16_t>()," frames.");
            }
        };
        
        auto max_threads = hardware_concurrency();
        if(max_threads > 1) {
            for(size_t i=0; i<max_threads; ++i) {
                fish_graphs.push_back(std::make_shared<PropertiesGraph>());
                library_cache.push_back(std::make_shared<Output::LibraryCache>());
            }
            
            size_t current_thread_id = 0;
            std::vector<std::thread*> threads;
            threads.resize(max_threads);
            
            std::vector<std::queue<std::tuple<Idx_t, Individual*>>> packages;
            packages.resize(max_threads);
            
            IndividualManager::transform_all([&](auto id, auto fish) {
                packages.at(current_thread_id).push({ id, fish });
                
                ++current_thread_id;
                if(current_thread_id >= max_threads) {
                    current_thread_id = 0;
                }
            });
            
            std::mutex lock;
            for (size_t i=0; i<threads.size(); ++i) {
                threads.at(i) = new std::thread([&packages, &work_item, &library_cache](size_t index)
                {
                    cmn::set_thread_name("Export::export_data("+Meta::toStr(index)+")");
                    
                    while(!packages.at(index).empty()) {
                        auto [id, fish] = packages.at(index).front();
                        packages.at(index).pop();
                        
                        work_item(index, id, fish);
                        
                        /*{
                            mem::OutputLibraryMemoryStats stats(library_cache.at(index));
                            auto str = Meta::toStr(FileSize{stats.bytes});
                            std::lock_guard guard(lock);
                            Print("-- thread ", index," finished fish ", fish->identity().ID()," with ",str," of cache");
                        }*/
                        
                        auto it = library_cache.at(index)->_cache.find(fish);
                        if(it != library_cache.at(index)->_cache.end()) {
                            library_cache.at(index)->_cache.erase(it);
                        }
                    }
                    
                    library_cache.at(index)->clear();
                    library_cache.at(index) = nullptr;
                    
                }, i);
            }
            
            for(auto thread : threads) {
                thread->join();
                delete thread;
            }
            
        } else {
            IndividualManager::transform_all([&](auto fdx, const auto fish){
                work_item(0, fdx, fish);
            });
        }
        
        if(tracker.has_vi_predictions()) {
            std::vector<float> vi_probabilities;
            vi_probabilities.reserve(SQR(tracker.identities().size()) * tracker.analysis_range().length().get());
            Print("* assumed ", vi_probabilities.capacity());
            
            size_t items = 0;
            tracker.transform_vi_predictions([&vi_probabilities, &items](Frame_t frame, auto& table){
                for(auto &[bdx, probs] : table) {
                    vi_probabilities.insert(vi_probabilities.end(), {
                        (float)frame.get(), (float)(int64_t)bdx
                    });
                    vi_probabilities.insert(vi_probabilities.end(), probs.begin(), probs.end());
                    ++items;
                }
            });
            
            Print("* collected ", vi_probabilities.size(), " (", vi_probabilities.capacity(),").");
            
            file::Path path{fishdata / (filename+"_vi_probs.npz")};
            temporary_save(path, [&](file::Path use_path) {
                cmn::npz_save(use_path.str(), "probs", vi_probabilities.data(), {
                    items,
                    vi_probabilities.size() / items
                }, "w");
                Print("Saved vi predictions at ", use_path,".");
            });
        }
        
        if(BOOL_SETTING(output_heatmaps)) {
            heatmap::HeatmapController svenja;
            svenja.save();
        }
        
        if(BOOL_SETTING(output_statistics))
        {
            file::Path path = (filename + "_statistics.npz");
            
            if(!(fishdata / path).exists() || Tracker::instance()->statistics().size() == Tracker::number_frames())
            {
                std::vector<long_t> frame_numbers;
                std::vector<float> statistics;
                for(auto && [frame, stats] : Tracker::instance()->statistics()) {
                    frame_numbers.push_back(frame.get());
                    statistics.insert(statistics.end(), (float*)&stats, (float*)&stats + sizeof(track::Statistics) / sizeof(float));
                }
                
                assert(sizeof(track::Statistics) / sizeof(float) * frame_numbers.size() == statistics.size());
                
                temporary_save(fishdata / path, [&](file::Path use_path) {
                    cmn::npz_save(use_path.str(), "stats", statistics.data(), { frame_numbers.size(), sizeof(track::Statistics) / sizeof(float) }, "w");
                    cmn::npz_save(use_path.str(), "frames", frame_numbers, "a");
                    Print("Saved statistics at ", fishdata.str(),".");
                });
                
                if(!auto_no_memory_stats) {
                    temporary_save(fishdata / (filename + "_memory.npz"), [&](file::Path path) {
                        Print("Generating memory stats...");
                        mem::IndividualMemoryStats overall;
                        std::map<track::Idx_t, mem::IndividualMemoryStats> indstats;
                        IndividualManager::transform_all([&](auto fdx, auto fish) {
                            mem::IndividualMemoryStats stats(fish);
                            indstats[fdx] = stats;
                            overall += stats;
                        });
                        
                        overall.print();
                        
                        std::vector<Idx_t> ids;
                        std::map<std::string, std::vector<uint64_t>> sizes;
                        
                        ids.push_back(Idx_t());
                        for (auto && [key, size] : overall.sizes) {
                            sizes[key].push_back(size);
                        }
                        
                        for(auto && [fdx, stats] : indstats) {
                            ids.push_back(fdx);
                            for(auto && [key, size] : stats.sizes) {
                                sizes[key].push_back(size);
                            }
                        }
                        
                        cmn::npz_save(path.str(), "id", ids, "w");
                        
                        for(auto && [key, size] : sizes) {
                            cmn::npz_save(path.str(), key, size, "a");
                        }
                        
                        auto f = fishdata / (std::string)path.filename();
                        Print("Saved memory stats at ",f.str());
                    });
                    
                    temporary_save(fishdata / (filename + "_global_memory.npz"), [&](file::Path path) {
                        mem::OutputLibraryMemoryStats ol;
                        ol.print();
                        
                        mem::TrackerMemoryStats tl;
                        tl.print();
                        
                        bool written = false;
                        if(!ol.sizes.empty()) {
                            cmn::npz_save(path.str(), "output_cache", std::vector<uint64_t>{ol.sizes.at("output_cache")}, "w");
                            written = true;
                        }
                        
                        for(auto && [key, size] : tl.sizes) {
                            cmn::npz_save(path.str(), key, std::vector<uint64_t>{size}, written ? "a" : "w");
                            written = true;
                        }
                    });
                }
                
            } else {
                path = fishdata / path;
                FormatWarning("Not writing statistics because _statistics array (", Tracker::instance()->statistics().size(),") is != frames added (", Tracker::number_frames(),") and path ",path," exists.");
            }
        }
        
        // if there are representative tracklet images, save them...
        if(!waiting_pixels.empty()) {
            progress_callback(0, "saving tracklet images...");
            
            std::vector<uchar> all_images, single_images, split_masks;
            std::vector<uint64_t> frame_segment_indexes;
            std::vector<uint32_t> frame_segment_Nx2;
            std::vector<long_t> all_ranges, single_frames, single_ids, split_frames, split_ids;
            const bool tracklet_force_normal_color = SETTING(tracklet_force_normal_color).value<bool>();
            
            const auto encoding = Background::meta_encoding();
            const uint8_t exp_channels = required_storage_channels(encoding);
            
            std::map<Idx_t, std::map<Range<Frame_t>, std::queue<std::tuple<Vec2, Frame_t, Idx_t, Image::Ptr>>>> queues;
            PPFrame obj;
            
            const bool can_we_expect_fix_dimensions = do_normalize_tracklets;
            std::vector<uint32_t> image_dimensions;
            std::vector<uint32_t> image_coords;
            
            size_t index = 0;
            pv::Frame vframe;
            
            for(auto && [frame, vec] : waiting_pixels) {
                //if(SETTING(terminate))
                //    break;
                
                {
                    static Timing timing("[tracklet_images] preprocess", 20);
                    TakeTiming take(timing);
                    video.read_with_encoding(vframe, frame, encoding);
                    Tracker::preprocess_frame(std::move(vframe), obj, &_blob_thread_pool, PPFrame::NeedGrid::NoNeed, video.header().resolution);
                }
                
                for(auto && [id, data] : vec) {
                    struct ImagePosition {
                        Image::Ptr image;
                        Vec2 pos;
                        pv::BlobPtr blob;
                    } reduced, full;
                    
                    reduced.blob = obj.create_copy(data.blob->blob_id());
                    if(!reduced.blob) {
                        auto small = Tracker::find_blob_noisy(obj, data.blob->blob_id(), data.blob->parent_id(), Bounds());
                        if(small)
                            reduced.blob = std::move(small);
                        
                        if(data.blob->parent_id().valid()) {
                            if(!full.blob)
                                full.blob = obj.create_copy(data.blob->parent_id());
                        }
                    }
                    
                    /*if(full.blob == nullptr) {
                        FormatExcept("Cannot find ", data.blob->blob_id());
                        Print("reduced: ", reduced.blob ? reduced.blob->blob_id() : pv::bid()," full: ",full.blob ? full.blob->blob_id() : pv::bid());
                    }*/
                    
                    if(!reduced.blob && full.blob)
                        FormatExcept("Frame ", frame,", fish ", data.fdx,"");
                    if(!reduced.blob && !full.blob)
                        FormatExcept("Frame ", frame,", fish ", data.fdx," nothing found");
                    
                    if(!reduced.blob || !reduced.blob->pixels())
                        continue; // cannot find blob for given id
                    
                    if(do_normalize_tracklets) {
                        if(not tracklet_force_normal_color) {
                            auto &&[image, pos] =
                                constraints::diff_image(normalize,
                                                        reduced.blob.get(),//data.blob.get(),
                                                        data.midline_transform,
                                                        data.median_midline_length_px,
                                                        output_size,
                                                        Tracker::background());
                            reduced.image = std::move(image);
                            reduced.pos = pos;
                            
                        } else {
                            auto &&[image, pos] = calculate_normalized_image(data.midline_transform, reduced.blob.get(), data.median_midline_length_px, output_size, normalize == default_config::individual_image_normalization_t::legacy, Tracker::background());
                            reduced.image = std::move(image);
                            reduced.pos = pos;
                        }
                        
                    } else {
                        if(not tracklet_force_normal_color) {
                            auto && [pos, img] = reduced.blob->difference_image(*Tracker::background(), 0);
                            reduced.image = std::move(img);
                            reduced.pos = pos;
                        } else {
                            auto && [pos, img] = reduced.blob->color_image(Tracker::background());
                            reduced.image = std::move(img);
                            reduced.pos = pos;
                        }
                    }
                    
                    // normalize image size and write to normal queue
                    if(reduced.image) {
                        const Image* chosen = reduced.image.get();
                        assert(chosen != nullptr);
                        
                        Image::Ptr ptr;
                        if(chosen->channels() == 3
                           && Background::meta_encoding() == meta_encoding_t::r3g3b2)
                        {
                            ptr = Image::Make(chosen->rows, chosen->cols, 1);
                            cv::Mat output = ptr->get();
                            convert_to_r3g3b2<3>(chosen->get(), output);
                            chosen = ptr.get();
                        }
                        
                        if(can_we_expect_fix_dimensions) {
                            if(chosen->cols != output_size.width
                               || chosen->rows != output_size.height
                               || chosen->channels() != exp_channels)
                            {
                                throw InvalidArgumentException("Invalid dimensions for output_size ", output_size, "x",exp_channels,": ", chosen->dimensions(), "x", chosen->channels(), " of ", reduced.blob.get());
                            }
                        }
                        
                        queues[data.fdx][data.range].push({ reduced.pos, frame, data.fdx, ptr ? std::move(ptr) : std::move(reduced.image) });
                        
                        /*cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
                        
                        auto outline = data.fish->outline(frame);
                        if(outline) {
                            auto points = outline->uncompress();
                            for(auto pt : points) {
                                //pt += reduced.blob->bounds().pos();
                                pt += -offset;
                                cv::circle(image, pt, 1, gui::Green);
                            }
                        }
                        
                        cv::rectangle(image, -offset, -offset + reduced.blob->bounds().size(), gui::White);
                        cv::circle(image,  -offset + reduced.blob->bounds().size() * 0.5, 5, gui::Red);
                        tf::imshow("blob "+Meta::toStr(frame)+" "+Meta::toStr(reduced.blob->blob_id()), image);*/
                    }
                
                    // if present, also save the split mask
                    if(full.blob) {
                        auto trans = data.midline_transform;
                        trans.translate(full.blob->bounds().pos() - reduced.blob->bounds().pos());
                        
                        if(do_normalize_tracklets) {
                            if(not tracklet_force_normal_color)
                                full.image = std::get<0>(calculate_normalized_diff_image(trans, full.blob.get(), data.median_midline_length_px, output_size, normalize == default_config::individual_image_normalization_t::legacy, Tracker::background()));
                            else
                                full.image = std::get<0>(calculate_normalized_image(trans, full.blob.get(), data.median_midline_length_px, output_size, normalize == default_config::individual_image_normalization_t::legacy, Tracker::background()));
                            
                        } else {
                            if(not tracklet_force_normal_color) {
                                auto && [pos, img] = full.blob->difference_image(*Tracker::background(), 0);
                                full.image = std::move(img);
                                full.pos = pos;
                                
                            } else {
                                auto && [pos, img] = full.blob->color_image(Tracker::background());
                                full.image = std::move(img);
                                full.pos = pos;
                            }
                        }
                        
                        if(full.image) {
                            if(!do_normalize_tracklets) {
                                cv::Mat image;
                                full.image->get().copyTo(image);
                                if(image.channels() == 1) {
                                    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
                                } else {
                                    assert(image.channels() == 3);
                                }
#ifndef NDEBUG
                                Vec2 offset = full.blob->bounds().pos() - reduced.blob->bounds().pos();
                                
                                /*auto outline = data.fish->outline(frame);
                                if(outline) {
                                    auto points = outline->uncompress();
                                    for(auto pt : points) {
                                        //pt += reduced.blob->bounds().pos();
                                        pt += -offset;
                                        cv::circle(image, pt, 1, gui::Green);
                                    }
                                }*/
#endif
                                Vec2 center_offset = ((reduced.blob->bounds().pos() + reduced.blob->bounds().size() * 0.5) - (full.blob->bounds().pos() + full.blob->bounds().size() * 0.5));
#ifndef NDEBUG
                                Vec2 org = offset + center_offset;
                                cv::rectangle(image, -org, -org + reduced.blob->bounds().size(), gui::Yellow);
                                
                                cv::rectangle(image, -offset, -offset + reduced.blob->bounds().size(), gui::White);
                                cv::circle(image, -offset + reduced.blob->bounds().size() * 0.5, 5, gui::Red);
                                cv::line(image, -offset + reduced.blob->bounds().size() * 0.5, -offset - center_offset + reduced.blob->bounds().size() * 0.5, gui::Yellow);
#endif
                                cv::Mat empty_file = cv::Mat::zeros((int)output_size.height, (int)output_size.width, CV_8UC3);
                                Bounds input_bounds(-center_offset, Size2(image) + center_offset);
#ifndef NDEBUG
                                cv::putText(image, "input "+Meta::toStr(input_bounds), Vec2(20), cv::FONT_HERSHEY_PLAIN, 0.7, gui::White);
#endif
                                int left = 0, top = 0;
                                
                                //Bounds output_bounds(
                                input_bounds << Vec2(input_bounds.pos() + (output_size - Size2(image)) * 0.5);
                                input_bounds << Size2(image);
                                if(input_bounds.x < 0) {
                                    left = -(int)input_bounds.x;
                                    input_bounds.x = 0;
                                    input_bounds.width -= left;
                                }
                                if(input_bounds.y < 0) {
                                    top = -(int)input_bounds.y;
                                    input_bounds.height -= top;
                                    input_bounds.y = 0;
                                }
                                
#ifndef NDEBUG
                                Vec2 gp(20, 40);
                                cv::putText(image, "left "+Meta::toStr(left), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                cv::putText(image, "top "+Meta::toStr(top), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                //cv::putText(image, "right "+Meta::toStr(right), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                //cv::putText(image, "bottom "+Meta::toStr(bottom), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                
                                cv::putText(image, "restricted "+Meta::toStr(input_bounds), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::Yellow); gp += Vec2(0, 20);
                                cv::putText(image, "size "+Meta::toStr(Size2(image)), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                //input_bounds.restrict_to(Bounds(image));
#endif
                                input_bounds.restrict_to(Bounds(empty_file));
                                auto img_bounds = Bounds(left, top, input_bounds.width, input_bounds.height);
                                img_bounds.restrict_to(Bounds(image));
                                
                                image(img_bounds).copyTo(empty_file(input_bounds));
                                empty_file.copyTo(image);
                                
                                //Tracker::average()(Bounds())
                                
                                //tf::imshow("full "+Meta::toStr(frame)+" "+Meta::toStr(reduced.blob->blob_id())+" "+Meta::toStr(Size2(image)), image);
                                
                                if(full.image->channels() == 1) {
                                    /// if it was grey, keep it grey
                                    cv::Mat grey;
                                    cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
                                    full.image = Image::Make(grey);
                                } else {
                                    assert(full.image->channels() == image.channels());
                                    full.image = Image::Make(image);
                                }
                            } //else
                            {
                                assert(   full.image->cols == output_size.width
                                       && full.image->rows == output_size.height
                                       && full.image->channels() == exp_channels);
                            }
                            
                            split_ids.push_back(data.fdx.get());
                            split_frames.push_back(frame.get());
                            split_masks.insert(split_masks.end(), full.image->data(), full.image->data() + full.image->size());
                        }
                    }
                }
                
                ++index;
                
                auto step = size_t(waiting_pixels.size() * 0.1);
                if(!waiting_pixels.empty() && step > 0 && index % step == 0) {
                    Print("[tracklet_images] Frame ",index,"/",waiting_pixels.size(), " (", dec<2>(FAST_SETTING(track_max_individuals) == 0 ? float(vec.size()) : float(vec.size()) / float((float)FAST_SETTING(track_max_individuals) + 0.0001) * 100),"% identities / frame)");
                    progress_callback(index / float(waiting_pixels.size()), "");
                }
            }
            
            // npz file sizes are limited to ~2GB, so we need to split up the files if necessary
            size_t part_counter = 0;
            size_t byte_counter = 0;
            
            auto export_singles = [&](file::Path path) {
                if(!single_frames.empty()) {
                    path = file::Path(path.str() + "_part"+Meta::toStr(part_counter)+".npz");
                    
                    size_t samples = single_frames.size();
                    Print("Saving single tracklet images to ", path,"... (",samples," images)");
                    
                    if(path.exists())
                        path.delete_file();
                    
                    temporary_save(path, [&](file::Path path){
                        if(can_we_expect_fix_dimensions) {
                            cmn::npz_save(path.str(), "images", single_images.data(), {
                                single_frames.size(),
                                (size_t)output_size.height,
                                (size_t)output_size.width,
                                (size_t)exp_channels
                            }, "w");
                            
                        } else {
                            cmn::npz_save(path.str(), "images", single_images.data(), {
                                single_images.size()
                            }, "w");
                        }
                        
                        if(not image_dimensions.empty()) {
                            assert(image_dimensions.size() % 3 == 0);
                            cmn::npz_save(path.str(), "dimensions", image_dimensions.data(), {
                                image_dimensions.size() / 3,
                                3u
                            }, "a");
                        }
                        
                        cmn::npz_save(path.str(), "positions", image_coords.data(), {
                            single_frames.size(),
                            2u
                        }, "a");
                        
                        cmn::npz_save(path.str(), "frame_segment_indexes", frame_segment_indexes, "a");
                        cmn::npz_save(path.str(), "tracklets", frame_segment_Nx2.data(), {
                            frame_segment_Nx2.size() / 2u,
                            2u
                        }, "a");
                        
                        cmn::npz_save(path.str(), "frames", single_frames, "a");
                        cmn::npz_save(path.str(), "ids", single_ids, "a");
                        
                        // export the current meta_encoding as a string to the npz file:
                        const std::string meta_encoding = Meta::toStr(Background::meta_encoding());
                        cmn::npz_save(path.str(), "encoding", meta_encoding, "a");
                    });
                    
                    single_frames.clear();
                    single_ids.clear();
                    single_images.clear();
                    frame_segment_indexes.clear();
                    frame_segment_Nx2.clear();
                    image_coords.clear();
                    image_dimensions.clear();
                    
                    ++part_counter;
                    byte_counter = 0;
                    
                } else {
                    FormatWarning("Called export_singles, but single_frames is empty.");
                }
            };
            
            // output network data
            file::Path path = fishdata / (filename + "_tracklet_images.npz");
            file::Path single_path = fishdata / (filename + "_tracklet_images_single");
            
            if(!split_masks.empty()) {
                auto path = single_path.str() + "_splits_part";
                Print("Saving split tracklet masks to ", path,"... (",split_frames.size()," images)");
                
                int64_t bytes_per_image = (int64_t)output_size.height * (int64_t)output_size.width;
                int64_t n_images = int64_t(1.5 *1000 * 1000 * 1000) / bytes_per_image;
                
                Print(n_images,"/",split_frames.size()," images fit in 1.5GB");
                
                int64_t offset = 0;
                size_t part = 0;
                int64_t N = narrow_cast<int64_t>(split_frames.size());
                
                while (offset < N) {
                    auto L = min(n_images, N - offset);
                    
                    auto sub_path = path + Meta::toStr(part) + ".npz";
                    ++part;
                    
                    Print("Saving to '",sub_path.c_str(),"' from ",offset,"-",offset+L," (",split_frames.size(),")");
                    
                    temporary_save(sub_path, [&](file::Path path) {
                        cmn::npz_save(path.str(), "images",
                                      split_masks.data() + offset * (int64_t)output_size.height * (int64_t)output_size.width * (int64_t)exp_channels,
                                      { sign_cast<size_t>(L), (size_t)output_size.height, (size_t)output_size.width, (size_t)exp_channels }, "w");
                        cmn::npz_save(path.str(), "frames", std::vector<long_t>(split_frames.begin() + offset, split_frames.begin() + offset + L), "a");
                        cmn::npz_save(path.str(), "ids", std::vector<long_t>(split_ids.begin() + offset, split_ids.begin() + offset + L), "a");
                    });
                    
                    offset += n_images;
                }
            }
            
            size_t range_index = 0;
            
            for(auto && [id, ranges] : queues) {
                cv::Mat tmp;
                std::vector<std::vector<hist_utils::Hist>> M; // histograms
                cv::Mat1b med;              // median image
                
                for(auto && [range, images] : ranges) {
                    size_t image_count = 0;
                    hist_utils::init(M, med, (int)output_size.height, (int)output_size.width);
                    
                    while(not images.empty()) {
                        auto [pos, frame, fid, image] = std::move(images.front());
                        images.pop();
                        
                        image_coords.push_back(saturate(pos.x, 0_F, (Float2_t)std::numeric_limits<uint32_t>::max()));
                        image_coords.push_back(saturate(pos.y, 0_F, (Float2_t)std::numeric_limits<uint32_t>::max()));
                        
                        image_dimensions.push_back(image->rows);
                        image_dimensions.push_back(image->cols);
                        image_dimensions.push_back(image->channels());
                        
                        single_frames.push_back(frame.get());
                        single_ids.push_back(fid.get());
                        
                        if(range_index != frame_segment_Nx2.size() / 2)
                            FormatWarning("range_index(",range_index,") is not ", frame_segment_Nx2.size() / 2);
                        frame_segment_indexes.push_back(range_index);
                        
                        if(not can_we_expect_fix_dimensions) {
                            single_images.insert(single_images.end(), image->data(), image->data() + image->size());
                            byte_counter += image->size();
                            
                        } else {
                            
                            auto mat = image->get();
                            cv::Mat1b tmp;
                            if(mat.channels() == 1) {
                                assert(mat.type() == CV_8UC1);
                                mat.copyTo(tmp);
                            } else if(mat.channels() == 3) {
                                cv::cvtColor(mat, tmp, cv::COLOR_BGR2GRAY);
                            } else
                                throw InvalidArgumentException("Invalid number of channels: ", *image);
                            
                            assert(image->cols == output_size.width
                                   && image->rows == output_size.height
                                   && image->channels() == exp_channels);
                            //mat.convertTo(tmp, CV_8UC(image->dims));
                            hist_utils::addImage(tmp, M, med);
                            
                            //if(tracklet_max_images == 0)
                            {
                                single_images.insert(single_images.end(), image->data(), image->data() + image->size());
                                byte_counter += image->size();
                            }
                        }
                        ++image_count;
                        
                        if(byte_counter >= 1.5 * 1000 * 1000 * 1000) {
                            // finish up this range
                            export_singles(single_path);
                        }
                    }
                    
                    frame_segment_Nx2.push_back(range.start.get());
                    frame_segment_Nx2.push_back(range.end.get());
                    
                    ++range_index;
                    
                    if(image_count > 1) {
                        med.copyTo(tmp);
                        assert(tmp.isContinuous() && tmp.channels() == 1);
                        all_images.insert(all_images.end(), tmp.data, tmp.data + tmp.cols * tmp.rows * tmp.channels());
                        all_ranges.push_back(id.get());
                        all_ranges.push_back(range.start.get());
                        all_ranges.push_back(range.end.get());
                        //tf::imshow("median"+fish->identity().name()+" - "+Meta::toStr(range.range), tmp);
                    }
                }
                
                if(id.get() % max(1, int(ceil(queues.size() * 0.01))) == 0)
                    Print("[tracklet_images] Fish ", id,"...");
            }
            
            size_t samples = all_images.size() / (size_t)output_size.height / (size_t)output_size.width;
            Print("Saving tracklet images to ", path,"... (",samples," samples)");
            
            temporary_save(path, [&](file::Path path){
                cmn::npz_save(path.str(), "images", all_images.data(), { samples, (size_t)output_size.height, (size_t)output_size.width }, "w");
                cmn::npz_save(path.str(), "meta", all_ranges.data(), { samples, 3 /* ID, frame_start, frame_end */ }, "a");
            });
            
            export_singles(single_path);
        }
        
    } catch(const UtilsException&) {}
    
    // reset values to previous setting
    SETTING(output_default_options) = previous_options;
    SETTING(output_fields) = previous_graphs;
    
    SETTING(output_frame_window) = previous_output_frame_window;
    
}
}
