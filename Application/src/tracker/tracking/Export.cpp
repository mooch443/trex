#include "Export.h"
#include <types.h>

#include <tracking/Tracker.h>
#include <tracker/misc/OutputLibrary.h>
#include <tracking/Recognition.h>
#include <tracker/gui/WorkProgress.h>
#include <tracker/gui/gui.h>
#include <tracker/gui/DrawGraph.h>
#include <misc/cnpy_wrapper.h>
#include <tracker/misc/MemoryStats.h>
#include <pv.h>
#include <misc/checked_casts.h>
#include <gui/IdentityHeatmap.h>

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
        tmp_path = "/tmp";
#endif

        if (tmp_path.exists()) {
            if (access(tmp_path.c_str(), W_OK) == 0)
                use_path = tmp_path / path.filename();
        }

        try {
            fn(use_path);

            static std::mutex mutex;
            std::lock_guard guard(mutex);
            if (final_path != use_path) {
                //Debug("Moving '%S' to '%S'...", &use_path.str(), &final_path.str());
                if (!use_path.move_to(final_path)) {
                    U_EXCEPTION("Cannot move file '%S' to '%S'.", &use_path.str(), &final_path.str());
                } //else
                  //  Debug("  Moved '%S' to '%S'.", &use_path.str(), &final_path.str());
            }

        }
        catch (const std::exception& ex) {
            Except("Problem copying file '%S' to '%S': %s", &use_path.str(), &final_path.str(), ex.what());
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

// (X[i], Y[i]) are coordinates of i'th point.
Float2_t polygonArea(const std::vector<Vec2>& pts)
{
    // Initialze area
    Float2_t area = 0.0;
    if(pts.empty())
        return 0;
    
    // Calculate value of shoelace formula
    auto n = pts.size();
    size_t j = n - 1;
    for (size_t i = 0; i < n; i++)
    {
        area += (pts[j].x + pts[i].x) * (pts[j].y - pts[i].y);
        j = i;  // j is previous vertex to i
    }
    
    // Return absolute value
    return cmn::abs(area / 2.0f);
}

void export_data(Tracker& tracker, long_t fdx, const Range<Frame_t>& range) {
    using namespace gui;
    GenericThreadPool _blob_thread_pool(cmn::hardware_concurrency());
    
    Tracker::LockGuard guard("GUI::export_tracks");
    tracker.wait();
    
    // save old values and remove all calculation/scaling options from output
    auto previous_graphs = SETTING(output_graphs).value<Output::Library::graphs_type>();
    auto previous_options = SETTING(output_default_options).value<Output::Library::default_options_type>();
    
    Output::Library::remove_calculation_options();
    
    auto previous_output_frame_window = SETTING(output_frame_window).value<long_t>();
    auto output_image_per_tracklet = GUI::instance() ? SETTING(output_image_per_tracklet).value<bool>() : false;
    auto recognition_enable = FAST_SETTINGS(recognition_enable);
    auto output_format = SETTING(output_format).value<default_config::output_format_t::Class>();
    auto output_posture_data = SETTING(output_posture_data).value<bool>();
    auto output_min_frames = SETTING(output_min_frames).value<uint16_t>();
    auto no_tracking_data = SETTING(auto_no_tracking_data).value<bool>();
    auto auto_no_memory_stats = SETTING(auto_no_memory_stats).value<bool>();
    
    auto normalize = SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>();
    if(!FAST_SETTINGS(calculate_posture) && normalize == default_config::recognition_normalization_t::posture)
        normalize = default_config::recognition_normalization_t::moments;
    
    if(no_tracking_data) {
        Warning("Not saving tracking data because of 'auto_no_tracking_data' flag being set.");
    }
    //auto calculate_posture = SETTING(calculate_posture).value<bool>();
    
    const Size2 output_size = SETTING(recognition_image_size);
    const bool do_normalize_tracklets = SETTING(tracklet_normalize_orientation);
    const bool do_normalize_output = SETTING(output_normalize_midline_data);
    const uint16_t tracklet_max_images = SETTING(tracklet_max_images);
    
    auto fishdata_dir = SETTING(fishdata_dir).value<file::Path>();
    auto fishdata = pv::DataLocation::parse("output", fishdata_dir);
    if(!fishdata.exists())
        if(!fishdata.create_folder())
            U_EXCEPTION("Cannot create folder '%S' for saving fishdata.", &fishdata.str());
    
    auto filename = std::string(SETTING(filename).value<file::Path>().filename());
    auto posture_path = (fishdata / (filename + "_posture_*.npz")).str();
    auto recognition_path = (fishdata / (filename + "_recognition_*.npz")).str();
    
    if(!range.empty())
        Debug("[exporting] Exporting range [%d-%d]", range.start, range.end);
    else
        Debug("[exporting] Exporting all frames (%d)", tracker.number_frames());
    auto individual_prefix = FAST_SETTINGS(individual_prefix);
    Debug("[exporting] Writing data from `output_graphs` to '%S/%S_%S*.%s'", &fishdata.str(), &filename, &individual_prefix, output_format.name());
    if(output_posture_data)
        Debug("[exporting] Writing posture data to '%S'", &posture_path);
    if(recognition_enable)
        Debug("[exporting] Writing recognition data to '%S'", &recognition_path);
    
    try {
        std::map<long_t, float> all_percents;
        std::mutex percent_mutex;
        for(auto && [id, fish] : Tracker::individuals())
            all_percents[id] = 0;
        
        using ImageData = Recognition::ImageData;
        std::map<Frame_t, std::map<long_t, ImageData>> waiting_pixels;
        std::mutex sync;
        
        std::vector<std::shared_ptr<PropertiesGraph>> fish_graphs;
        std::vector<Output::LibraryCache::Ptr> library_cache;
        float last_percent = -1;
        
        auto work_item = [&](size_t thread_index, long_t id, Individual* fish){
            if(fdx != -1 && fdx != id)
                return;
            
            if(SETTING(terminate))
                return;
            
            std::function<void(float)> callback = [id, &percent_mutex, &all_percents, &last_percent, &fishdata, output_posture_data](float percent) {
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
                    
                    if(GUI::instance()) {
                        GUI::instance()->work().set_percent(overall_percent / all_percents.size() * (output_posture_data ? 0.5f : 1.0f));
                        overall_percent = GUI::instance()->work().percent();
                    }
                }
                
                {
                    // synchronize with debug messages
                    std::lock_guard<std::mutex> lock(DEBUG::debug_mutex());
                    if(cmn::abs(last_percent - overall_percent) >= 0.05) {
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
                }
            };
            
            if (fish->frame_count() >= output_min_frames) {
                if(!no_tracking_data) {
                    if(!range.empty())
                        fish_graphs.at(thread_index)->setup_graph(range.start.get(), Rangel(range.start.get(), range.end.get()), fish, library_cache.at(thread_index));
                    else
                        fish_graphs.at(thread_index)->setup_graph(
                              fish->start_frame().get(),
                              Rangel(fish->start_frame().get(), fish->end_frame().get()),
                              fish, library_cache.at(thread_index));
                    
                    file::Path path = ((std::string)SETTING(filename).value<file::Path>().filename() + "_" + fish->identity().name() + "." + output_format.name());
                    file::Path final_path = fishdata / path;
                    
                    try {
                        if(output_format == default_config::output_format_t::npz) {
                            temporary_save(final_path, [&](file::Path use_path) {
                                fish_graphs.at(thread_index)->graph().save_npz(use_path.str(), &callback, true);
                                
                                std::vector<Frame_t> segment_borders;
                                std::vector<float> vxy;
                                vxy.reserve(fish->frame_count() * 2);
                                
                                for(auto & segment : fish->frame_segments()) {
                                    segment_borders.push_back(segment->start());
                                    segment_borders.push_back(segment->end());
                                    
                                    for(auto frame = segment->start() + 1_f; frame <= segment->end(); frame += 1_f)
                                    {
                                        auto idx = segment->basic_stuff(frame);
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
                                cnpy::npz_save(use_path.str(), "frame_segments", segment_borders.data(), std::vector<size_t>{segment_borders.size() / 2, 2}, "a");
                                cnpy::npz_save(use_path.str(), "segment_vxys", vxy.data(), std::vector<size_t>{vxy.size() / 4, 4}, "a");
                            });
                            
                        } else
                            fish_graphs.at(thread_index)->graph().export_data(final_path.str(), &callback);
                        
                    } catch(const UtilsException&) {
                        Except("Failed to save data for fish '%S' to location '%S'.", &fish->identity().raw_name(), &final_path.str());
                    }
                }
                
                const file::Path tags_path = FAST_SETTINGS(tags_path);
                
                for(auto &seg : fish->frame_segments()) {
                    //for(auto frameIndex = seg->start(); frameIndex <= seg->end(); ++frameIndex) {
                        auto set = fish->has_tag_images_for(seg->end());
                        if(set && !set->empty()) {
                            std::vector<uchar> arrays;
                            std::vector<long_t> frame_indices;
                            std::vector<pv::bid> blob_ids;
                            
                            std::vector<uchar> image_data;
                            Size2 shape;
                            
                            printf("tags for %u: ", (uint32_t)fish->identity().ID());
                            for(auto && [var, bid, ptr, frame] : *set) {
                                shape = Size2(ptr->cols, ptr->rows);
                                // had previous frame, lost in this frame (finalize segment)
                                assert(frame <= seg->end());
                                auto before = arrays.size();
                                arrays.resize(arrays.size() + ptr->size());
                                
                                printf("%d ", frame.get());
                                frame_indices.push_back(frame.get());
                                blob_ids.push_back(bid);
                                std::copy(ptr->data(), ptr->data() + ptr->size(), arrays.begin() + before);
                            }
                            printf("\n");
                            
                            if(arrays.size() > 0) {
                                auto range = fish->get_segment(seg->end());
                                
                                if(!fish->has(range.start()))
                                    U_EXCEPTION("Range starts at %d, but frame is not set for fish %d.", range.start(), fish->identity().ID());
                                auto start_blob_id = fish->blob(range.start())->blob_id();
                                
                                file::Path path(tags_path / SETTING(filename).value<file::Path>().filename() / ("frame"+range.start().toStr()+"_blob"+Meta::toStr(start_blob_id)+".npz"));
                                if(!path.remove_filename().exists()) {
                                    if(!path.remove_filename().create_folder())
                                        U_EXCEPTION("Cannot create folder '%S' please check permissions.", &path.remove_filename().str());
                                }
                                
                                Debug("Writing %d images '%S'", set->size(), &path.str());
                                cmn::npz_save(path.str(), "images", arrays.data(), {set->size(), (uint)shape.width, (uint)shape.height});
                                
                                //path = path.remove_filename() / ("fdx_"+path.filename().to_string());
                                cmn::npz_save(path.str(), "frames", frame_indices, "a");
                                cmn::npz_save(path.str(), "blob_ids", blob_ids, "a");
                            }
                        }
                    //}
                }
                
                /**
                 * Output representative images for each segment (that is long_t enough).
                 * These are currently median images and will all be saved into one big NPZ file.
                 * TODO: need to check for size issues (>=4GB?) - shouldnt happen too often though
                 */
                if(output_image_per_tracklet) {
                    Debug("Generating tracklet images for fish '%S'...", &fish->identity().raw_name());
                    
                    for(auto &range : fish->frame_segments()) {
                        // only generate an image if the segment is long_t enough
                        if(range->length() >= (long_t)output_min_frames) {
                            auto filters = std::make_shared<TrainingFilterConstraints>(Tracker::recognition()->local_midline_length(fish, range->range));
                            // Init data strutctures
                            size_t image_count = 0;
                            
                            if(filters->median_midline_length_px > 0) {
                                std::set<Frame_t> frames(range->range.iterable().begin(), range->range.iterable().end());
                                
                                if(tracklet_max_images != 0 && frames.size() > tracklet_max_images) {
                                    
                                //}
                                //if(frames.size() > 100 /** magic number of frames **/) {
                                    auto step_size = frames.size() / tracklet_max_images;
                                    std::set<Frame_t> tmp;
                                    for(auto it = frames.begin(); it != frames.end();) {
                                        //Debug("%d-%d adding %d (%d)", range.start(), range.end(), *it, step_size);
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
                                    if(midline) {
                                        auto blob = fish->blob(frame);
                                        assert(blob);
                                        
                                        auto trans = midline->transform(normalize);
                                        pv::bid org_id;
                                        
                                        if(SETTING(tracklet_restore_split_blobs) && blob->parent_id().valid()) {
                                            pv::Frame pvf;
                                            GUI::instance()->video_source()->read_frame(pvf, (uint64_t)frame.get());
                                            auto bs = pvf.get_blobs();
                                            //Debug("Blob %d in frame %d has been split (%d)", blob->blob_id(), frame, blob->parent_id());
                                            
                                            for(auto &b : bs) {
                                                if(b->blob_id() == blob->parent_id()) {
                                                    //Debug("Replacing blob %d with parent blob %d", blob->blob_id(), b->blob_id());
                                                    b->calculate_moments();
                                                    trans.translate(-(blob->bounds().pos() - b->bounds().pos()));
                                                    org_id = blob->blob_id();
                                                    blob = b;
                                                    break;
                                                }
                                            }
                                        }
                                        
                                        ImageData data(ImageData::Blob{
                                            blob->num_pixels(), 
                                            blob->blob_id(), 
                                            org_id, 
                                            blob->parent_id(), 
                                            blob->bounds()
                                        }, frame, *range, fish, fish->identity().ID(), trans);
                                        data.filters = filters;
                                        assert(data.segment.contains(frame));
                                        
                                        std::lock_guard<std::mutex> guard(sync);
                                        waiting_pixels[frame][id] = data;
                                        ++image_count;
                                    }
                                }
                            } // </median_midline_length>
                            
                        } // </output_min_frames>
                    }
                }
                
                if(recognition_enable && SETTING(output_recognition_data)) {
                    // output network data
                    file::Path path = ((std::string)SETTING(filename).value<file::Path>().filename() + "_recognition_" + fish->identity().name() + ".npz");
                    
                    Range<Frame_t> fish_range(range);
                    if(range.empty())
                        fish_range = Range<Frame_t>(fish->start_frame(), fish->end_frame());
                    
                    std::vector<float> probabilities;
                    probabilities.reserve((size_t)fish_range.length().get() * Recognition::number_classes());
                    
                    std::vector<long_t> recognition_frames;
                    
                    for(auto frame : fish_range.iterable()) {
                        auto blob = fish->blob(frame);
                        if(blob) {
                            auto map = Tracker::recognition()->ps_raw(frame, blob->blob_id());
                            if(!map.empty()) {
                                for(auto && [rid, p] : map) {
                                    probabilities.push_back(p);
                                }
                                
                                recognition_frames.push_back(frame.get());
                            }
                        }
                    }
                    
                    //Debug("Saving recognition data to '%S'", &path.str());
                    file::Path final_path = fishdata / path;
                    temporary_save(fishdata / path, [&](file::Path use_path) {
                        cmn::npz_save(use_path.str(), "frames", recognition_frames);
                        cmn::npz_save(use_path.str(), "probs", probabilities.data(), { recognition_frames.size(), Recognition::number_classes() }, "a");
                    });
                }
                
                if(output_posture_data) {
                    file::Path path = ((std::string)SETTING(filename).value<file::Path>().filename() + "_posture_" + fish->identity().name() + ".npz");
                    
                    Range<Frame_t> fish_range(range);
                    if(range.empty())
                        fish_range = Range<Frame_t>(fish->start_frame(), fish->end_frame());
                    
                    //Debug("Writing posture data to '%S' [%d-%d]...", &path.str(), fish_range.start, fish_range.end);
                    
                    std::vector<Vec2> midline_points, outline_points, midline_points_raw;
                    std::vector<Vec2> offsets;
                    std::vector<float> midline_angles, midline_cms, areas, midline_offsets;
                    midline_points.reserve((size_t)fish_range.length().get() * 2 * FAST_SETTINGS(midline_resolution));
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
                            
                            midline_angles.push_back(midline->angle());
                            midline_offsets.push_back(atan2(midline->segments().back().pos - midline->segments().front().pos));
                            
                            // transform for transforming coordinates to real-world
                            Transform tf = midline->transform(default_config::recognition_normalization_t::none, true);
                            
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
                            midline_cms.push_back(midline->len() * FAST_SETTINGS(cm_per_pixel));
                            areas.push_back(polygonArea(points));
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
                Warning("Not exporting individual %d because it only has %d/%d frames.", fish->identity().ID(), fish->frame_count(), SETTING(output_min_frames).value<uint16_t>());
            }
        };
        
        auto max_threads = hardware_concurrency();
        if(max_threads > 1) {
            for(size_t i=0; i<max_threads; ++i) {
                fish_graphs.push_back(std::make_shared<PropertiesGraph>(tracker, Vec2()));
                library_cache.push_back(std::make_shared<Output::LibraryCache>());
            }
            
            size_t current_thread_id = 0;
            std::vector<std::thread*> threads;
            threads.resize(max_threads);
            
            std::vector<std::queue<std::tuple<long_t, Individual*>>> packages;
            packages.resize(max_threads);
            
            for (auto&& [id, fish] : tracker.individuals()) {
                packages.at(current_thread_id).push({ id, fish });
                
                ++current_thread_id;
                if(current_thread_id >= max_threads) {
                    current_thread_id = 0;
                }
            }
            
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
                            Debug("-- thread %d finished fish %d with %S of cache", index, fish->identity().ID(), &str);
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
            for (auto&& [id, fish] : tracker.individuals())
                work_item(0, id, fish);
        }
        
        if(SETTING(output_heatmaps)) {
            heatmap::HeatmapController svenja;
            svenja.save();
        }
        
        if(SETTING(output_statistics))
        {
            file::Path path = ((std::string)SETTING(filename).value<file::Path>().filename() + "_statistics.npz");
            
            if(!(fishdata / path).exists() || Tracker::instance()->_statistics.size() == Tracker::number_frames())
            {
                std::vector<long_t> frame_numbers;
                std::vector<float> statistics;
                for(auto && [frame, stats] : Tracker::instance()->_statistics) {
                    frame_numbers.push_back(frame.get());
                    statistics.insert(statistics.end(), (float*)&stats, (float*)&stats + sizeof(track::Tracker::Statistics) / sizeof(float));
                }
                
                assert(sizeof(track::Tracker::Statistics) / sizeof(float) * frame_numbers.size() == statistics.size());
                
                temporary_save(fishdata / path, [&](file::Path use_path) {
                    cmn::npz_save(use_path.str(), "stats", statistics.data(), { frame_numbers.size(), sizeof(track::Tracker::Statistics) / sizeof(float) }, "w");
                    cmn::npz_save(use_path.str(), "frames", frame_numbers, "a");
                    Debug("Saved statistics at '%S'.", &fishdata.str());
                });
                
                if(!auto_no_memory_stats) {
                    temporary_save(fishdata / ((std::string)SETTING(filename).value<file::Path>().filename() + "_memory.npz"), [&](file::Path path) {
                        Debug("Generating memory stats...");
                        mem::IndividualMemoryStats overall;
                        std::map<track::Idx_t, mem::IndividualMemoryStats> indstats;
                        for(auto && [fdx, fish] : tracker.individuals()) {
                            mem::IndividualMemoryStats stats(fish);
                            indstats[fdx] = stats;
                            overall += stats;
                        }
                        
                        overall.print();
                        
                        std::vector<long_t> ids;
                        std::map<std::string, std::vector<uint64_t>> sizes;
                        
                        ids.push_back(-1);
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
                        Debug("Saved memory stats at '%S'", &f.str());
                    });
                    
                    temporary_save(fishdata / ((std::string)SETTING(filename).value<file::Path>().filename() + "_global_memory.npz"), [&](file::Path path) {
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
                Warning("Not writing statistics because _statistics array (%d) is != frames added (%d) and path '%S' exists.", Tracker::instance()->_statistics.size(), Tracker::number_frames(), &path.str());
            }
        }
        
        // if there are representative tracklet images, save them...
        if(!waiting_pixels.empty()) {
            if(GUI::instance())
                GUI::instance()->work().set_item("saving tracklet images...");
            
            std::vector<uchar> all_images, single_images, split_masks;
            std::vector<long_t> all_ranges, single_frames, single_ids, split_frames, split_ids;
            
            std::map<long_t, std::map<Range<Frame_t>, std::queue<std::tuple<Frame_t, long_t, Image::UPtr>>>> queues;
            PPFrame obj;
            
            size_t index = 0;
            for(auto && [frame, vec] : waiting_pixels) {
                if(SETTING(terminate))
                    break;
                
                {
                    static Timing timing("[tracklet_images] preprocess", 20);
                    TakeTiming take(timing);
                    auto active = frame == Tracker::start_frame() ? Tracker::set_of_individuals_t() : Tracker::active_individuals(frame - 1_f);
                    GUI::instance()->video_source()->read_frame(obj.frame(), sign_cast<uint64_t>(frame.get()));
                    Tracker::instance()->preprocess_frame(obj, active, &_blob_thread_pool);
                }
                
                for(auto && [id, data] : vec) {
                    struct ImagePosition {
                        Image::UPtr image;
                        Vec2 pos;
                        pv::BlobPtr blob;
                    } reduced, full;
                    
                    reduced.blob = Tracker::find_blob_noisy(obj, data.blob.blob_id, data.blob.parent_id, Bounds());
                    if(data.blob.org_id.valid()) {
                        full.blob = Tracker::find_blob_noisy(obj, data.blob.org_id, data.blob.parent_id, Bounds());
                    }
                    
                    if(data.blob.org_id.valid() && !full.blob)//if(!reduced.blob)
                    {
                        for(auto b : obj.frame().blobs()) {
                            if(b->blob_id() == data.blob.blob_id) {
                                //Debug("Frame %d: Found blob %d in original array, not in the parsed one.", frame, data.blob.blob_id);
                                full.blob = b;
                                break;
                            }
                        }
                    }
                    
                    if(data.blob.org_id.valid() && full.blob == nullptr) {
                        Except("Cannot find %d and %ld", data.blob.blob_id, data.blob.org_id);
                        Debug("reduced: %d full: %d", reduced.blob ? reduced.blob->blob_id() : 0, full.blob ? full.blob->blob_id() : 0);
                    }
                    
                    if(!reduced.blob && full.blob)
                        Except("Frame %d, fish %d", frame, data.fish->identity().ID());
                    if(!reduced.blob && !full.blob)
                        Except("Frame %d, fish %d nothing found", frame, data.fish->identity().ID());
                    
                    if(!reduced.blob)
                        continue; // cannot find blob for given id
                    
                    if(do_normalize_tracklets)
                        reduced.image = std::move(std::get<0>(data.fish->calculate_normalized_diff_image(data.midline_transform, reduced.blob, data.filters->median_midline_length_px, output_size, normalize == default_config::recognition_normalization_t::legacy)));
                    else {
                        //auto && [img, pos] = data.fish->calculate_diff_image(blob, output_size);
                        auto && [pos, img] = reduced.blob->difference_image(*Tracker::instance()->background(), 0);
                        
                        reduced.image = std::move(img);
                        reduced.pos = pos;
                    }
                    
                    // normalize image size and write to normal queue
                    if(reduced.image) {
                        cv::Mat image;
                        reduced.image->get().copyTo(image);
                        
                        GUI::pad_image(image, output_size);
                        assert(image.cols == output_size.width && image.rows == output_size.height);
                        
                        queues[data.fish->identity().ID()][data.segment.range].push({ frame, data.fish->identity().ID(), Image::Make(image) });
                        
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
                        
                        if(do_normalize_tracklets)
                            full.image = std::get<0>(data.fish->calculate_normalized_diff_image(trans, full.blob, data.filters->median_midline_length_px, output_size, normalize == default_config::recognition_normalization_t::legacy));
                        else {
                            auto && [pos, img] = full.blob->difference_image(*Tracker::instance()->background(), 0);
                            full.image = std::move(img);
                            full.pos = pos;
                        }
                        
                        if(full.image) {
                            if(!do_normalize_tracklets) {
                                cv::Mat image;
                                full.image->get().copyTo(image);
                                
                                Vec2 offset = full.blob->bounds().pos() - reduced.blob->bounds().pos();
                                //image(Bounds(-offset, Size2(image) + offset)).copyTo(image);
                                
                                cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
                                
                                auto outline = data.fish->outline(frame);
                                if(outline) {
                                    auto points = outline->uncompress();
                                    for(auto pt : points) {
                                        //pt += reduced.blob->bounds().pos();
                                        pt += -offset;
                                        cv::circle(image, pt, 1, gui::Green);
                                    }
                                }
                                
                                Vec2 center_offset = ((reduced.blob->bounds().pos() + reduced.blob->bounds().size() * 0.5) - (full.blob->bounds().pos() + full.blob->bounds().size() * 0.5));
                                Vec2 org = offset + center_offset;
                                cv::rectangle(image, -org, -org + reduced.blob->bounds().size(), gui::Yellow);
                                
                                cv::rectangle(image, -offset, -offset + reduced.blob->bounds().size(), gui::White);
                                cv::circle(image, -offset + reduced.blob->bounds().size() * 0.5, 5, gui::Red);
                                cv::line(image, -offset + reduced.blob->bounds().size() * 0.5, -offset - center_offset + reduced.blob->bounds().size() * 0.5, gui::Yellow);
                                
                                cv::Mat empty_file = cv::Mat::zeros((int)output_size.height, (int)output_size.width, CV_8UC3);
                                Bounds input_bounds(-center_offset, Size2(image) + center_offset);
                                cv::putText(image, "input "+Meta::toStr(input_bounds), Vec2(20), cv::FONT_HERSHEY_PLAIN, 0.7, gui::White);
                                
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
                                
                                Vec2 gp(20, 40);
                                cv::putText(image, "left "+Meta::toStr(left), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                cv::putText(image, "top "+Meta::toStr(top), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                //cv::putText(image, "right "+Meta::toStr(right), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                //cv::putText(image, "bottom "+Meta::toStr(bottom), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                
                                cv::putText(image, "restricted "+Meta::toStr(input_bounds), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::Yellow); gp += Vec2(0, 20);
                                cv::putText(image, "size "+Meta::toStr(Size2(image)), gp, cv::FONT_HERSHEY_PLAIN, 0.7, gui::White); gp += Vec2(0, 20);
                                //input_bounds.restrict_to(Bounds(image));
                                
                                input_bounds.restrict_to(Bounds(empty_file));
                                
                                image(Bounds(left, top, input_bounds.width, input_bounds.height)).copyTo(empty_file(input_bounds));
                                empty_file.copyTo(image);
                                
                                //tf::imshow("full "+Meta::toStr(frame)+" "+Meta::toStr(reduced.blob->blob_id())+" "+Meta::toStr(Size2(image)), image);
                                
                                cv::Mat grey;
                                cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
                                full.image = Image::Make(grey);
                            }
                            
                            assert(full.image->cols == output_size.width && full.image->rows == output_size.height);
                            
                            split_ids.push_back(data.fish->identity().ID());
                            split_frames.push_back(frame.get());
                            split_masks.insert(split_masks.end(), full.image->data(), full.image->data() + full.image->size());
                        }
                    }
                }
                
                ++index;
                
                auto step = size_t(waiting_pixels.size() * 0.1);
                if(!waiting_pixels.empty() && step > 0 && index % step == 0) {
                    Debug("[tracklet_images] Frame %d/%d... (%.2f%% identities / frame)", index, waiting_pixels.size(), FAST_SETTINGS(manual_identities).empty() ? float(vec.size()) : float(vec.size()) / float(FAST_SETTINGS(manual_identities).size() + 0.0001) * 100);
                    if(GUI::instance())
                        GUI::instance()->work().set_percent(index / float(waiting_pixels.size()));
                }
            }
            
            // npz file sizes are limited to ~2GB, so we need to split up the files if necessary
            size_t part_counter = 0;
            size_t byte_counter = 0;
            
            auto export_singles = [&](file::Path path) {
                if(!single_frames.empty()) {
                    path = file::Path(path.str() + "_part"+Meta::toStr(part_counter)+".npz");
                    
                    size_t samples = single_frames.size();
                    Debug("Saving single tracklet images to '%S'... (%d images)", &path.str(), samples);
                    
                    if(path.exists())
                        path.delete_file();
                    
                    temporary_save(path, [&](file::Path path){
                        cmn::npz_save(path.str(), "images", single_images.data(), { single_frames.size(), (size_t)output_size.height, (size_t)output_size.width }, "w");
                        cmn::npz_save(path.str(), "frames", single_frames, "a");
                        cmn::npz_save(path.str(), "ids", single_ids, "a");
                    });
                    
                    single_frames.clear();
                    single_ids.clear();
                    single_images.clear();
                    
                    ++part_counter;
                    byte_counter = 0;
                    
                } else {
                    Warning("Called export_singles, but single_frames is empty.");
                }
            };
            
            // output network data
            file::Path path = fishdata / ((std::string)SETTING(filename).value<file::Path>().filename() + "_tracklet_images.npz");
            file::Path single_path = fishdata / ((std::string)SETTING(filename).value<file::Path>().filename() + "_tracklet_images_single");
            
            if(!split_masks.empty()) {
                auto path = single_path.str() + "_splits_part";
                Debug("Saving split tracklet masks to '%S'... (%d images)", &path, split_frames.size());
                
                int64_t bytes_per_image = (int64_t)output_size.height * (int64_t)output_size.width;
                int64_t n_images = int64_t(1.5 *1000 * 1000 * 1000) / bytes_per_image;
                
                Debug("%d/%d images fit in 1.5GB", n_images, split_frames.size());
                
                int64_t offset = 0;
                size_t part = 0;
                int64_t N = narrow_cast<int64_t>(split_frames.size());
                
                while (offset < N) {
                    auto L = min(n_images, N - offset);
                    
                    auto sub_path = path + Meta::toStr(part) + ".npz";
                    ++part;
                    
                    Debug("Saving to '%S' from %d-%d (%d)", &sub_path, offset, offset+L, split_frames.size());
                    
                    temporary_save(sub_path, [&](file::Path path) {
                        cmn::npz_save(path.str(), "images",
                                      split_masks.data() + offset * (int64_t)output_size.height * (int64_t)output_size.width,
                                      { sign_cast<size_t>(L), (size_t)output_size.height, (size_t)output_size.width }, "w");
                        cmn::npz_save(path.str(), "frames", std::vector<long_t>(split_frames.begin() + offset, split_frames.begin() + offset + L), "a");
                        cmn::npz_save(path.str(), "ids", std::vector<long_t>(split_ids.begin() + offset, split_ids.begin() + offset + L), "a");
                    });
                    
                    offset += n_images;
                }
                
                /*size_t nfiles = ceil(bytes / (1.5 *1000 * 1000 * 1000));
                auto str = FileSize{bytes}.to_string();
                Debug("Need %d files to accommodate %S", nfiles, &str);
                
                cmn::npz_save(path, "images", split_masks.data(), { split_frames.size(), (size_t)output_size.height, (size_t)output_size.width }, "w");
                cmn::npz_save(path, "frames", split_frames, "a");
                cmn::npz_save(path, "ids", split_ids, "a");*/
            }
            
            for(auto && [id, ranges] : queues) {
                cv::Mat tmp;
                std::vector<std::vector<hist_utils::Hist>> M; // histograms
                cv::Mat1b med;              // median image
                
                for(auto && [range, images] : ranges) {
                    size_t image_count = 0;
                    hist_utils::init(M, med, (int)output_size.height, (int)output_size.width);
                    
                    while(!images.empty()) {
                        auto [frame, fid, image] = std::move(images.front());
                        images.pop();
                        
                        auto mat = image->get();
                        cv::Mat1b tmp;
                        mat.convertTo(tmp, CV_8UC1);
                        hist_utils::addImage(tmp, M, med);
                        
                        if(tracklet_max_images == 0) {
                            single_images.insert(single_images.end(), image->data(), image->data() + image->size());
                            single_frames.push_back(frame.get());
                            single_ids.push_back(fid);
                            byte_counter += image->size();
                        }
                        ++image_count;
                        
                        if(byte_counter >= 1.5 * 1000 * 1000 * 1000) {
                            // finish up this range
                            export_singles(single_path);
                        }
                    }
                    
                    if(image_count > 1) {
                        med.copyTo(tmp);
                        assert(tmp.isContinuous() && tmp.channels() == 1);
                        all_images.insert(all_images.end(), tmp.data, tmp.data + tmp.cols * tmp.rows);
                        all_ranges.push_back(id);
                        all_ranges.push_back(range.start.get());
                        all_ranges.push_back(range.end.get());
                        //tf::imshow("median"+fish->identity().name()+" - "+Meta::toStr(range.range), tmp);
                    }
                }
                
                if(id % max(1, int(ceil(queues.size() * 0.01))) == 0)
                    Debug("[tracklet_images] Fish %d...", id);
            }
            
            size_t samples = all_images.size() / (size_t)output_size.height / (size_t)output_size.width;
            Debug("Saving tracklet images to '%S'... (%d samples)", &path.str(), samples);
            
            temporary_save(path, [&](file::Path path){
                cmn::npz_save(path.str(), "images", all_images.data(), { samples, (size_t)output_size.height, (size_t)output_size.width }, "w");
                cmn::npz_save(path.str(), "meta", all_ranges.data(), { samples, 3 /* ID, frame_start, frame_end */ }, "a");
            });
            
            export_singles(single_path);
        }
        
    } catch(const UtilsException&) {}
    
    // reset values to previous setting
    SETTING(output_default_options) = previous_options;
    SETTING(output_graphs) = previous_graphs;
    
    SETTING(output_frame_window) = previous_output_frame_window;
    
}
}
