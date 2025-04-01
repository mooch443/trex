#include "IdentityHeatmap.h"
#include <tracking/Tracker.h>
#include <tracking/Individual.h>
#include <misc/cnpy_wrapper.h>
#include <gui/Export.h>
#include <file/DataLocation.h>
#include <tracking/IndividualManager.h>

using namespace track;

namespace cmn::gui {
namespace heatmap {

static std::map<std::string, std::vector<double>> _statistics;
static std::mutex _statistics_mutex;

template<typename T>
inline bool overlaps(const Range<T>& A, const Range<T>& B) {
    return A.start <= B.end && B.start <= A.end;
}

void push_timing(const char* name, double s) {
    std::lock_guard<std::mutex> guard(_statistics_mutex);
    auto &v = _statistics[name];
    if(v.size() != Grid::statistics_items)
        v.resize(Grid::statistics_items, infinity<double>());
    std::rotate(v.begin(), ++v.begin(), v.end());
    v.back() = s;
}

inline std::string get_stats() {
    decltype(_statistics) stats;
    
    {
        std::lock_guard<std::mutex> guard(_statistics_mutex);
        stats = _statistics;
    }
    
    std::stringstream ss;
    
    for(auto && [name, values] : stats) {
        double sum = 0, samples = 0;
        for(auto v : values)
            if(v != infinity<double>()) {
                sum += v;
                ++samples;
            }
        
        if(samples > 0)
            ss << "   " << name << ": " << sum / samples * 1000 << "ms" << std::endl;
    }
    
    return ss.str();
}

void Grid::print_stats(const std::string& title) {
    auto str = get_stats();
    Print(title.c_str(),"\n",str.c_str());
}

/**
 * GUI interface functions
 */

HeatmapController::HeatmapController() : uniform_grid_cell_size(0), stride(0), N(0), smooth_heatmap_factor(0) {
    
}

void HeatmapController::paint_heatmap() {
    sort_data_into_custom_grid();
    
    static Timer timer;
    timer.reset();
    
    if(smooth_heatmap_factor != 0) {
        if(_viridis.cols != N * stride * smooth_heatmap_factor)
            _viridis = gpuMat(N * stride * smooth_heatmap_factor, N * stride * smooth_heatmap_factor, CV_8UC4);
        
        grid_image->get().copyTo(_gpuGrid);
        cv::resize(_gpuGrid, _viridis, cv::Size(_viridis.cols, _viridis.rows), 0, 0, cv::INTER_CUBIC);
        assert(_viridis.type() == CV_8UC4);
        
    } else {
        grid_image->get().copyTo(_viridis);
        assert(_viridis.type() == CV_8UC4);
    }
    
    if(!_image || !_image->source() || (int)_image->source()->cols != _viridis.cols) {
        _image = std::make_shared<ExternalImage>(Image::Make(_viridis.rows, _viridis.cols, 4), Vec2());
    }
    
    _image->update_with(_viridis);
    //_viridis.copyTo(_image->get());
    _image->set_pos(Vec2());
    _image->set_scale(Vec2(smooth_heatmap_factor > 0 ? 1.0f / smooth_heatmap_factor : stride));
    
    push_timing("gui_heatmap", timer.elapsed());
    
    /*bool draw = _frame % 10 == 0;
    if(draw) {
        const double scale = stride * 0.15;
        
        static cv::Mat mat;
        cv::cvtColor(_viridis(Bounds(0, 0, _viridis.cols, _viridis.rows)), mat, cv::COLOR_BGRA2RGB);
        
        cv::resize(mat, mat, cv::Size(_viridis.cols * scale, _viridis.rows * scale), 0, 0, cv::INTER_CUBIC);
        
        for (uint32_t x = 0; x < N; ++x) {
            for (uint32_t y = 0; y < N; ++y) {
                cv::rectangle(mat, Vec2(x * scale, y * scale), Vec2((x+1)*scale, (y+1) * scale), White);
            }
        }
        cv::imshow("viridis", mat);
        cv::waitKey(1);
        //tf::imshow("viridis", mat);
    }*/
}

void HeatmapController::save() {
    update_variables();
    
    //_frame_context = 0;
    //_normalization = normalization_t::none;
    //custom_heatmap_value_range = Range<double>(-1, -1);
    
    size_t count_frames = 0, package_count = 0;
    size_t max_frames = sign_cast<size_t>((Tracker::end_frame() - Tracker::start_frame()).get());
    size_t print_step = max_frames / 10 + 1;

    std::vector<double> per_frame;
    uint64_t expected = uint64_t((max_frames + 1) * N * N * 2);
    const bool be_quiet = GlobalSettings::is_runtime_quiet();
    //if (!be_quiet) 
    {
        Print("Likely memory size: ", FileSize{ expected * sizeof(double) });
    }

    const uint64_t value_size = sizeof(decltype(per_frame)::value_type);
    const uint64_t maximum_package_size = uint64_t(4.0 * 1024.0 * 1024.0 * 1024.0 / double(value_size));
    bool enable_packages = expected >= maximum_package_size;

    Print("ValueSize=", value_size," MaximumPackageSize=",maximum_package_size);

    if (enable_packages) {
        per_frame.reserve(maximum_package_size);
    } else
        per_frame.reserve(expected);

    std::vector<long_t> frames;
    size_t package_index = 0;
    
    auto data_prefix = SETTING(data_prefix).value<file::Path>();
    auto fishdata = file::DataLocation::parse("output", data_prefix);
    if(!fishdata.exists())
        if(!fishdata.create_folder())
            throw U_EXCEPTION("Cannot create folder ",fishdata.str()," for saving fishdata.");

    auto save_package = [&]() {
        std::vector<size_t> shape = {
            package_count, 2, N, N
        };

        auto str = Meta::toStr(shape);
        Print("Done (", expected," / ", per_frame.size(),", shape ",str,").");
        auto source = _source;
        if(source.find('#') != std::string::npos)
            source = source.substr(0, source.find('#'));
        
        file::Path path = fishdata /
                ((std::string)SETTING(filename).value<file::Path>().filename()
                + "_heatmap" 
                + "_p" + Meta::toStr(package_index) + "_"
                + Meta::toStr(uniform_grid_cell_size) 
                + "_" + Meta::toStr(N) + "x" + Meta::toStr(N)
                + (source.empty() ? "" : ("_" + source))
                + ".npz");
        
        DebugHeader("Saving package ", package_index," to ", path, "...");
        temporary_save(path, [&](file::Path use_path) {
            cmn::npz_save(use_path.str(), "heatmap", per_frame.data(), shape);
            cmn::npz_save(use_path.str(), "frames", frames, "a");
            
            const auto frame_range = _frame_context.valid()
                ? _frame_context
                : Frame_t(narrow_cast<Frame_t::number_t>(FAST_SETTING(video_length)));
            cmn::npz_save(use_path.str(), "meta", std::vector<double>{
                (double)package_index,
                (double)uniform_grid_cell_size,
                (double)_normalization.value(),
                (double)frame_range.get()
            }, "a");
        });

        Print("Saved to ", path.str(),".");

        per_frame.clear();
        frames.clear();
        package_count = 0;
        ++package_index;
    };

    for(Frame_t frame = Tracker::start_frame(); frame <= Tracker::end_frame(); ++frame) {
        update_data(frame);
        sort_data_into_custom_grid();
        //set_frame(frame);
        per_frame.insert(per_frame.end(), _array_grid.begin(), _array_grid.end());
        per_frame.insert(per_frame.end(), _array_samples.begin(), _array_samples.end());
        frames.push_back(frame.get());
        
        if(!be_quiet && count_frames % print_step == 0) {
            Print("Saving heatmap ",dec<2>(double(count_frames) / double(max_frames) * 100),"% ... (frame ",frame," / ",Tracker::end_frame(),")");
        }

        ++count_frames;
        ++package_count;
        
        if (enable_packages && per_frame.size() >= maximum_package_size) {
            auto size0 = FileSize{ per_frame.size() * sizeof(decltype(per_frame)::value_type) }.to_string(), 
                 size1 = FileSize{ maximum_package_size * sizeof(decltype(per_frame)::value_type) }.to_string();
            Print("Splitting package at ",size0," / ",size1,".");

            save_package();
        }
    }

    save_package();
    update_variables();
}

void HeatmapController::sort_data_into_custom_grid() {
    static Timer timer;
    timer.reset();
    
    double minimum = 0, maximum = 0;
    std::vector<double> values;
    std::fill(_array_samples.begin(), _array_samples.end(), _normalization == normalization_t::cell ? 1 : 0);
    std::fill(_array_grid.begin(), _array_grid.end(), 0);
    if(_normalization == normalization_t::variance)
        std::fill(_array_sqsum.begin(), _array_sqsum.end(), 0);
    
    if(_normalization == normalization_t::cell
       && isPowerOfTwo(uniform_grid_cell_size))
    {
        _grid.apply<Region>([this, &maximum, stride = double(stride)](const Region &r) -> bool {
            if(r.pixel_size() > uniform_grid_cell_size)
                return true;
            
            auto cx = sign_cast<uint64_t>((double((double)r.x().start + 0.5) / stride));
            auto cy = sign_cast<uint64_t>((double((double)r.y().start + 0.5) / stride));
            
            size_t i = cy * N + cx;
            _array_grid.at(i) += r.size();
            if(r.size() > maximum)
                maximum = r.size();
            
            //! do not traverse deeper
            return false;
        });
        
    } else if(isPowerOfTwo(uniform_grid_cell_size)
              && is_in(_normalization, normalization_t::none, normalization_t::value, normalization_t::variance))
    {
        values.clear();
        
        _grid.apply<Region>([this, &values, stride = double(stride)](const Region &r) -> bool {
            if(r.pixel_size() > uniform_grid_cell_size)
                return true;
            
            auto cx = sign_cast<uint64_t>((double((double)r.x().start + 0.5) / stride));
            auto cy = sign_cast<uint64_t>((double((double)r.y().start + 0.5) / stride));
            
            size_t i = cy * N + cx;
            
            _array_grid.at(i) += r.value_sum();
            _array_samples[i] += r.size();
            
            if(_normalization == normalization_t::variance) {
                _array_sqsum.at(i) += r.value_sqsum();
                
            } else {
                if(_normalization != normalization_t::none) {
                    values.push_back(r.value_range().start);
                    values.push_back(r.value_sum() / double(r.size()));
                    values.push_back(r.value_range().end);
                }
            }
            
            //! do not traverse deeper
            return false;
        });
        
    } else {
        if(_normalization != normalization_t::none)
            values.reserve(_grid.size());
        
        _grid.apply<Leaf>([&](const Leaf& leaf) -> bool {
            auto cx = sign_cast<uint64_t>((double((double)leaf.x().start + 0.5) / double(stride)));
            auto cy = sign_cast<uint64_t>((double((double)leaf.y().start + 0.5) / double(stride)));
            
            size_t i = cy * N + cx;
            assert((size_t)i < _array_grid.size());

            if(_normalization == normalization_t::value) {
                double v = 0;
                
                for(auto &point : leaf.data()) {
                    //if(point.value > maximum)
                    //    maximum = point.value;
                    v += point.value;
                    if(_normalization != normalization_t::none)
                        values.push_back(point.value);
                }
                
                _array_samples[i] += leaf.data().size();
                _array_grid[i] += v;
                
            } else {
                if(_normalization == normalization_t::variance) {
                    _array_sqsum.at(i) += leaf.value_sqsum();
                }
                
                _array_grid[i] += leaf.data().size();
            }
            
            return true;
        });
    }
    
    // calculate standard deviation within each cell
    if(_normalization == normalization_t::variance) {
        minimum = maximum = infinity<double>();
        auto m = _grid.root()->value_sum() / double(_grid.size());
        
        for (size_t i=0; i<_array_grid.size(); ++i) {
            if(_array_samples[i] > 1) {
                //auto m = _array_grid[i] / _array_samples[i];
                _array_grid[i] = sqrt((- 2 * m * _array_grid[i] + _array_sqsum[i] + N * SQR(m))
                                        / (_array_samples[i] - 1));
                
                if(minimum > _array_grid[i] || minimum == infinity<double>()) {
                    minimum = _array_grid[i];
                }
                if(maximum < _array_grid[i] || maximum == infinity<double>()) {
                    maximum = _array_grid[i];
                }
                
            } else
                _array_grid[i] = _array_samples[i] = 0;
        }
    }
    
    if(_normalization != normalization_t::none // none never does any normalization
       && !custom_heatmap_value_range.empty())
    {
        minimum = custom_heatmap_value_range.start;
        maximum = custom_heatmap_value_range.end;
        
    } else {
        switch(_normalization) {
            case normalization_t::variance:
                break;
                
            case normalization_t::cell:
                if(!_array_grid.empty()) {
                    minimum = 0;
                    maximum = *std::max_element(_array_grid.begin(), _array_grid.end());
                }
                break;
                
            case normalization_t::value:
                if(!values.empty()) {
                    std::sort(values.begin(), values.end());
                    auto percentiles = percentile(values, {0.05, 0.95});
                    //minimum = max(percentiles.front(), minimum);
                    maximum = min(percentiles.back(), maximum);
                    break;
                }
                
            case normalization_t::none:
            default:
                minimum = 0;
                maximum = 1;
                break;
        }
    }
    
    if(!grid_image || grid_image->bounds().size() != Size2(N, N))
        grid_image = Image::Make(N, N, 4);
    
    auto mat = grid_image->get();
    
    static auto empty = (cv::Scalar)cmap::ColorMap::value<cmap::CMaps::viridis>(0.0).alpha(0);
    static auto empty_variance = (cv::Scalar)cmap::ColorMap::value<cmap::CMaps::viridis>(1.0).alpha(200);
    if(_normalization == normalization_t::variance)
        mat.setTo(empty_variance);
    else
        mat.setTo(empty);
    
    double percentage;
    auto ML = maximum - minimum;
    if(ML == 0)
        ML = 1;
    
    auto samples = _array_samples.data();
    auto grid_values = _array_grid.data();
    
    static_assert(sizeof(Color) == sizeof(cv::Vec4b), "sizeof(Color) and cv::Vec4b are assumed to be equal.");
    for (auto ptr = (Color*)grid_image->data(), to = ptr + grid_image->cols * grid_image->rows; ptr != to; ++ptr, ++samples, ++grid_values)
    {
        if(*samples > 0) {
            percentage = (*grid_values / *samples - minimum) / ML;
            if(_normalization == normalization_t::variance)
                percentage = 1 - percentage;
            
            *ptr = cmap::ColorMap::value<cmap::CMaps::viridis>(percentage).alpha(uint8_t(percentage * 200));
        }
    }
    
    /*for (uint32_t x = 0; x < N; ++x) {
        for (uint32_t y = 0; y < N; ++y) {
            size_t i = y * N + x;
            if(_array_samples[i] > 0) {
                percentage = (_array_grid[i] / _array_samples[i] - minimum) / ML;
                if(_normalization == normalization_t::variance)
                    percentage = 1 - percentage;
                mat.at<cv::Vec4b>(y, x) = Viridis::value(percentage).alpha(uint8_t(percentage * 200));
            }
        }
    }*/
    //tf::imshow("Viridis", mat);
        
    push_timing("sort_data_into_custom_grid", timer.elapsed());
}

void HeatmapController::frames_deleted_from(Frame_t frame) {
    _iterators.clear();
    _capacities.clear();
    _grid.keep_only(Range<Frame_t>(0_f, frame.try_sub(1_f)));
}

HeatmapController::UpdatedStats HeatmapController::update_data(Frame_t current_frame) {
    Timer timer;
            
    static std::vector<heatmap::DataPoint> data;
    UpdatedStats updated;
    
    if(current_frame.valid()) {
        auto d = _frame.valid()
                ? ((current_frame >= _frame)
                   ? (current_frame - _frame)
                   : (_frame - current_frame))
                : current_frame;
        const auto frame_range = _frame_context.valid()
            ? _frame_context
            : Frame_t(narrow_cast<Frame_t::number_t>(FAST_SETTING(video_length)));
        
        if(not _frame.valid()
           || _grid.empty()
           || (_frame_context.valid() && d >= _frame_context))
        {
            // we cant use any frames from before
            updated.removed = _grid.size();
            _grid.clear();
            updated.add_range = Range<Frame_t>(current_frame.try_sub(frame_range),
                                               current_frame + frame_range + 1_f);
            _iterators.clear();
            _capacities.clear();
            
        } else if(_frame_context.valid()) {
            if(current_frame > _frame) {
                //removed = _grid.erase(Range<long_t>(0, max(0, current_frame - frame_range)));
                //remove_range = Range<long_t>(0, max(0u, current_frame - frame_range));
                updated.add_range = Range<Frame_t>(_frame + frame_range + 1_f,
                                                   current_frame + frame_range + 1_f);
            } else {
                //removed = _grid.erase(Range<long_t>(current_frame + frame_range + 1, std::numeric_limits<long_t>::max()));
                //remove_range = Range<long_t>(current_frame + frame_range + 1, std::numeric_limits<long_t>::max());
                updated.add_range = Range<Frame_t>(current_frame.try_sub(frame_range),
                                                  min((_frame.valid() ? _frame.try_sub(frame_range) : 0_f), current_frame + frame_range + 1_f));
            }
            
            updated.remove_range = Range<Frame_t>(current_frame.try_sub(frame_range),
                                                  current_frame + frame_range + 1_f);
        }
        
        //if(!remove_range.empty())
        //    removed = _grid.erase(remove_range);
        
        if(!updated.remove_range.empty()) {
            updated.removed = _grid.keep_only(updated.remove_range);
        }
        
        if(!updated.add_range.empty()) {
            data.clear();
            data.reserve(frame_range.get() * 2u * max(1u, FAST_SETTING(track_max_individuals)));
            Individual::tracklet_map::const_iterator kit;
            
            auto &range = updated.add_range;
            IndividualManager::transform_all([&](auto id, auto fish) {
                if(!_ids.empty()) {
                    if(!contains(_ids, id)) {
                        return;
                    }
                }
                
                auto frame = max(Tracker::start_frame(), range.start);
                if(fish->empty())
                    return;
                if(fish->end_frame() < frame)
                    return;
                if(fish->start_frame() > range.end)
                    return;
                
                auto it = _iterators.find(fish);
                if(it == _iterators.end()) {
                    kit = fish->iterator_for(frame);
                } else {
                    if(_capacities[fish] != fish->tracklets().capacity()) {
                        _capacities[fish] = fish->tracklets().capacity();
                        kit = fish->iterator_for(frame);
                    } else
                        kit = it->second;
                }
                
                if(kit == fish->tracklets().end() && range.end >= fish->start_frame())
                {
                    kit = fish->iterator_for(fish->start_frame());
                }
                
                if(kit != fish->tracklets().end() && !(*kit)->contains(frame)) {
                    if((*kit)->end() < frame) {
                        
                        // everything okay
                        do {
                            ++kit;
                        } while(kit != fish->tracklets().end() && (*kit)->end() < frame);
                        
                    } else if(fish->has(frame)) {
                        kit = fish->iterator_for(frame);
                    }
                }
                
//                       if(kit == fish->tracklets().end())
                Output::Library::LibInfo info(fish, _mods);
                
                for(; frame < min(Tracker::end_frame(), range.end); ++frame) {
                    if(_grid.root()->frame_range().contains(frame))
                        continue;
                    //break;
                    //auto basic = fish->basic_stuff(frame);
                    //if(kit == fish->tracklets().end())
                    //    continue;
                    
                    while(kit != fish->tracklets().end() && frame > (*kit)->end()) {
                        ++kit;
                        //if(kit == fish->tracklets().end())
                        //    break; // no point in trying to find more data
                    }
                    
#ifndef NDEBUG
                    auto kiterator = fish->iterator_for(frame);
                    auto is_end = kiterator == fish->tracklets().end();
                    auto is_end_kit = kit == fish->tracklets().end();
                    if(fish->has(frame) && kit != kiterator)
                        FormatWarning("Frame ",frame,": fish",fish->identity().ID(),", Iterator for frame ",frame," != iterator_for (iterator_for: ",is_end ? 1 : 0,", starting at ",!is_end ? kiterator->get()->start() : Frame_t()," / vs. kit: ",is_end_kit,", starting at ",!is_end_kit ? kit->get()->start() : Frame_t(),")");
#endif
                    
                    if(kit == fish->tracklets().end() || !(*kit)->contains(frame))
                        continue; // skipping some frames in between
                    
                    auto bid = (*kit)->basic_stuff(frame);
                    if(bid != -1) {
                        auto &basic = fish->basic_stuff()[(uint32_t)bid];
                        auto pos = basic->centroid.template pos<Units::PX_AND_SECONDS>();
                        //auto speed = basic->centroid->speed(Units::PX_AND_SECONDS);
                        
                        double v = 1;
                        if(!_source.empty())
                            v = Output::Library::get_with_modifiers(_source, info, frame);
                        if(!GlobalSettings::is_invalid(v)) {
                            data.push_back(heatmap::DataPoint{
                                .frame   = frame,
                                .x       = uint32_t(pos.x),
                                .y       = uint32_t(pos.y),
                                .ID      = uint32_t(fish->identity().ID().get()),
                                .IDindex = uint32_t(0),
                                .value   = v
                            });
                        }
                    }
                }
                
                _iterators[fish] = kit;
            });
            
            updated.added = data.size();
            _grid.fill(data);
        }
        
        /*size_t counter = 0;
        Range<long_t> range(-1, -1);
        _grid.root()->apply([current_frame, &counter, &range, this](const DataPoint & pt) -> bool {
            ++counter;
            
            if(_frame_context > 0) {
                if(pt.frame < current_frame - _frame_context) {
                    Print("Encountered a wild ", pt.frame," < ",current_frame - _frame_context);
                } else if(pt.frame > current_frame + _frame_context)
                    Print("Encountered a wild ", pt.frame," > ",current_frame + _frame_context);
            }
            
            if(range.start == -1 || range.start > pt.frame) range.start = pt.frame;
            if(range.end < pt.frame + 1) range.end = pt.frame + 1;
            return true;
        });
        assert(_grid.root()->frame_range() == range);
        
        //if(_frame % 50 == 0)
        Print("Frame ",current_frame,": ",data.size()," elements (added ",updated.added,", (removed)",updated.removed," + (replaced)",0,", range ",range.start,"-",range.end,", reported ",_grid.root()->frame_range().start,"-",_grid.root()->frame_range().end,")");*/
    }
    
    //auto str = Meta::toStr(data);
    _frame = current_frame;
    push_timing("Heatmap::update_data()", timer.elapsed());
    
    return updated;
}

void HeatmapController::update() {
    if(!content_changed())
        return;
    
    OpenContext([this]{
        if(_image)
            advance_wrap(*_image);
    });
    
    auto_size(Margin{0, 0});
}

bool HeatmapController::update_variables() {
    bool has_to_paint = false;
    
    if(!_grid.root()) {
        _grid.create(track::Tracker::average().bounds().size());
        has_to_paint = true;
        set_content_changed(true);
    }
    
    const uint32_t res = max(2u, min((uint32_t)(Tracker::average().bounds().size().min() * 0.5),  SETTING(heatmap_resolution).value<uint32_t>()));
    
    if(res != uniform_grid_cell_size) {
        uniform_grid_cell_size = res;
        stride = uniform_grid_cell_size;
        N = ceil(double(Tracker::average().bounds().size().max()) / double(stride));
        
        if(_array_grid.size() != N * N) {
            _array_grid.resize(N * N);
            _array_sqsum.resize(N * N);
            _array_samples.resize(N * N);
        }
        
        has_to_paint = true;
    }
    
    const Range<double> custom_value_range = SETTING(heatmap_value_range).value<Range<double>>();
    if(custom_value_range != custom_heatmap_value_range) {
        custom_heatmap_value_range = custom_value_range;
        has_to_paint = true;
    }
    
    const double heatmap_smooth = max(0, min(1, SETTING(heatmap_smooth).value<double>()));
    if(smooth_heatmap_factor != heatmap_smooth) {
        smooth_heatmap_factor = heatmap_smooth;
        has_to_paint = true;
    }
    
    //SETTING(heatmap_resolution) = uniform_grid_cell_size + 1;
    auto ids = SETTING(heatmap_ids).value<std::vector<Idx_t>>();
    if(ids != _ids) {
        has_to_paint = true;
        
        bool different = _ids.size() >= ids.size() || (_ids.empty() && !ids.empty());
        if(!different) {
            for(auto id : _ids) {
                if(!contains(ids, id)) {
                    different = true;
                    break;
                }
            }
        }
        
        _ids = ids;
        
        // only completely clear if an id was removed, not if we are only adding one
        if(different) {
            _grid.clear();
        }
        
        _frame.invalidate();
    }
    
    auto norm = SETTING(heatmap_normalization).value<default_config::heatmap_normalization_t::Class>();
    if(norm != _normalization) {
        _normalization = norm;
        has_to_paint = true;
    }
    
    Frame_t context;
    if(SETTING(heatmap_dynamic)) {
        context = max(1_f, Frame_t(SETTING(heatmap_frames).value<uint32_t>()));
        
    } else {
        context.invalidate();
    }
    
    if(_frame_context != context) {
        has_to_paint = true;
        _frame.invalidate();
        _frame_context = context;
    }
    
    std::string source = SETTING(heatmap_source);
    
    if(_original_source != source) {
        _original_source = source;
        
        _mods.clear();
        auto array = utils::split(source, '#');
        if(array.size() > 0) {
            for(size_t i=1; i<array.size(); ++i)
                Output::Library::parse_modifiers(array[i], _mods);
            source = array[0];
        }
        
        _source = source;
        
        _frame.invalidate();
        _grid.clear();
        has_to_paint = true;
    }
    
    if(has_to_paint)
        set_content_changed(true);
    
    return has_to_paint;
}

void HeatmapController::set_frame(Frame_t current_frame) {
    bool has_to_paint = update_variables();
    
    //! check if we have to update the data
    if(not _frame.valid() || current_frame != _frame) {
        auto updated = update_data(current_frame);
        if(updated.added != 0 || updated.removed != 0)
            has_to_paint = true;
        
        if(_frame.valid()
           && _frame.get() % 50 == 0)
        {
            Print("-------------------");
            Grid::print_stats("STATS (frame "+Meta::toStr(_frame)+", "+Meta::toStr(_grid.root()->IDs())+")");
            Print("");
        }
    }
    
    if(has_to_paint) {
        paint_heatmap();
    }
}

/**
 *  =====================
 *       CACHE STUFF
 *  =====================
 */
static std::queue<Node::Ptr> _cached_regions;
static std::queue<Node::Ptr> _cached_leafs;
static std::mutex _cache_mutex;

inline Node::Ptr retrieve_region() {
    //return new Region();//std::make_shared<Region>();
    
    //std::lock_guard<std::mutex> guard(_cache_mutex);
    Node::Ptr ptr;
    if(!_cached_regions.empty()) {
        ptr = _cached_regions.front();
       //_cached_regions.erase(--_cached_regions.end());
        _cached_regions.pop();
    } else {
        ptr = new Region();//std::make_shared<Region>();
    }
    
    return ptr;
}

inline Node::Ptr retrieve_leaf() {
    //return new Leaf();//std::make_shared<Leaf>();
    
   // std::lock_guard<std::mutex> guard(_cache_mutex);
    Node::Ptr ptr;
    if(!_cached_leafs.empty()) {
        ptr = _cached_leafs.front();
        //_cached_leafs.erase(--_cached_leafs.end());
        _cached_leafs.pop();
    } else {
        ptr = new Leaf();//std::make_shared<Leaf>();
    }
    
    return ptr;
}

inline void push_region(Node::Ptr ptr) {
    //delete ptr;
    //std::lock_guard<std::mutex> guard(_cache_mutex);
    //_cached_regions.push_back(ptr);
    ptr->clear();
    _cached_regions.push(ptr);
    
    //if(_cached_regions.size() % 10000 == 0)
}

inline void push_leaf(Node::Ptr ptr) {
    //delete ptr;
    //std::lock_guard<std::mutex> guard(_cache_mutex);
    //_cached_leafs.push_back(ptr);
    ptr->clear();
    _cached_leafs.push(ptr);
    
    //if(_cached_leafs.size() % 10000 == 0)
}

void Node::init(const Grid* grid, Node::Ptr parent, const Range<uint32_t>& x, const Range<uint32_t>& y) {
    _x = x;
    _y = y;
    _parent = parent;
    _grid = grid;
    
    _frame_range.start.invalidate();
    _frame_range.end.invalidate();
    _value_sum = 0;
    _value_sqsum = 0;
    _value_range = Range<double>(infinity<double>(), infinity<double>());
    _IDs.clear();
}

void Node::clear() {
    _frame_range.start.invalidate();
    _frame_range.end.invalidate();
    //_parent = nullptr;
    //_grid = nullptr;
    //_IDs.clear();
    //_x.start = _x.end = std::numeric_limits<uint32_t>::max();
    //_y.start = _y.end = std::numeric_limits<uint32_t>::max();
}

void Leaf::clear() {
    Node::clear();
    _data.clear();
}


void Grid::create(const Size2 &image_dimensions) {
    auto dim = sign_cast<uint32_t>(image_dimensions.max());
    Print(image_dimensions, " -> ", next_pow2<uint32_t>(dim), " and ", dim, " vs ", next_pow2<uint32_t>(1280), " ", Size2(1280,720).max());
    dim = next_pow2<uint32_t>(dim); // ensure that it is always divisible by two
    Print("Creating a grid of size ",dim,"x",dim," (for image of size ",image_dimensions.width,"x",image_dimensions.height,")");
    
    if(_root) {
        _root->clear();
        delete _root;
    }
    
    _root = new Region();//std::make_shared<Region>();
    _root->init(this, nullptr, Range<uint32_t>(0, dim), Range<uint32_t>(0, dim), dim);
    _elements = 0;
}

size_t Grid::erase(Range<Frame_t> frames) {
    static Timer timer;
    timer.reset();
    
    auto removed = _root->erase(frames);
    assert(removed <= _elements);
    _elements -= removed;
    
    push_timing("Grid::erase", timer.elapsed());
    
    return removed;
}

const Range<Frame_t>& Node::frame_range() const {
    return _frame_range;
}

size_t Region::keep_only(const Range<Frame_t> &frames) {
    assert(overlaps(frames, _frame_range));
    
    size_t count = 0;
    for(auto & r : _regions) {
        if(!r)
            continue;
        
        if(overlaps(frames, r->frame_range())) {
            count += r->keep_only(frames);
            
            if(r->empty()) {
                if(r->is_leaf()) push_leaf(r);
                else push_region(r);
                r = nullptr;
            }
        }
    }
    
    if(count)
        update_ranges();
    return count;
}

size_t Region::erase(const Range<Frame_t> &frames) {
    if(!overlaps(frames, _frame_range))
        return 0;
    
    size_t count = 0;
    
    for(auto & r : _regions) {
        if(r) {
            count += r->erase(frames);
            
            if(r->empty()) {
                if(r->is_leaf())
                    push_leaf(r);
                else
                    push_region(r);
                r = nullptr;
            }
        }
    }
    
    if(count > 0)
        update_ranges();
    return count;
}

#ifndef NDEBUG
template<typename T>
bool float_equals(T a, T b) {
    static constexpr auto tolerance = std::numeric_limits<float>::epsilon();
    T diff = cmn::abs(a - b);
    if (diff <= tolerance)
        return true;

    if (diff < cmn::max(cmn::abs(a), cmn::abs(b)) * tolerance)
        return true;
    
    return false;
}
#endif

void Region::check_range() const {
#ifndef NDEBUG
    Range<Frame_t> range({},{});
    Range<double> vrange(infinity<double>(), infinity<double>());
    double sum = 0;
    size_t count = 0;
    std::vector<uint32_t> ids;
    ids.resize(_grid->identities().size(), 0);
    
    apply([&range, &count, &sum, &ids, &vrange](auto &pt) -> bool
    {
        if(!range.start.valid() || pt.frame < range.start) range.start = pt.frame;
        if(!range.end.valid() || pt.frame + 1_f > range.end) range.end = pt.frame + 1_f;
        
        if(vrange.start > pt.value) vrange.start = pt.value;
        if(vrange.end == infinity<double>() || vrange.end < pt.value) vrange.end = pt.value;
        
        ++ids[pt.IDindex];
        ++count;
        
        sum += pt.value;
        return true;
    });
    if(_frame_range != range)
        FormatWarning("Frame range ",_frame_range.start,"-",_frame_range.end," != ",range.start,"-",range.end," actual range");
    if(count != _size)
        FormatWarning("Size (", count,") does not match reported size (",_size,").");
    if(!float_equals(sum, _value_sum))
        FormatWarning("Value sum (", sum,") does not match reported (",_value_sum,").");
    if(vrange != _value_range)
        FormatWarning("Value range (",vrange.start,"-",vrange.end,") does not match reported (",_value_range.start,"-",_value_range.end,").");
    /*if(ids != _IDs) {
        auto str0 = Meta::toStr(ids);
        auto str1 = Meta::toStr(_IDs);
        FormatWarning("IDs ", str0," did not match reported IDs ",str1,".");
    }*/
#endif
}

std::string DataPoint::toStr() const {
    return "DataPoint<" + Meta::toStr(frame) + "," + Meta::toStr(x) + "," + Meta::toStr(y) + ">";
}

size_t Leaf::keep_only(const Range<Frame_t> &frames) {
    size_t count = _data.size();
    
    auto it = std::upper_bound(_data.begin(), _data.end(), frames.start.try_sub(1_f), [](Frame_t frame, const DataPoint& A) -> bool
    {
        return frame < A.frame;
    });
    
    if(_data.begin() != it)
        _data.erase(_data.begin(), it);
    
    it = std::upper_bound(_data.begin(), _data.end(), frames.end.try_sub(1_f), [](Frame_t frame, const DataPoint& A) -> bool
    {
        return frame < A.frame;
    });
    
    if(_data.end() != it)
        _data.erase(it, _data.end());
    
    update_ranges();
    return count - _data.size();
}

void Leaf::update_ranges() {
    _value_range = Range<double>(infinity<double>(), infinity<double>());
    _value_sum = 0;
    _value_sqsum = 0;
    
    /*if(_IDs.size() != _grid->identities().size()) {
        _IDs.resize(_grid->identities().size());
        _values_per_id.resize(_grid->identities().size());
        _value_range_per_id.resize(_grid->identities().size());
    }
    
    std::fill(_IDs.begin(), _IDs.end(), 0);
    std::fill(_values_per_id.begin(), _values_per_id.end(), 0);
    std::fill(_value_range_per_id.begin(), _value_range_per_id.end(), Range<double>(infinity<double>(), infinity<double>()));*/
    
    for(auto &d : _data) {
        if(_value_range.start > d.value)
            _value_range.start = d.value;
        if(_value_range.end == infinity<double>() || _value_range.end < d.value)
            _value_range.end = d.value;
        
        _value_sum += d.value;
        _value_sqsum += SQR(d.value);
        
        /*++_IDs[d.IDindex];
        _values_per_id[d.IDindex] += d.value;
        
        if(_value_range_per_id[d.IDindex].start > d.value)
            _value_range_per_id[d.IDindex].start = d.value;
        if(_value_range_per_id[d.IDindex].end == infinity<double>() || _value_range_per_id[d.IDindex].end < d.value)
            _value_range_per_id[d.IDindex].end = d.value;*/
    }
    
    _frame_range.start = empty() ? Frame_t() : min_frame();
    _frame_range.end = empty() ? Frame_t() : (max_frame() + 1_f);
}

size_t Leaf::erase(const Range<Frame_t> &frames) {
    //static Timing timing("Leaf::erase", 0.01);
    //TakeTiming take(timing);
    
    size_t count = _data.size();
    auto it = std::lower_bound(_data.begin(), _data.end(), frames.start, [](const DataPoint& A, Frame_t frame) -> bool
    {
        return A.frame < frame;
    });
    
    if(it != _data.end()) {
        if(_data.back().frame < frames.end) {
            //_value_sum -= std::accumulate(it, _data.end(), 0);
            _data.erase(it, _data.end());
            update_ranges();
        } else {
            auto end = std::upper_bound(it, _data.end(), frames.end.try_sub(1_f), [](Frame_t frame, const DataPoint& A) -> bool
            {
                return frame < A.frame;
            });
            
            if(end != _data.end()) {
                //_value_sum -= std::accumulate(it, end, 0);
                _data.erase(it, end);
                update_ranges();
            }
        }
    }
    
    return count - _data.size();
}

Frame_t Leaf::min_frame() const {
    return _data.empty() ? Frame_t() : _data.front().frame;
}

Frame_t Leaf::max_frame() const {
    return _data.empty() ? Frame_t() : _data.back().frame;
}

Grid::~Grid() {
    if(_root) {
        _root->clear();
        //_root = nullptr;
        delete _root;
    }
    
}

void Grid::prepare_data(std::vector<DataPoint> &data) {
    for(auto &d : data) {
        auto it = _identity_aliases.find(d.ID);
        if(it != _identity_aliases.end())
            d.IDindex = it->second;
        else {
            _identity_aliases[d.ID] = d.IDindex = narrow_cast<uint32_t>(_identities.size());
            assert(!contains(_identities, d.ID));
            _identities.push_back(d.ID);
        }
    }
}

void Grid::fill(const std::vector<DataPoint> &data)
{
    static Timer timer;
    timer.reset();
    
    if(!_root)
        throw U_EXCEPTION("Have to create a grid first.");
    
    std::vector<DataPoint> copy = data;
    prepare_data(copy);
    
    _root->insert(copy);
    _elements += copy.size();
    
    push_timing("Grid::fill", timer.elapsed());
    
#if false
    static size_t index = 0;
    ++index;
    
    //if(index % 20 != 0)
    //    return;
    
    
    static Image image(_root->pixel_size(), _root->pixel_size(), 3);
    bool draw = index % 20 == 0;
    //draw = false;
    
    //if(draw)
    //    std::fill(image.data(), image.data() + image.size(), 0);
    auto mat = image.get();
    
    const uint32_t uniform_grid_cell_size = 64;
    const uint32_t stride = uniform_grid_cell_size;
    const uint32_t N = ceil(double(Size2(Tracker::average()).max()) / double(stride));
    
    {
        static Timer timer;
        timer.reset();
        
        double maximum = 0;
        static std::vector<double> grid(N * N), samples(N * N);
        std::fill(grid.begin(), grid.end(), 0);
        std::fill(samples.begin(), samples.end(), 0);
        
        apply<Leaf>([&](const Leaf& leaf) -> bool {
            int64_t cx = double(leaf.x().start + 0.5) / double(stride);
            int64_t cy = double(leaf.y().start + 0.5) / double(stride);
            
            size_t i = cy * N + cx;
            assert((size_t)i < grid.size());
            
            for(auto &point : leaf.data()) {
                if(point.value > maximum)
                    maximum = point.value;
                
                grid[i] += point.value;
            }
            
            samples[i] += leaf.size();
            
            return true;
        });
        
        static Image grid_image(N, N, 3);
        grid_image.get().setTo((cv::Scalar)Viridis::value(0));
        //std::fill(grid_image.data(), grid_image.data() + grid_image.size(), 0);
        
        auto mat = grid_image.get();
        
        for (uint32_t x = 0; x < N; ++x) {
            for (uint32_t y = 0; y < N; ++y) {
                size_t i = y * N + x;
                if(grid.at(i) > 0)
                    mat.at<cv::Vec3b>(y, x) = Viridis::value(grid[i] / maximum);
                    //cv::rectangle(mat, Vec2(x, y) * stride, (Vec2(x, y) + 1) * stride, Viridis::value(grid.at(y * N + x) / maximum), -1);
            }
        }
        
        push_timing("Grid::grid_cell", timer.elapsed());
        
        if(draw) {
            //static Image big(2048, 2048, 3);
            //cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
            
            static gpuMat viridis(N * stride, N * stride, CV_8UC3);
            static gpuMat gpuGrid;
            mat.copyTo(gpuGrid);
            //viridis.setTo(cv::Scalar(Black));
            cv::resize(gpuGrid, viridis, cv::Size(viridis.cols, viridis.rows), 0, 0, cv::INTER_CUBIC);
            viridis.copyTo(image.get()(Bounds(0, 0, viridis.cols, viridis.rows)));
            
            /*for(auto cell : cells) {
                cv::rectangle(image.get(), Vec2(cell->x().start, cell->y().start) + 1, Vec2(cell->x().end, cell->y().end) - 1, Color(0, 50, 125, 255).brighten(0.5), -1);
            }*/
        }
    }
    
    size_t objects_grid = 0, objects_grid_filtered = 0, objects_upper = 0, objects_grid_leaf = 0, objects_grid_region = 0, objects = 0, objects_loop = 0, objects_queue = 0;
    
    {
        Timer timer;
        apply<DataPoint>([&](const DataPoint &pt){
            ++objects_grid_filtered;
            return true;
        }, Range<long_t>(-1,-1));
        push_timing("Grid::apply<DataPoint>::filtered", timer.elapsed());
    }
    
    {
        Timer timer;
        apply<DataPoint>([&](const DataPoint &pt){
            ++objects_upper;
            return true;
        }, Range<long_t>(-1,-1), Range<uint32_t>(0, _root->pixel_size() / 2), Range<uint32_t>(0, _root->pixel_size() / 2));
        push_timing("Grid::apply<DataPoint>::filtered::upper", timer.elapsed());
    }
    
    {
        Timer timer;
        apply<DataPoint>([&](const DataPoint &pt){
            ++objects_grid;
            return true;
        });
        push_timing("Grid::apply<DataPoint>", timer.elapsed());
        assert(objects_grid_filtered == objects_grid);
    }
    
    {
        Timer timer;
        apply<Leaf>([&](const Leaf &pt){
            ++objects_grid_leaf;
            return true;
        });
        push_timing("Grid::apply<Leaf>", timer.elapsed());
    }
    
    {
        Timer timer;
        apply<Region>([&](const Region &pt){
            ++objects_grid_region;
            return true;
        });
        push_timing("Grid::apply<Region>", timer.elapsed());
    }
    
    {
        Timer timer;
        _root->apply([&](const DataPoint &pt){
            //cv::circle(mat, Vec2(pt.x, pt.y), 5, Green, -1);
            ++objects;
            return true;
        });
        push_timing("_root->apply", timer.elapsed());
        assert(objects == objects_grid);
    }
    
    {
        Timer timer;
        static std::vector<Node::Ptr> q;
        q.clear();
        
        q.push_back(_root);
        
        while(!q.empty()) {
            auto ptr = q.back();
            q.pop_back();
            
            for(auto& r : ((Region*)ptr)->regions()) {
                if(r && !r->is_leaf())
                    q.push_back(r);
                else if(r) {
                    for(auto &d : ((Leaf*)r)->data())
                        ++objects_loop;
                }
            }
        }
        
        push_timing("loop_apply", timer.elapsed());
        assert(objects == objects_loop);
    }
    
    {
        Timer timer;
        std::queue<Node::Ptr> q;
        q.push(_root);
        
        while(!q.empty()) {
            auto ptr = q.front();
            q.pop();
            
            for(auto& r : ((Region*)ptr)->regions()) {
                if(r && !r->is_leaf())
                    q.push(r);
                else if(r) {
                    for(auto &d : ((Leaf*)r)->data())
                        ++objects_queue;
                }
            }
        }
        
        push_timing("queue_apply", timer.elapsed());
        assert(objects == objects_loop && objects == objects_queue);
    }
    
    if (draw) {
        size_t i = 0;
        for (auto& pt : data) {
            Color clr = White;

            if (pt._d == Direction::TL) clr = Red;
            if (pt._d == Direction::TR) clr = Blue;
            if (pt._d == Direction::BL) clr = Green;
            if (pt._d == Direction::BR) clr = Yellow;

            clr.r = uint32_t(i * 0.1) % 255;
            clr.g = uint32_t(i * 0.01) % 255;
            clr.b = uint32_t(i * 0.001) % 255;

            cv::circle(mat, Vec2(pt.x, pt.y), 5, White, -1);
            ++i;
        }

        std::queue<Node::Ptr> q;
        q.push(_root);

        ColorWheel wheel;

        while (!q.empty()) {
            auto ptr = q.front();
            q.pop();

            //if(((Region*)ptr)->pixel_size() < uniform_grid_cell_size)
            cv::rectangle(mat, Vec2(ptr->x().start, ptr->y().start), Vec2(ptr->x().end, ptr->y().end), wheel.next());

            for (auto& r : ((Region*)ptr)->regions()) {
                if (r && !r->is_leaf() && ((Region*)r)->pixel_size() > 0)
                    q.push(r);
            }
        }

        cv::rectangle(mat, Vec2(1), Vec2(Tracker::average().cols - 1, Tracker::average().rows - 1), White);

        static gpuMat gpu;
        mat.copyTo(gpu);
        cv::cvtColor(gpu, gpu, cv::COLOR_BGR2RGB);

        Bounds bounds((Vec2()), Size2(Tracker::average()));
        //bounds.combine(Bounds(Vec2(), Size2(N * stride)));

        static gpuMat smaller;
        double ratio = bounds.height / bounds.width;

        cv::resize(gpu(bounds), smaller, cv::Size(2048, 2048 * ratio), 0, 0, cv::INTER_LANCZOS4);

        std::vector<std::tuple<Vec2, std::string>> labels;
        auto pos = Vec2(10, 35);
        for (auto& line : utils::split(get_stats(), '\n')) {
            labels.push_back({ pos, line });
            pos += Vec2(0, 12);
        }

        pos += Vec2(750, 0);
        auto sub = smaller(Bounds(1, 1, pos.x, pos.y - 12));
        cv::multiply(sub, cv::Scalar(0.5), sub);
        //cv::rectangle(smaller, Vec2(), pos, )

        std::string header = "frames " + Meta::toStr(_root->frame_range()) + ": " + Meta::toStr(objects) + " objects, " + Meta::toStr(objects_grid_leaf) + " leafs, " + Meta::toStr(objects_grid_region) + " regions, " + Meta::toStr(objects_upper) + " objects upper left";
        cv::putText(smaller, header, Vec2(10, 15), cv::FONT_HERSHEY_PLAIN, 1, White);

        for (auto [p, l] : labels) {
            cv::putText(smaller, l, p, cv::FONT_HERSHEY_PLAIN, 1, Color(200, 200, 200, 255));
        }

#ifndef __APPLE__
        static file::Path path("C:/Users/mooch/Desktop/visualization_cells_"+SETTING(filename).value<file::Path>().filename().to_string()+"_"+SETTING(output_prefix).value<std::string>()+".avi");
#else
        static file::Path path("/Users/tristan/Desktop/visualization_cells_"+SETTING(filename).value<file::Path>().filename().to_string() + "_" + SETTING(output_prefix).value<std::string>() +".avi");
#endif
        static cv::VideoWriter writer(path.str(), cv::VideoWriter::fourcc('F','F','V','1'), FAST_SETTING(frame_rate), cv::Size(smaller.cols, smaller.rows), true);
        
        static cv::Mat to_write;
        smaller.copyTo(to_write);
        writer.write(to_write);
        
        if(index % 20 == 0) {
            //cv::imshow("image", smaller);
            //cv::waitKey(1);
            tf::imshow("image", smaller);
        }
    }
    
    //
    //writer.write(mat);
    return replaced;
#endif
}

size_t Grid::size() const {
    return _elements;
}

void Grid::clear() {
    static Timer timer;
    timer.reset();
    
    if(_root)
        _root->clear();
    _elements = 0;
    _identities.clear();
    _identity_aliases.clear();
    
    push_timing("Grid::clear", timer.elapsed());
}

void Region::clear() {
    Node::clear();
    
    for(auto &child : _regions) {
        if(child) {
            child->clear();
            
            if(child->is_leaf())
                push_leaf(child);
            else
                push_region(child);
            
            child = nullptr;
        }
    }
}

void Grid::collect_cells(uint32_t grid_size, std::vector<Region *> &output) const {
    output.clear();
    
    const uint32_t cell_size = next_pow2<uint32_t>(grid_size);
    std::queue<Node::Ptr> q;
    q.push(_root);
    
    while (!q.empty()) {
        auto ptr = q.front();
        q.pop();
        
        for(auto& r : ((Region*)ptr)->regions()) {
            if(r && !r->is_leaf()) {
                if(((Region*)r)->pixel_size() > cell_size)
                    q.push(r);
                else
                    output.push_back((Region*)r);
            }
        }
    }
}

size_t Grid::keep_only(const Range<Frame_t> &frames) {
    static Timer timer;
    timer.reset();
    
    //auto removed = erase(Rangel(max(0, frames.end), std::numeric_limits<long_t>::max()));
    //removed += erase(Rangel(0, max(0, frames.start)));
    auto removed = _root->keep_only(frames);
    _elements -= removed;
    
    push_timing("Grid::keep_only", timer.elapsed());
    return removed;
    
    /*size_t removed = 0;
    std::vector<Region*> q;
    //std::vector<const Leaf*> leafs;
    q.push_back(_root);
    
    while(!q.empty()) {
        auto ptr = q.back();
        q.pop_back();
        
        for(auto& r : ptr->regions()) {
            if(r && !r->is_leaf()) {
                if(overlaps(r->frame_range(), frames)) {
                    if(((Region*)r)->pixel_size() <= 2) {
                        removed += r->erase(Rangel(frames.end, std::numeric_limits<long_t>::max()));
                        if(frames.start > 0)
                            removed += r->erase(Rangel(0, frames.start));
                    } else
                        q.push_back((Region*)r);
                }
                
            } else if(r) {
                throw U_EXCEPTION("Shouldnt happen.");
            }
        }
    }
    
    push_timing("Grid::keep_only", timer.elapsed());
    return true;*/
}

std::vector<Leaf*> Grid::collect_leafs(uint32_t uniform_grid_cell_size) const {
    const uint32_t stride = uniform_grid_cell_size;
    const uint32_t N = double(_root->pixel_size() + 0.5) / double(stride);
    
    
    std::queue<Node::Ptr> q;
    q.push(_root);
    
    std::vector<Leaf*> leafs;
    leafs.resize(SQR(N));
    
    while (!q.empty()) {
        auto ptr = q.front();
        q.pop();
        
        for(auto& r : ((Region*)ptr)->regions()) {
            if(r && !r->is_leaf())
                q.push(r);
            else if(r)
                leafs.push_back((Leaf*)r);
        }
    }
    
    return leafs;
}

void Region::insert(std::vector<DataPoint> &data) {
    insert(data.begin(), data.end());
}

void Region::insert(std::vector<DataPoint>::iterator start, std::vector<DataPoint>::iterator end) {
    assert(_pixel_size > 1);
    
    _size += std::distance(start, end);
    
    uint32_t next_pixel_size = _pixel_size / 2;
    
    const Range<uint32_t> LEFT(_x.start, _x.start + next_pixel_size);
    const Range<uint32_t> RIGHT(LEFT.end, _x.end);
    const Range<uint32_t> TOP(_y.start, _y.start + next_pixel_size);
    const Range<uint32_t> BOTTOM(TOP.end, _y.end);
    
    const std::array<Range<uint32_t>, 4 * 2> ranges { \
        LEFT, TOP, \
        RIGHT,TOP, \
        RIGHT, BOTTOM, \
        LEFT, BOTTOM
    };
    
    for(auto it = start; it != end; ++it) {
        auto &pt = *it;
        
        assert(pt.x >= _x.start && pt.x < _x.end);
        assert(pt.y >= _y.start && pt.y < _y.end);
        
        if(LEFT.contains(pt.x)) {
            if(!TOP.contains(pt.y)) pt._d = Direction::BL;
            else pt._d = Direction::TL;
            
        } else {
            if(TOP.contains(pt.y))  pt._d = Direction::TR;
            else pt._d = Direction::BR;
        }
    }
    
    std::sort(start, end, [](const DataPoint& A, const DataPoint& B) -> bool {
        return A._d < B._d || (A._d == B._d && (A.y < B.y || (A.y == B.y && A.x < B.x)));
    });
    
    auto previous = start != end ? start->_d : (Direction)0;
    auto last_section = start;
    
    std::array<decltype(start), 4 * 2> sections {
        end, end,
        end, end,
        end, end,
        end, end
    };
    
    auto it = start;
    //size_t items = std::distance(start, end);
    //size_t counted = 0;
    
    for(; it != end; ++it) {
        if(it->_d != previous) {
            sections[size_t(previous) * 2] = last_section;
            sections[size_t(previous) * 2 + 1] = it;
            
            //counted += std::distance(last_section, it);
            previous = it->_d;
            last_section = it;
        }
    }
    
    if(std::distance(last_section, it) != 0) {
        sections[size_t(previous) * 2] = last_section;
        sections[size_t(previous) * 2 + 1] = end;
        //counted += std::distance(last_section, it);
    }
    
    for(size_t i=0; i<4; ++i) {
        if(std::distance(sections[i * 2], sections[i * 2 + 1]) != 0) {
            auto& region = _regions[i];
            
            if(next_pixel_size <= 1) {
                if(!region) {
                    region = retrieve_leaf();
                    ((Leaf*)region)->init(_grid, this,
                        ranges[size_t(i) * 2],
                        ranges[size_t(i) * 2 + 1]);
                }
                
                region->insert(sections[i * 2], sections[i * 2 + 1]);
                
            } else {
                if(!region) {
                    region = retrieve_region();
                    ((Region*)region)->init(_grid, this, ranges[size_t(i) * 2], ranges[size_t(i) * 2 + 1], next_pixel_size);
                }
                
                region->insert(sections[i * 2], sections[i * 2 + 1]);
            }
        }
    }
    
    update_ranges();
}

void Region::update_ranges() {
    _value_sum = 0;
    _value_sqsum = 0;
    _value_range.start = _value_range.end = infinity<double>();
    
    /*if(_IDs.size() != _grid->identities().size()) {
        _IDs.resize(_grid->identities().size(), 0);
        _values_per_id.resize(_IDs.size(), 0);
        _value_range_per_id.resize(_IDs.size(), Range<double>(infinity<double>(), infinity<double>()));
    }
    
    std::fill(_IDs.begin(), _IDs.end(), 0);
    std::fill(_values_per_id.begin(), _values_per_id.end(), 0);
    std::fill(_value_range_per_id.begin(), _value_range_per_id.end(), Range<double>(infinity<double>(), infinity<double>()));*/
    
    _size = 0;
    _frame_range.start.invalidate();
    _frame_range.end.invalidate();
    
    for(auto r : regions()) {
        if(!r)
            continue;
    
        assert(!r->empty());
        
        _size += r->size();
        _value_sum += r->value_sum();
        _value_sqsum += r->value_sqsum();
        
        if(_value_range.start > r->value_range().start)
            _value_range.start = r->value_range().start;
        if(_value_range.end == infinity<double>() || _value_range.end < r->value_range().end)
            _value_range.end = r->value_range().end;
        
        /*assert(_IDs.size() >= r->IDs().size());
        for(size_t i=0; i<_IDs.size() && i < r->IDs().size(); ++i) {
            _IDs[i] += r->IDs()[i];
            _values_per_id[i] += r->values_per_id()[i];
            
            if(_value_range_per_id[i].start > r->value_range_per_id()[i].start)
                _value_range_per_id[i].start = r->value_range_per_id()[i].start;
            if(_value_range_per_id[i].end == infinity<double>() || _value_range_per_id[i].end < r->value_range_per_id()[i].end)
                _value_range_per_id[i].end = r->value_range_per_id()[i].end;
        }*/
        
        auto &range = r->frame_range();
        if(not _frame_range.start.valid() || range.start < _frame_range.start)
            _frame_range.start = range.start;
        if(not _frame_range.end.valid() || range.end > _frame_range.end)
            _frame_range.end = range.end;
    }
    
    check_range();
}

bool Region::apply(const std::function<bool(const DataPoint&)>& fn) const {
    std::vector<const Region*> q;
    //std::vector<const Leaf*> leafs;
    q.push_back(this);
    
    while(!q.empty()) {
        auto ptr = q.back();
        q.pop_back();
        
        for(auto& r : ptr->regions()) {
            if(r && !r->is_leaf())
                q.push_back((Region*)r);
            else if(r) {
                for(auto &d : ((Leaf*)r)->data()) {
                    if(!fn(d))
                        return false;
                }
            }
        }
    }
    
    return true;
}

bool Region::apply(const std::function<bool(const DataPoint&)>& fn, const Range<Frame_t>& frames, const Range<uint32_t>& xs, const Range<uint32_t>& ys) const {
    for(auto &r : _regions) {
        if(r) {
            if((!xs.empty() && !overlaps(r->x(), xs)) || (!ys.empty() && !overlaps(r->y(), ys)))
                continue;
            
            if(r->is_leaf()) {
                auto leaf = (Leaf*)r;
                for(auto &pt : leaf->data()) {
                    if(frames.contains(pt.frame))
                    if(!fn(pt))
                        return false;
                }
                
            } else if(!(((Region*)r)->apply(fn, frames, xs, ys)))
                return false;
        }
    }
    
    return true;
}

Region::Region()
    : _pixel_size(0), _size(0)
{
    std::fill(_regions.begin(), _regions.end(), nullptr);
}

Region::~Region() {
}

void Region::init(const Grid* grid, Node::Ptr parent, const Range<uint32_t>& x, const Range<uint32_t>& y, uint32_t pixel_size)
{
    _pixel_size = pixel_size;
    _size = 0;
    
    Node::init(grid, parent, x, y);
}

Leaf::Leaf()
{
}

void Leaf::insert(std::vector<DataPoint>::iterator start, std::vector<DataPoint>::iterator end)
{
#ifndef NDEBUG
    for(auto it = start; it != end; ++it)
        assert(it->x == x().start && it->y == y().start);
#endif
    _data.insert(_data.end(), start, end);
    /*if(_IDs.size() != _grid->identities().size()) {
        _IDs.resize(_grid->identities().size());
        _value_range_per_id.resize(_IDs.size(), Range<double>(infinity<double>(), infinity<double>()));
        _values_per_id.resize(_IDs.size());
    }*/
    
    for(auto it = start; it != end; ++it) {
        _value_sum += it->value;
        _value_sqsum += SQR(it->value);
        
        if(_value_range.start > it->value)
            _value_range.start = it->value;
        if(_value_range.end == infinity<double>() || _value_range.end < it->value)
            _value_range.end = it->value;
        
        /*++_IDs[it->IDindex];
        _values_per_id[it->IDindex] += it->value;
        
        if(_value_range_per_id[it->IDindex].start > it->value)
            _value_range_per_id[it->IDindex].start = it->value;
        if(_value_range_per_id[it->IDindex].end == infinity<double>() || _value_range_per_id[it->IDindex].end < it->value)
            _value_range_per_id[it->IDindex].end = it->value;*/
    }
    //_value_sum = std::accumulate(start, end, _value_sum);
    
    std::sort(_data.begin(), _data.end(), [](const DataPoint& A, const DataPoint& B) {
        return A.frame < B.frame;
    });
    
    _frame_range.start = min_frame();
    _frame_range.end = max_frame() + 1_f;
}

Node::Node() : _frame_range({}, {})
{}

}
}
