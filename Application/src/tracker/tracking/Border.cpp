#include "Border.h"
#include <misc/CircularGraph.h>
#include <tracking/Tracker.h>
#include <misc/PixelTree.h>
#include <misc/Timer.h>
#include <misc/default_config.h>
#include <gui/GuiTypes.h>

namespace track {
    Border::Border()
    : _type(Type::none), _recognition_border_size_rescale(-1)
    {}
    
    periodic::points_t smooth_outline(const periodic::points_t& points, float range, long step) {
        const long L = points->size();
        
        if(L > range) {
            periodic::points_t smoothed = std::make_shared<std::vector<Vec2>>();
            smoothed->reserve(L);
            
            const float step_row = range * step;
            
            std::vector<float> weights;
            weights.reserve(L);
            {
                float sum = 0;
                for (int i=-step_row; i<=step_row; i+=step) {
                    float val = (step_row-cmn::abs(i))/step_row;
                    sum += val;
                    
                    weights.push_back(val);
                }
                for (auto &v : weights) {
                    v /= sum;
                }
            }
            
            Vec2 pt;
            for (long i=0; i<L; i++) {
                long samples = 0;
                pt.x = pt.y = 0;
                
                for (long j=i-step_row; j<=i+step_row; j+=step)
                {
                    auto idx = j;
                    while (idx < 0)
                        idx += L;
                    while (idx >= L)
                        idx -= L;
                    
                    pt += points->operator[](idx) * weights[samples++];
                }
                
                if(samples)
                    smoothed->push_back(pt);
            }
            
            return smoothed;
        }
        
        return nullptr;
    }
    
    /*bool pnpoly(const std::vector<Vec2> pts, const Vec2& pt)
    {
        int npol = (int)pts.size();
        int i, j;
        bool c = false;
        for (i = 0, j = npol-1; i < npol; j = i++) {
            if ((((pts[i].y <= pt.y) && (pt.y < pts[j].y)) ||
                 ((pts[j].y <= pt.y) && (pt.y < pts[i].y))) &&
                (pt.x < (pts[j].x - pts[i].x) * (pt.y - pts[i].y) / (pts[j].y - pts[i].y) + pts[i].x))
                c = !c;
        }
        return c;
    }*/
    
    void Border::clear() {
        //LockGuard guard;
        std::lock_guard<std::mutex> guard(mutex);
        _vertices.clear();
        //x_range.clear();
        //y_range.clear();
        _mask = nullptr;
    }
    
    void Border::update_heatmap(pv::File& video) {
        if(_mask)
            return; // already generated a mask
        
        constexpr size_t grid_res = 100;
        const Size2 grid_size = Tracker::average().bounds().size() / float(grid_res);
        auto pos2grid = [&grid_size](const Vec2& pos) {
            return pos.div(grid_size).map<round>();
        };
        
        auto value = [this](const Vec2& pos) -> uint32_t {
            auto it = grid_cells.find({uint16_t(pos.x), uint16_t(pos.y)});
            return it == grid_cells.end() ? 0 : it->second;
        };
        
        if(grid_cells.empty() || _recognition_border_size_rescale != SETTING(recognition_border_size_rescale).value<float>()) {
            _recognition_border_size_rescale = SETTING(recognition_border_size_rescale).value<float>();
            grid_cells.clear();
            
            auto access = [this](const Vec2& pos) -> uint32_t& {
                return grid_cells[{uint16_t(pos.x), uint16_t(pos.y)}];
            };
            
            const float sqcm = SQR(FAST_SETTING(cm_per_pixel));
            const float rescale = 1 - min(0.9, max(0, SETTING(recognition_border_size_rescale).value<float>()));
            
            print("Reading video...");
            pv::Frame frame;
            size_t step_size = max(1, video.length() * 0.0002);
            const size_t count_steps = video.length() / step_size;
            for (size_t i=0; i<video.length(); i+=step_size) {
                video.read_frame(frame, i);
                auto blobs = frame.get_blobs();
                
                if(SETTING(terminate))
                    break;
                
                for(auto b : blobs) {
                    auto pb = pixel::threshold_blob(b, FAST_SETTING(track_threshold), Tracker::instance()->background());
                    
                    for(auto b : pb) {
                        auto size = b->num_pixels() * sqcm;
                        if(FAST_SETTING(blob_size_ranges).in_range_of_one(size, rescale)) {  //size >= min_size && size <= max_size) {
                            for(auto &line : b->hor_lines()) {
                                for(ushort x=line.x0; x<=line.x1; ++x) {
                                    auto pos = Vec2(x, line.y);//b->bounds().pos() + b->bounds().size() * 0.5;
                                    auto grid = pos2grid(pos);
                                    
                                    ++access(grid);
                                }
                            }
                        }
                    }
                }
                
                if((i / step_size) % size_t(count_steps * 0.1) == 0)
                    print("[border] ", i/step_size," / ",count_steps);
            }
            
            print("Done.");
        }
        
        std::multiset<uint32_t> counts;
        for(auto && [coord, count] : grid_cells) {
            if(count)
                counts.insert(count);
        }
        
        auto it = counts.begin();
        std::advance(it, counts.size() * 0.05);
        auto middle = counts.size() == 0 ? 0 : *it;
        
        _mask = Image::Make(video.size().height, video.size().width);
        for(ushort x = 0; x < _mask->cols; ++x) {
            for (ushort y = 0; y < _mask->rows; ++y) {
                auto p = pos2grid(Vec2(x, y));
                _mask->data()[y * _mask->cols + x] = value(p) >= middle ? 255 : 0;
            }
        }
        
        cv::Mat out;
        const size_t morph_size = max(1, video.size().width * 0.025), morph_size1 = max(1, morph_size * (1 - SETTING(recognition_border_shrink_percent).value<float>()));
        static const cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, (cv::Size)Size2( 2*morph_size + 1, 2*morph_size+1 ), Vec2( morph_size, morph_size ) );
        static const cv::Mat element1 = cv::getStructuringElement( cv::MORPH_ELLIPSE, (cv::Size)Size2( 2*morph_size1 + 1, 2*morph_size1+1 ), Vec2( morph_size1, morph_size1 ) );
        
        cv::blur(_mask->get(), out, (cv::Size)Size2(size_t(video.size().width * 0.07) | 1,size_t(video.size().height * 0.07) | 1));
        cv::threshold(out, out, 150, 255, cv::THRESH_BINARY);
        
#ifndef NDEBUG
        tf::imshow("threshold", out);
#endif
        
        cv::erode(out, out, element);
        cv::dilate(out, out, element);
        cv::erode(out, out, element1);
      
#ifndef NDEBUG
        tf::imshow("out", out);
        tf::imshow("in", _mask->get());
#endif
        
        _mask = Image::Make(out);
    }
    
    void Border::update_outline(pv::File &video) {
        _type = Type::outline;
        
        if(_vertices.empty()) {
            Timer timer;
            print("Generating outline...");
            
            if((size_t)video.size().height > (size_t)USHRT_MAX)
                throw U_EXCEPTION("Video is too big (max: ",USHRT_MAX,"x",USHRT_MAX,")");
            
            if(x_valid.empty() && y_valid.empty()) {
                // generate one line for every row
                x_range.clear();
                y_range.clear();
                
                x_range.resize(video.size().height);
                y_range.resize(video.size().width);
                
                const float sqcm = SQR(FAST_SETTING(cm_per_pixel));
                
                std::vector<pv::BlobPtr> collection;
                pv::Frame frame;
                for (size_t i=0; i<video.length(); i+=max(1, video.length() * 0.0005)) {
                    video.read_frame(frame, i);
                    auto blobs = frame.get_blobs();
                    
                    for(auto b : blobs) {
                        auto pb = pixel::threshold_blob(b, FAST_SETTING(track_threshold), Tracker::instance()->background());
                        
                        for(auto b : pb) {
                            auto size = b->num_pixels() * sqcm;
                            if(FAST_SETTING(blob_size_ranges) .in_range_of_one(size, 0.5) ) //size >= min_size && size <= max_size)
                                collection.push_back(b);
                        }
                    }
                }
                
                print("Collected ", collection.size()," blobs between sizes in ",FAST_SETTING(blob_size_ranges)," with scale 0.5");
                
                std::vector<std::multiset<ushort>> xs;
                std::vector<std::multiset<ushort>> ys;
                
                x_valid.clear();
                y_valid.clear();
                xs.resize(video.size().height);
                ys.resize(video.size().width);
                x_valid.resize(ys.size());
                y_valid.resize(xs.size());
                
                for(auto &x : x_range)
                    x = Rangef(0, video.size().width-1);
                for(auto &y : y_range)
                    y = Rangef(0, video.size().height-1);
                
                Vec2 average_height;
                float height_samples = 0;
                for(auto &b : collection) {
                    //Vec2 center = b->bounds().pos() + b->bounds().size() * 0.5;
                    average_height += b->bounds().size() * 0.5;
                    ++height_samples;
                    
                    for(auto &k : b->hor_lines()) {
                        if(k.x0 < ys.size())
                            ys.at(k.x0).insert(k.y);
                        if(k.x1 < ys.size())
                            ys.at(k.x1).insert(k.y);
                        
                        if(k.y < xs.size()) {
                            xs.at(k.y).insert(k.x0);
                            xs.at(k.y).insert(k.x1);
                        }
                    }
                }
                
                std::multiset<float> max_y, max_x;
                
                for(ushort y = 0; y<xs.size(); ++y) {
                    if(!xs[y].empty()) {
                        y_valid.at(y) = true;
                        
                        auto it = xs.at(y).begin();
                        std::advance(it, xs[y].size() * 0.03);
                        
                        auto rit = xs[y].rbegin();
                        std::advance(rit, xs[y].size() * 0.03);
                        
                        x_range.at(y).start = *it;
                        x_range.at(y).end = *rit;
                        
                        max_x.insert(*it);
                        max_x.insert(*rit);
                        
                    } else if(y > 0 && y_valid[y-1]) {
                        y_valid[y] = true;
                        
                        x_range[y].start = x_range[y-1].start;
                        x_range[y].end = x_range[y-1].end;
                    }
                }
                
                for(ushort x = 0; x<ys.size(); ++x) {
                    if(!ys[x].empty()) {
                        x_valid.at(x) = true;
                        
                        auto it = ys.at(x).begin();
                        std::advance(it, ys[x].size() * 0.03);
                        
                        auto rit = ys[x].rbegin();
                        std::advance(rit, ys[x].size() * 0.03);
                        
                        y_range.at(x).start = *it;
                        y_range.at(x).end = *rit;
                        
                        max_y.insert(*it);
                        max_y.insert(*rit);
                        
                    } else if(x > 0 && x_valid[x-1]) {
                        x_valid[x] = true;
                        
                        y_range[x].start = y_range[x-1].start;
                        y_range[x].end = y_range[x-1].end;
                    }
                }
                
                if(!max_y.empty()) {
                    auto it = max_y.begin();
                    std::advance(it, max_y.size() * 0.02);
                    
                    auto rit = max_y.rbegin();
                    std::advance(rit, max_y.size() * 0.02);
                    assert(*it < y_valid.size());
                    
                    print("Invalidate y from ", *it," to ",*rit);
                    
                    for(ushort y=0; y<min(*it, *rit, y_valid.size()); ++y) {
                        y_valid[y] = false;
                    }
                    
                    for(ushort y=*rit; y<y_valid.size(); ++y) {
                        y_valid[y] = false;
                    }
                }
                
                if(!max_x.empty()) {
                    auto it = max_x.begin();
                    std::advance(it, max_x.size() * 0.02);
                    
                    auto rit = max_x.rbegin();
                    std::advance(rit, max_x.size() * 0.02);
                    assert(*rit < x_valid.size());
                    
                    print("Invalidate x from ", *it," to ",*rit);
                    
                    for(ushort x=0; x<min(*it, *rit, x_valid.size()); ++x) {
                        x_valid[x] = false;
                    }
                    
                    for(ushort x=*rit; x<x_valid.size(); ++x) {
                        x_valid[x] = false;
                    }
                }
            }
            
            _vertices.resize(1);
            
            for(ushort x = 0; x<y_range.size(); ++x) {
                if(!x_valid[x])
                    continue;
                
                Vec2 pt(x, y_range.at(x).start);
                _vertices.front().push_back(pt);
            }
            
            for(ushort y = 0; y<x_range.size(); ++y) {
                if(!y_valid[y])
                    continue;
                
                Vec2 pt(x_range.at(y).end, y);
                _vertices.front().push_back(pt);
            }
            
            for(ushort x = y_range.size()-1; x>0; --x) {
                if(!x_valid[x])
                    continue;
                
                Vec2 pt(x, y_range.at(x).end);
                _vertices.front().push_back(pt);
            }
            
            for(ushort y = x_range.size() - 1; y>0; --y) {
                if(!y_valid[y])
                    continue;
                
                Vec2 pt(x_range.at(y).start, y);
                _vertices.front().push_back(pt);
            }
            
            uint16_t coeff = SETTING(recognition_coeff);
            if(coeff > 0) {
                auto ptr = std::make_shared<std::vector<Vec2>>(_vertices.front());
                ptr = smooth_outline(ptr, SETTING(recognition_smooth_amount).value<uint16_t>(), 1);
                
                Vec2 middle;
                float samples = 0;
                for(auto &pt : _vertices.front()) {
                    middle += pt;
                    ++samples;
                }
                if(samples)
                    middle /= samples;
                
                //auto ptr = std::make_shared<std::vector<Vec2>>(_vertices);
                auto && [cw, p] = periodic::differentiate_and_test_clockwise(ptr);
                if(cw < 0) {
                    std::reverse(ptr->begin(), ptr->end());
                }
                auto c = periodic::eft(ptr, coeff);
                ptr = periodic::ieft(c, coeff, /*_vertices.size() * 0.01*/ min(coeff * 2.0, 50), Vec2(), false).back();
                
                if(ptr) {
                    _vertices.front() = *ptr;
                    
                    for(auto & pt : _vertices.front()) {
                        pt.x = (pt.x + middle.x); //* 0.95;
                        pt.y = (pt.y + middle.y); //* 0.95;
                    }
                    
                } else
                    _vertices.clear();
                
                print("Generating mask...");
                Timer timer;
                _mask = Image::Make(video.size().height, video.size().width);
                if(!_vertices.empty()) {
                    for(ushort x = 0; x < _mask->cols; ++x) {
                        for (ushort y = 0; y < _mask->rows; ++y) {
                            _mask->data()[y * _mask->cols + x] = pnpoly(_vertices.front(), Vec2(x,y)) ? 255 : 0;
                        }
                    }
                    
                } else
                    memset(_mask->data(), 0, _mask->size());
                
                //tf::imshow("mask", _mask->get());
                
                auto sec = timer.elapsed() / _mask->size() * 1000 * 1000;
                auto str = Meta::toStr(DurationUS{uint64_t(sec)});
                print("Mask took ",str,"/pixel");
                poly_set = true;
                
            } else
                poly_set = false;
            
            print("This took ", DurationUS{uint64_t(timer.elapsed() * 1000 * 1000)}," (",_vertices.size()," points)");
        }
    }
    
    void Border::update(pv::File& video) {
        using namespace default_config;
        _type = SETTING(recognition_border).value<recognition_border_t::Class>();
        
        //LockGuard guard;
        std::lock_guard<std::mutex> guard(mutex);
        
        switch(_type) {
            case Type::none:
                break;
                
            case Type::grid: {
                auto grid_points = SETTING(grid_points).value<Settings::grid_points_t>();
                if(grid_points.size() < 2) {
                    FormatError("Cannot calculate average intra-grid for just one grid point.");
                }
                
                // if this is a grid, we need to calculate the distances between grid centers.
                // once thats calculated, the distance of a point to a grid cell center can be
                // used as a warning of border effects for recognition.
                long samples = 0;
                _max_distance = 0;
                
                for (size_t i=0; i<grid_points.size(); ++i) {
                    float min_distance = FLT_MAX;
                    
                    for (size_t j=0; j<grid_points.size(); ++j) {
                        if(i == j)
                            continue;
                        float d = euclidean_distance(grid_points[j], grid_points[i]);
                        if(d < min_distance)
                            min_distance = d;
                    }
                    
                    if(min_distance != FLT_MAX) {
                        _max_distance += min_distance;
                        samples++;
                    }
                }
                
                _max_distance = _max_distance / (float)samples * 0.5 * SETTING(grid_points_scaling).value<float>();
                
                break;
            }
                
            case Type::shapes:
                _vertices = FAST_SETTING(recognition_shapes);
                _min_distance = 1;
                break;
            case Type::outline:
                update_outline(video);
                break;
            case Type::heatmap:
                update_heatmap(video);
                break;
                
            case Type::circle:
                // probably circle mask
                _max_distance = euclidean_distance(Vec2(0, Tracker::average().rows * 0.5),
                                                   Vec2(Tracker::average().bounds().size()) * 0.5) * 0.95;
                break;
                
            default:
                print("Unknown border type ",_type);
        }
        
        update_polygons();
    }
    
    float Border::distance(const Vec2& pt) const {
        assert(_type != Type::none);
        
        if(_type == Type::grid) {
            float min_d = FLT_MAX;
            
            for (auto &grid_pt : FAST_SETTING(grid_points)) {
                float d = euclidean_distance(pt, grid_pt);
                if(d < min_d)
                    min_d = d;
            }
            
            return min_d;
            
        } else if(_type == Type::shapes) {
            //Bounds r(Tracker::average());
            auto r = FAST_SETTING(recognition_shapes);
            for(auto &shape : r) {
                if(pnpoly(shape, pt))
                    return 1;
            }
            return 0;
            //float d0 = min(abs(r.x - pt.x), abs(r.y - pt.y));
            //float d1 = min(abs(r.width - pt.x), abs(r.height - pt.y));
            //return min(d0, d1);
            
        } else if(_type == Type::circle) {
            return euclidean_distance(pt, Vec2(Tracker::average().bounds().size() * 0.5));
        }
        
        throw U_EXCEPTION("Unknown border type (",_type,").");
    }
    
    bool Border::in_recognition_bounds(const cmn::Vec2 &pt) const {
        //std::lock_guard<std::mutex> guard(mutex);
        
        if(_type == Type::none)
            return true;
        if(_type == Type::shapes) {
            if(_polygons.empty())
                return true;
            
            for(auto &shape : _polygons) {
                auto ptr = (const gui::Polygon*)shape.get();
                if(pnpoly(*ptr->vertices(), pt))
                    return true;
            }
            
            return false;
        }
        if(_type == Type::outline) {
            /*if(pt.y < _lines.size()) {
             return _lines.at(floor(pt.y)).inside(pt.x, pt.y);
             }*/
            if(poly_set && !_mask)
                return pnpoly(_vertices.front(), pt);
            else if(poly_set && _mask)
                return (size_t)pt.x < _mask->cols && (size_t)pt.y < _mask->rows && _mask->data()[(size_t)pt.x + (size_t)pt.y * _mask->cols] == 255;
            
            if((size_t)pt.y < x_range.size() && (size_t)pt.x < y_range.size()) {
                return x_range.at(pt.y).contains(pt.x) && y_range.at(pt.x).contains(pt.y);
            }
            
            return false;
            //
        }
        if(_type == Type::heatmap) {
            if(_mask)
                return (size_t)pt.x < _mask->cols && (size_t)pt.y < _mask->rows && _mask->data()[(size_t)pt.x + (size_t)pt.y * _mask->cols] != 0;
            else
                return false;
        }
        
        auto d = distance(pt);
        return d <= _max_distance;
    }

void Border::draw(gui::DrawStructure& graph) {
    for(auto &p : _polygons) {
        graph.wrap_object(*p);
    }
}

void Border::update_polygons() {
    _polygons.clear();
    for(auto &shape : _vertices) {
        auto convex = poly_convex_hull(&shape);
        auto ptr = std::make_shared<gui::Polygon>(convex);
        ptr->set_fill_clr(gui::Transparent);
        ptr->set_border_clr(gui::Cyan);
        _polygons.push_back((std::shared_ptr<gui::Drawable>)ptr);
        
    }
}

}
