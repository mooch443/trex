#pragma once

#include <gui/types/Entangled.h>
#include <gui/types/Layout.h>
#include <misc/Timer.h>
#include <misc/OutputLibrary.h>
#include <misc/ThreadPool.h>
#include <misc/idx_t.h>

namespace track {
class Individual;
}

namespace gui {
namespace heatmap {

using grid_t = grid::Grid2D<std::tuple<double, long_t>, std::vector<grid::pixel<std::tuple<double, long_t>>>>;

enum Direction {
    TL = 0,
    TR = 1,
    BR = 2,
    BL = 3
};

struct DataPoint {
    Frame_t frame;
    uint32_t x,y;
    uint32_t ID, IDindex;
    
    using Data = double;
    Data value;
    Direction _d;
    
    std::string toStr() const;
    operator double() const { return value; }
    bool operator<(Frame_t frame) const {
        return this->frame < frame;
    }
};

class Grid;

class Node {
public:
    using Ptr = Node*;//std::shared_ptr<Node>;
    
protected:
    GETTER(Range<uint32_t>, x);
    GETTER(Range<uint32_t>, y);
    Range<Frame_t> _frame_range;
    GETTER(Ptr, parent);
    const Grid* _grid;
    GETTER(double, value_sum);
    GETTER(double, value_sqsum);
    GETTER(Range<double>, value_range);
    
    GETTER(std::vector<uint32_t>, IDs);
    GETTER(std::vector<double>, values_per_id);
    GETTER(std::vector<Range<double>>, value_range_per_id);
    
public:
    Node();
    virtual ~Node() {}
    
    virtual bool is_leaf() const = 0;
    virtual bool empty() const = 0;
    virtual size_t size() const = 0;
    virtual const Range<Frame_t>& frame_range() const;
    
    void init(const Grid* grid, Node::Ptr parent, const Range<uint32_t>& x, const Range<uint32_t>& y);
    virtual void clear();
    virtual size_t erase(const Range<Frame_t>& frames) = 0;
    virtual size_t keep_only(const Range<Frame_t>& frames) = 0;
    virtual void insert(std::vector<DataPoint>::iterator start, std::vector<DataPoint>::iterator end) = 0;
};

class Region : public Node {
public:
    using regions_t = std::array<Node::Ptr, 4>;
    
protected:
    GETTER(regions_t, regions);
    GETTER(uint32_t, pixel_size);
    size_t _size;
    
public:
    Region();
    ~Region();
    
    //! halfes the image to perform a binary search in the end
    void init(const Grid* grid, Node::Ptr parent, const Range<uint32_t>& x, const Range<uint32_t>& y, uint32_t pixel_size);
    
    //! aggregates are not leafs
    bool is_leaf() const override { return false; }
    
    bool apply(const std::function<bool(const DataPoint&)>& fn, const Range<Frame_t>& frames, const Range<uint32_t>& xs = Range<uint32_t>(0, std::numeric_limits<uint32_t>::max()), const Range<uint32_t>& ys = Range<uint32_t>(0, std::numeric_limits<uint32_t>::max())) const;
    bool apply(const std::function<bool(const DataPoint&)>& fn) const;
    
    //! insert a number of data points and automatically push them down if possible
    void insert(std::vector<DataPoint>& data);
    void insert(std::vector<DataPoint>::iterator A, std::vector<DataPoint>::iterator B) override;
    
    size_t erase(const Range<Frame_t>& frames) override;
    size_t keep_only(const Range<Frame_t>& frames) override;
    
    virtual void clear() override;
    bool empty() const override { return _size == 0; }
    size_t size() const override { return _size; }
    void check_range() const;
    
private:
    void update_ranges();
};

class Leaf : public Node {
    GETTER(std::vector<DataPoint>, data);
    
public:
    Leaf();
    
    Frame_t min_frame() const;
    Frame_t max_frame() const;
    
    void insert(std::vector<DataPoint>::iterator start, std::vector<DataPoint>::iterator end) override;
    bool is_leaf() const override { return true; }
    void clear() override;
    
    std::vector<DataPoint>::const_iterator begin() const { return _data.begin(); }
    std::vector<DataPoint>::const_iterator end() const { return _data.end(); }
    size_t size() const override { return _data.size(); }
    bool empty() const override { return _data.empty(); }
    
    size_t erase(const Range<Frame_t>& frames) override;
    size_t keep_only(const Range<Frame_t>& frames) override;
private:
    void update_ranges();
};

class Grid {
public:
    static constexpr uint32_t statistics_items = 15;
    static void print_stats(const std::string& title);
    using alias_map_t = std::map<uint32_t, uint32_t>;
    
protected:
    //GETTER(std::shared_ptr<Region>, root);
    GETTER_PTR(Region*, root)
    size_t _elements;
    GETTER(std::vector<uint32_t>, identities);
    GETTER(alias_map_t, identity_aliases);
    
public:
    Grid() : _root(nullptr), _elements(0) {}
    ~Grid();
    
    //! creates a grid using the maximum of width/height
    void create(const Size2& image_dimensions);
    
    //! fill with a collection of data points and sort them into the grid
    void fill(const std::vector<DataPoint>& data);
    
    //! remove all data points belonging to a given frame
    size_t erase(Range<Frame_t> frames);
    size_t keep_only(const Range<Frame_t>& frames);
    
    //! apply with filters
    template<typename T>
    bool apply(const std::function<bool(const T&)>& fn, const Range<Frame_t>& frames, const Range<uint32_t>& xs = Range<uint32_t>(0, std::numeric_limits<uint32_t>::max()), const Range<uint32_t>& ys = Range<uint32_t>(0, std::numeric_limits<uint32_t>::max())) const;
    
    //! apply without filters
    template<typename T>
    bool apply(const std::function<bool(const T&)>& fn) const;
    
    void collect_cells(uint32_t grid_size, std::vector<Region*>& output) const;
    std::vector<Leaf*> collect_leafs(uint32_t) const;
    void clear();
    size_t size() const;
    bool empty() const { return _root == nullptr || size() == 0; }
private:
    void prepare_data(std::vector<DataPoint>& data);
};

template<typename T>
bool Grid::apply(const std::function<bool(const T&)>& fn) const {
    static_assert(std::is_base_of<Node, T>::value || std::is_base_of<DataPoint, T>::value, "Class passed to Grid::apply is not derived from Node or DataPoint."); // could more strictly be limited to leaf/region and data points
    
    if(!_root)
        return false;
    
    std::vector<const Region*> q;
    std::vector<const T*> result;
    q.reserve(_root->size());
    q.push_back(_root);
    
    while(!q.empty()) {
        auto ptr = q.back();
        q.pop_back();
        
        for(auto r : ptr->regions()) {
            if(r && !r->empty()) {
                if constexpr(std::is_same<T, Region>::value) {
                    if(!r->is_leaf()) {
                        if(fn(*(Region*)r)) // here semantics change. if returned true, we traverse the region (and potentially its children)
                            q.push_back((Region*)r);
                    } else
                        break; // all further ones are going to be leaf nodes as well, so skip
                    
                } else {
                    if(!r->is_leaf()) {
                        q.push_back((Region*)r);
                        
                    } else if constexpr(std::is_same<T, Leaf>::value) {
                        if(!fn(*((Leaf*)r)))
                            return false;
                        
                    } else {
                        static_assert(std::is_base_of<DataPoint, T>::value, "Assuming DataPoint type, since it is neither Leaf nor Region.");
                        
                        for(auto &d : ((Leaf*)r)->data()) {
                            if(!fn(d))
                               return false;
                        }
                    }
                }
            }
        }
    }
    
    return true;
}

template<typename T>
bool Grid::apply(const std::function<bool(const T&)>& fn, const Range<Frame_t>& frames, const Range<uint32_t>& xs, const Range<uint32_t>& ys) const
{
    static_assert(std::is_base_of<Node, T>::value || std::is_base_of<DataPoint, T>::value, "Class passed to Grid::apply is not derived from Node or DataPoint."); // could more strictly be limited to leaf/region and data points
    
    if(!_root)
        return false;
    
    std::vector<const Region*> q;
    std::vector<const T*> result;
    q.reserve(_root->size());
    q.push_back(_root);
    
    while(!q.empty()) {
        auto ptr = q.back();
        q.pop_back();
        
        for(auto r : ptr->regions()) {
            if(r && !r->empty()) {
                if((!xs.empty() && !r->x().overlaps(xs))
                   || (!ys.empty() && !r->y().overlaps(ys)))
                    continue;
                if(!frames.empty() && ((Region*)r)->frame_range().end != frames.start && !((Region*)r)->frame_range().overlaps(frames))
                    continue;
                
                if constexpr(std::is_same<T, Region>::value) {
                    if(!r->is_leaf()) {
                        if(fn(*(Region*)r)) // here semantics change. if returned true, we traverse the region (and potentially its children)
                            q.push_back((Region*)r);
                    } else
                        break; // all further ones are going to be leaf nodes as well, so skip
                    
                } else {
                    if(!r->is_leaf()) {
                        q.push_back((Region*)r);
                        
                    } else if constexpr(std::is_same<T, Leaf>::value) {
                        if(!fn(*((Leaf*)r)))
                            return false;
                        
                    } else {
                        static_assert(std::is_base_of<DataPoint, T>::value, "Assuming DataPoint type, since it is neither Leaf nor Region.");
                        
                        if(frames.empty()) {
                            for(auto &d : ((Leaf*)r)->data()) {
                                if(!fn(d))
                                   return false;
                            }
                            
                        } else {
                            for(auto &d : ((Leaf*)r)->data()) {
                                if(!frames.contains(d.frame))
                                    continue;
                                
                                if(!fn(d))
                                   return false;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return true;
}

namespace normalization_t = ::default_config::heatmap_normalization_t;

/**
 * A class providing the GUI interface for the above Grid.
 */
class HeatmapController : public Entangled {
protected:
    normalization_t::Class _normalization;
    Grid _grid;
    Frame_t _frame;
    
    uint32_t uniform_grid_cell_size;
    uint32_t stride, N;
    Range<double> custom_heatmap_value_range;
    Frame_t _frame_context;
    std::vector<track::Idx_t> _ids;
    double smooth_heatmap_factor;
    
    Image::Ptr grid_image;
    std::string _original_source, _source;
    Output::Options_t _mods;
    std::shared_ptr<ExternalImage> _image;
    
    std::vector<double> _array_grid, _array_sqsum, _array_samples;
    
    gpuMat _viridis, _gpuGrid;
    std::map<track::Individual*, track::Individual::segment_map::const_iterator> _iterators;
    std::map<track::Individual*, size_t> _capacities;
    
public:
        HeatmapController();
    
    void set_frame(Frame_t frame);
    void update() override;
    void paint_heatmap();
    void save();
    void frames_deleted_from(Frame_t frame);
    
private:
    struct UpdatedStats {
        size_t added;
        size_t removed;
        Range<Frame_t> add_range, remove_range;
        
        UpdatedStats() : added(0), removed(0), add_range({}, {}), remove_range({}, {}) {}
    };
    UpdatedStats update_data(Frame_t frame);
    void sort_data_into_custom_grid();
    bool update_variables();
};

}
}
