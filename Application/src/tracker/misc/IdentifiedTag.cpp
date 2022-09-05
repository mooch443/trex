#include "IdentifiedTag.h"
#include <misc/ProximityGrid.h>
#include <misc/ranges.h>


namespace track {
namespace tags {

inline static std::shared_mutex grid_mutex;
inline static std::atomic<uint64_t> added_entries{ 0 };

struct BidAndProbability {
    pv::bid bdx;
    float p;
    
    constexpr auto operator<=>(const BidAndProbability&) const = default;
    std::string toStr() const {
        return "BidP<bdx:"+Meta::toStr(bdx)+" p:"+Meta::toStr(p)+">";
    }
};

struct AssignmentsPerTag {
    Idx_t tag;
    mutable std::shared_mutex mutex;
    std::unordered_map<Frame_t, BidAndProbability> detections;
    
    auto operator<=>(const AssignmentsPerTag& other) const {
        return tag <=> other.tag;
    }
    
    BidAndProbability assignment_in_frame(Frame_t frame) const {
        std::shared_lock guard(mutex);
        auto it = detections.find(frame);
        if(it != detections.end()) {
            return it->second;
        }
        return BidAndProbability{};
    }
    
    void add_assignment(Frame_t frame, BidAndProbability&& assignment) {
        std::unique_lock guard(mutex);
        auto it = detections.find(frame);
        if(it != detections.end()) {
            assert(assignment.bdx.valid());
            if(it->second.bdx.valid() && it->second.bdx != assignment.bdx) {
                if(it->second.p < assignment.p) {
                    // replace because the probability is worse
                    //detections.erase(it);
#ifndef NDEBUG
                    FormatWarning("There is already a detection for tag ", tag, " in frame ", frame, ": ", it->second, ". Using new: ", assignment);
#endif
                    ++added_entries;
                    detections.insert_or_assign(it->first, std::move(assignment));
                    //std::swap(std::move(it->second), std::move(assignment));
                } else {
#ifndef NDEBUG
                    FormatWarning("There is already a detection for tag ", tag, " in frame ", frame, ": ", it->second, ". Skipping assignment to ", assignment);
#endif
                }
            } else
                it->second = std::move(assignment);
            
            return;
        }
        
        ++added_entries;
        detections.emplace(frame, std::move(assignment));
    }
    
    size_t hash() const {
        return std::hash<Idx_t>()( tag );
    }
};

}
}

namespace std {
    template <>
    struct hash<track::tags::Detection> {
        size_t operator()(const track::tags::Detection& k) const {
            return k.hash();
        }
    };

    template <>
    struct hash<track::tags::Assignment> {
        size_t operator()(const track::tags::Assignment& k) const {
            return k.hash();
        }
    };

    template <>
    struct hash<track::tags::AssignmentsPerTag> {
        size_t operator()(const track::tags::AssignmentsPerTag& k) const {
            return k.hash();
        }
    };
}

namespace track {
namespace tags {
    static constexpr int proximity_res = 100;

    bool Detection::operator==(const pv::bid& b) const {
        return this->bid == b;
    }

    bool Assignment::operator==(const pv::bid& b) const {
        return this->bid == b;
    }

    class DetectionGrid : public grid::Grid2D<Assignment, std::vector<grid::pixel<Assignment>>> {
    public:
        using result_t = std::tuple<float, Assignment>;

        DetectionGrid();

        UnorderedVectorSet<result_t> query(const Vec2& pos, float max_d) const;
        //std::string str(fdx_pos dx, Vec2 point, float max_d) const;

    private:
        virtual Assignment query(float, float) const override { return Assignment{}; }
    };

    DetectionGrid::DetectionGrid()
        : Grid2D(Size2(32, 32), proximity_res) {}

    UnorderedVectorSet<DetectionGrid::result_t> DetectionGrid::query(const Vec2& point, float max_d) const {
        std::vector<result_t> result;
        std::vector<Assignment> found;

        //! assume squared grid
        assert(_scale.x == _scale.y);

        //! this is the number of grid cells in x/y direction that have to be walked until max_d is reached anyway
        // +1 because we are somewhere in a cell, so just be sure (but we dont need to add here because we use <=)
        int max_cells = ceilf(max_d / _scale.max());

        max_d *= max_d;

        //! this is the grid cell in the center, go in each direction from this cell
        int cdx = floorf(max(0, point.x) / _scale.x);
        int cdy = floorf(max(0, point.y) / _scale.y);

        Rangei cells_x(min(proximity_res - 1, max(0, cdx - max_cells)), min(proximity_res - 1, max(0, cdx + max_cells)));
        Rangei cells_y(min(proximity_res - 1, max(0, cdy - max_cells)), min(proximity_res - 1, max(0, cdy + max_cells)));

        decltype(grid)::const_iterator start = grid.begin(), end = grid.end();
        decltype(grid)::const_iterator git = start + (cells_x.start + cells_y.start * proximity_res);
        const size_t stride = (proximity_res - cells_x.end + cells_x.start - 1);

        for(int y = cells_y.start; y <= cells_y.end; ++y) {
            assert(git == start + (cells_x.start + y * proximity_res));
            assert(git < grid.end());

            for(int x = cells_x.start; x <= cells_x.end; ++x, ++git) {
                assert(git < grid.end());

                for(auto&& [x, y, v] : *git) {
                    auto d = sqdistance(Vec2(x, y), point);
                    if(d < max_d) {
                        if(contains(found, v)) {
                            decltype(result)::iterator it = result.begin();
                            for(; it != result.end(); ++it) {
                                if(std::get<1>(*it) == v)
                                    break;
                            }

                            // should be true
                            assert(it != result.end());

                            if(std::get<0>(*it) > d) {
                                result.erase(it);
                                result.push_back({ d, v });
                            }

                            continue;
                        }

                        found.push_back(v);
                        result.push_back({ d, v });
                    }
                }
            }

            // go to the next row inside the grid array
            if(git < end - stride) {
                git += stride;
            } else {
                assert(y >= cells_y.end);
            }
        }

        for(auto& [d, v] : result) {
            d = sqrtf(d);
            //return_v.insert({sqrtf(d), v});
        }

        return UnorderedVectorSet(std::move(result));
    }

    inline static std::unordered_map<Idx_t, AssignmentsPerTag> assignments;
    inline static std::unordered_map<Frame_t, DetectionGrid> grid;

    void detected(Frame_t frame, Detection&& tag) {
        BidAndProbability translate{.bdx=tag.bid, .p=tag.p};

        std::unique_lock guard(grid_mutex);
        auto it = assignments.find(tag.id);
        if(!tag.id.valid())
            FormatWarning("Tag id ", tag.id, " is not valid for blob ", tag.bid, " at ", tag.pos);
        if(it != assignments.end())
            it->second.add_assignment(frame, std::move(translate));
        else {
            auto &atg = assignments[tag.id];
            atg.tag = tag.id;
            atg.add_assignment(frame, std::move(translate));
        }
        //grid[frame].insert(tag.pos.x, tag.pos.y, Assignment{std::move(tag)});
    }

    /*UnorderedVectorSet<std::tuple<float, Assignment>> query(Frame_t f, const Vec2& p, float d) {
        std::shared_lock guard(grid_mutex);
        return grid[f].query(p, d);
    }*/

    void remove(Frame_t /*frame*/, pv::bid /*bid*/) {
        std::unique_lock guard(grid_mutex);
        throw U_EXCEPTION("Remove not implemented");
        //grid[frame].erase(Assignment{.bid = bid});
    }

    void write(Data& data) {
        std::shared_lock guard(grid_mutex);
        print("Writing ", assignments.size(), " tags to file...");

        uint64_t counter = sizeof(uint32_t);
        for(const auto &[id, tag] : assignments) {
            counter += 2 * sizeof(uint32_t)
                + tag.detections.size() * (sizeof(uint32_t) * 2 + sizeof(float));
        }
        
        print("tags take up ", FileSize{counter});
        
        DataPackage package(counter);
        package.write<uint32_t>(assignments.size());
        
        for(const auto &[id, tag] : assignments) {
            package.write<uint32_t>(id._identity);
            package.write<uint32_t>(tag.detections.size());
            for(auto &[frame, pair] : tag.detections) {
                package.write<uint32_t>(frame.get());
                package.write<uint32_t>(pair.bdx._id);
                package.write<float>(pair.p);
            }
        }
        
        data.write(package);
    }

    void read(Data& data) {
        std::unique_lock guard(grid_mutex);
        uint32_t N;
        uint32_t identity;
        BidAndProbability pair;
        uint32_t frame;
        
        data.read<uint32_t>(N);
        assignments.clear();
        added_entries = 0;
        
        for (uint32_t i=0; i<N; ++i) {
            data.read<uint32_t>(identity);
            
            uint32_t Na;
            data.read<uint32_t>(Na);
            added_entries += Na;
            auto &assign = assignments[Idx_t(identity)];
            
            for (uint32_t j=0; j<Na; ++j) {
                data.read<uint32_t>(frame);
                data.read<uint32_t>(pair.bdx._id);
                data.read<float>(pair.p);
                assign.detections[Frame_t(frame)] = std::move(pair);
            }
        }
        
        if(N>0)
            print("Read ", N, " tags.");
    }

    Assignment find(Frame_t frame, pv::bid bdx) {
        std::shared_lock guard(grid_mutex);
        for(const auto &[id, tag] : assignments) {
            auto a = tag.assignment_in_frame(frame);
            if(a.bdx.valid() && bdx == a.bdx) {
                return Assignment{id, bdx, a.p};
            }
        }
        
        return Assignment{};
        
        /*std::shared_lock guard(grid_mutex);
        auto it = grid.find(frame);
        if (it == grid.end())
            return {};

#if !defined(_MSC_VER) || _MSC_VER >= 1930
        return it->second.find(bdx);
#else
        // Old MSVC versions (e.g. 2019) will not compile
        // the above code, unfortunately. So here we go...
        for (const auto& set : it->second.get_grid()) {
            auto it = std::find(set.begin(), set.end(), bdx);
            if (it != set.end())
                return it->v;
        }

        return Assignment{};
#endif*/
    }

    bool available() {
        return added_entries.load() > 0;
    }

}
}
