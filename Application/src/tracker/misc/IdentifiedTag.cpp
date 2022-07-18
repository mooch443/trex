#include "IdentifiedTag.h"
#include <misc/ProximityGrid.h>
#include <misc/ranges.h>

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

	inline static std::unordered_map<Frame_t, DetectionGrid> grid;
    inline static std::mutex grid_mutex;
    inline static std::atomic<uint64_t> added_entries{ 0 };

    void detected(Frame_t frame, Detection&& tag) {
        ++added_entries;

        std::unique_lock guard(grid_mutex);
        grid[frame].insert(tag.pos.x, tag.pos.y, Assignment{std::move(tag)});
    }

    UnorderedVectorSet<std::tuple<float, Assignment>> query(Frame_t f, const Vec2& p, float d) {
        std::unique_lock guard(grid_mutex);
        return grid[f].query(p, d);
    }

    void remove(Frame_t frame, pv::bid bid) {
        std::unique_lock guard(grid_mutex);
        throw U_EXCEPTION("Remove not implemented");
        //grid[frame].erase(Assignment{.bid = bid});
    }

    Assignment find(Frame_t frame, pv::bid bdx) {
        std::unique_lock guard(grid_mutex);
        auto it = grid.find(frame);
        if (it == grid.end())
            return {};

#if !defined(_MSC_VER) || _MSC_VER >= 1930
        return it->second.find(bdx);
#else
        for (const auto& set : it->second.get_grid()) {
            auto it = std::find(set.begin(), set.end(), bdx);
            if (it != set.end())
                return it->v;
        }

        return Assignment{};
#endif
    }

    bool available() {
        return added_entries.load() > 0;
    }

}
}
