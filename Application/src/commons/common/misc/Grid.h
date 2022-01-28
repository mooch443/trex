#pragma once

#include <misc/defines.h>
#include <misc/vec2.h>

namespace cmn {
namespace grid {
    template<typename T>
    struct pixel {
        float x, y;
        T v;
        
        pixel(float x = 0, float y = 0, T v = T())
            : x(x), y(y), v(v)
        { }
        
        bool operator<(const pixel& other) const {
            return y < other.y || (y == other.y && x < other.x);
        }
        bool operator<(const Vec2& other) const {
            return y < other.y || (y == other.y && x < other.x);
        }
        bool operator==(const pixel& other) const {
            return other.v == v && other.x == x && other.y == y;
        }
        bool operator==(const T& v) const {
            return this->v == v;
        }
    };
}
}

namespace std
{
    template <typename T>
    struct hash<cmn::grid::pixel<T>>
    {
        size_t operator()(const cmn::grid::pixel<T>& k) const
        {
            return ((hash<float>()(k.x)
                           ^ (hash<float>()(k.y) << 1)) >> 1)
                           ^ (hash<T>()(k.v) << 1);
        }
    };
}

namespace cmn {
namespace grid {

    template<typename T, typename Set_t = std::unordered_set<pixel<T>>>
    class Grid2D {
    public:
        typedef Set_t set_t;
        
    protected:
        //using pixel = std::tuple<float, float, T>;
        
        std::vector<Set_t> grid;
        GETTER_CONST(const Vec2, scale)
        GETTER_CONST(const Size2, resolution)
        GETTER(uint, n)
        GETTER(uint, N)
        
    public:
        Grid2D(const Size2& resolution, uint n)
            :
            _scale(ceilf(resolution.max() / float(n))/*ceilf(resolution.width / float(n)), ceilf(resolution.height / float(n))*/),
            _resolution(Vec2(resolution.max(), resolution.max())),
            _n(n),
            _N(n*n)
        {
            /*using set_t = typename decltype(grid)::value_type;
            for(int i=0; i<_N; ++i)
                grid.push_back(set_t([](const pixel& a, const pixel& b) -> bool{
                    return a.y < b.y || (a.y == b.y && a.x < b.x);
                }));*/
            grid.resize(_N);
             
        }
        
        virtual ~Grid2D() {}
        
        decltype(grid)& get_grid() { return grid; }
        const decltype(grid)& get_grid() const { return grid; }
        
        int64_t idx(float x, float y) const {
            return int64_t(max(0, floorf(min(_resolution.width-1,x) / _scale.x)) + max(0, floorf(min(_resolution.height-1, y) / _scale.y)) * _n);
        }
        
        std::tuple<size_t, size_t> point2grid(float x, float y) const {
            return {
                (size_t)cmn::max(0.f, floor(min(_resolution.width-1,x) / _scale.x)),
                (size_t)cmn::max(0.f, floor(min(_resolution.height-1, y) / _scale.y))
            };
        }
        
        template<typename S = Set_t>
        void insert(float x, float y, T v, typename std::enable_if<is_set<S>::value, void>::type* = nullptr) {
            const auto i = idx(x, y);
            assert(i < grid.size());
            grid[i].insert(pixel<T>(x, y, v));
        }
        
        template<typename S = Set_t>
        void insert(float x, float y, T v, typename std::enable_if<is_container<S>::value, void>::type* = nullptr) {
            const auto i = idx(x, y);
            assert(size_t(i) < grid.size());
            grid[i].push_back(pixel<T>(x, y, v));
        }
        
        template<typename S = Set_t>
        typename Set_t::const_iterator find(const typename decltype(grid)::const_iterator& cell, pixel<T> p, typename std::enable_if<is_set<S>::value, void>::type* = nullptr) const {
            return cell->find(p);
        }
        
        template<typename S = Set_t>
        typename Set_t::const_iterator find(const typename decltype(grid)::const_iterator& cell, pixel<T> p, typename std::enable_if<is_container<S>::value, void>::type* = nullptr) const {
            return std::find(cell->begin(), cell->end(), p);
        }
        
        virtual T query(float x, float y) const {
            auto idx = this->idx(x, y);
            
            assert((size_t)idx < grid.size());
            auto cell = grid.begin() + idx;
            auto it = find(cell, pixel<T>(x, y));
            if(it == cell->end())
                return T();
            
            return it->v;
            /*for(auto && [px, py, pv] : grid[idx]) {
                if(px == x && py == y)
                    return pv;
            }*/
            
            //return T();
        }
        
        virtual void copy_row(float x0, float x1, float y, T* data) const {
            auto tup = point2grid(x0, y);
            auto gx = std::get<0>(tup), gy = std::get<1>(tup);
            
            auto cell = grid.begin() + (long)(gx + gy * _n);
            pixel<T> px(x0, y);
            auto it = find(cell, px);
            float i=0;
            const auto L = x1 - x0;
            
            for(; i<=L && i + x0 < _resolution.width;){
                if(it == cell->end() || it->y != y) {
                    // search within cell, as the X coordinate is within bounds
                    if(i + x0 < ceil((gx + 1) * _scale.x)) {
                        px.x = i + x0;
                        it = find(cell, px);
                    }
                    else if(gx+1 >= (size_t)_n) {
                        it = cell->end();
                        
                    } else {
                        ++gx;
                        px.x = x0 + i;
                        it = find(++cell, px);
                    }
                }
                
                if(it == cell->end() || it->y != y)
                    data[size_t(i)] = T();
                else if(it->x != x0 + i) {
                    ++it;
                    assert(it->x > x0 + i);
                    std::fill(data + size_t(i), data + (size_t)cmn::min(it->x - x0 + 1, L + 1), T());
                    i = it->x - x0;
                    continue;
                }
                else {
                    data[size_t(i)] = it->v;
                    ++it;
                }
                
                ++i;
            }
        }
        
        void clear() {
            for(auto &g : grid)
                g.clear();
        }
    };

    class PixelGrid : public Grid2D<uint8_t> {
        cv::Mat background;
        
    public:
        PixelGrid(uint n, const Size2& resolution, const cv::Mat& bg);
        virtual ~PixelGrid() {}
        
        virtual uint8_t query(float x, float y) const override;
    };
}
}
