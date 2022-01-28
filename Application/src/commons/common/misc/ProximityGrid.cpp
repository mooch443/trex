#include "ProximityGrid.h"
#include <misc/metastring.h>
#include <misc/ranges.h>

namespace cmn {
namespace grid {

    ProximityGrid::ProximityGrid(const Size2& resolution, int r)
        : Grid2D(resolution, r != -1 ? r : proximity_res)
    {}
    
    std::unordered_set<ProximityGrid::result_t> ProximityGrid::query(const Vec2& point, float max_d) const {
        std::vector<result_t> result;
        std::vector<fdx_pos> found;
        
        //! assume squared grid
        assert(_scale.x == _scale.y);
        
        //! this is the number of grid cells in x/y direction that have to be walked until max_d is reached anyway
        // +1 because we are somewhere in a cell, so just be sure (but we dont need to add here because we use <=)
        int max_cells = ceilf(max_d / _scale.max());
        
        max_d *= max_d;
        
        //! this is the grid cell in the center, go in each direction from this cell
        int cdx = floorf(max(0, point.x) / _scale.x);
        int cdy = floorf(max(0, point.y) / _scale.y);
        
        //Debug("In cell %d,%d / %d,%d", cdx, cdy, proximity_res,proximity_res);
        Rangei cells_x(min(proximity_res - 1, max(0, cdx - max_cells)), min(proximity_res - 1, max(0, cdx + max_cells)));
        Rangei cells_y(min(proximity_res - 1, max(0, cdy - max_cells)), min(proximity_res - 1, max(0, cdy + max_cells)));
        
        decltype(grid)::const_iterator start = grid.begin(), end = grid.end();
        decltype(grid)::const_iterator git = start + (cells_x.start + cells_y.start * proximity_res);
        const size_t stride = (proximity_res - cells_x.end + cells_x.start - 1);
        
        for (int y = cells_y.start; y <= cells_y.end; ++y) {
            assert(git == start + (cells_x.start + y * proximity_res));
            assert(git < grid.end());
            
            for(int x = cells_x.start; x <= cells_x.end; ++x, ++git) {
                assert(git < grid.end());
                
                for (auto && [x,y,v] : *git) {
                    auto d = sqdistance(Vec2(x,y), point);
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
                                result.push_back({d, v});
                            }
                            
                            continue;
                        }
                        
                        found.push_back(v);
                        result.push_back({d, v});
                    }
                }
            }
            
            // go to the next row inside the grid array
            if (git < end - stride) {
                git += stride;
            }
            else {
                assert(y >= cells_y.end);
            }
        }
        
        std::unordered_set<result_t> return_v;
        for(auto && [d, v] : result) {
            return_v.insert({sqrtf(d), v});
        }
        
        return return_v;
    }
    
    std::string ProximityGrid::str(fdx_pos dx, Vec2 point, float max_d) const {
        int max_cells = ceilf(max_d / _scale.max());
        
        int cdx = floorf(point.x / _scale.x);
        int cdy = floorf(point.y / _scale.y);
        
        //Debug("In cell %d,%d / %d,%d", cdx, cdy, proximity_res,proximity_res);
        Rangei cells_x(min(proximity_res - 1, max(0, cdx - max_cells)), min(proximity_res - 1, max(0, cdx + max_cells)));
        Rangei cells_y(min(proximity_res - 1, max(0, cdy - max_cells)), min(proximity_res - 1, max(0, cdy + max_cells)));
        
        std::stringstream ssg;
        ssg << "Grid[" << proximity_res << "^2]";
        ssg << " calculated position: (" << cdx << "," << cdy << ") " << cells_x.start << "-" << cells_x.end << " | " << cells_y.start << "-" << cells_y.end;
        ssg << "\n";
        
        /*for(size_t i=0; i<grid.size(); ++i) {
            auto &g = grid[i];
            if(!g.empty()) {
                //std::apply([](auto ...x){std::make_tuple(x...);} , *g.begin());
                size_t x = i % 100, y = i / 100;
                ssg << "\t" << x << "," << y << ": {";
                for(auto && [x,y, fdx] : g)
                    if(fdx == -1 || fdx == dx)
                        ssg << "["<<x<<", "<<y<<", "<<fdx<<"] ";
                ssg << "}";
            }
        }*/
        
        decltype(grid)::const_iterator start = grid.begin();
        decltype(grid)::const_iterator git = start + (cells_x.start + cells_y.start * proximity_res);
        const size_t stride = (proximity_res - cells_x.end + cells_x.start - 1);
        
        for (int y = cells_y.start; y <= cells_y.end; ++y) {
            assert(git == start + (cells_x.start + y * proximity_res));
            
            for(int x = cells_x.start; x <= cells_x.end; ++x, ++git) {
                assert(git < grid.end());
                
                for (auto && [fx,fy,v] : *git) {
                    if(v == -1 || v == dx) {
                        ssg << "["<<fx<<", "<<fy<<", "<<v<<" ";
                        auto d = sqrtf(sqdistance(Vec2(fx,fy), point));
                        ssg << d << "] ";
                    }
                }
            }
            
            git += stride;
        }
        
        
        return ssg.str();
    }

}
}
