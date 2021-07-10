#include "PPFrame.h"
#include <tracking/Tracker.h>
#include <tracking/Categorize.h>

namespace track {

#define ASSUME_NOT_FINALIZED _assume_not_finalized( __FILE__ , __LINE__ )

inline void insert_line(grid::ProximityGrid& grid, const HorizontalLine* ptr, uint32_t blob_id, size_t step_size)
{
    auto d = ptr->x1 - ptr->x0;
    grid.insert(ptr->x0, ptr->y, blob_id);
    grid.insert(ptr->x1, ptr->y, blob_id);
    grid.insert(ptr->x0 + d * 0.5, ptr->y, blob_id);

    if(d >= (short)step_size * 2 && step_size >= 5) {
        for(auto x = ptr->x0 + step_size; x <= ptr->x1 - step_size; x += step_size) {
            grid.insert(x, ptr->y, blob_id);
        }
    }
}

PPFrame::PPFrame()
    : _blob_grid(Tracker::average().bounds().size())
{ }

void PPFrame::_add_to_map(const pv::BlobPtr &blob) {
    _bdx_to_ptr[blob->blob_id()] = blob;
}

void PPFrame::_remove_from_map(uint32_t bdx) {
    _bdx_to_ptr.erase(bdx);
}

void PPFrame::_assume_not_finalized(const char* file, int line) {
    if(_finalized) {
        U_EXCEPTION("PPFrame already finalized at [%s:%d].", file, line);
    }
}

int PPFrame::label(const pv::BlobPtr& blob) const {
    auto l = Categorize::DataStore::ranged_label(Frame_t(index()), blob->blob_id());
    if(l)
        return l->id;
    return -1;
}

void PPFrame::add_noise(const pv::BlobPtr & blob) {
    ASSUME_NOT_FINALIZED;
    
    _add_to_map(blob);
    _noise.emplace_back(blob);
    _num_pixels += blob->num_pixels();
    ++_pixel_samples;
}

void PPFrame::add_noise(std::vector<pv::BlobPtr>&& v) {
    ASSUME_NOT_FINALIZED;
    
    for(auto &b:v) {
        _add_to_map(b);
        _num_pixels += b->num_pixels();
    }
    
    _pixel_samples += v.size();
    _noise.insert(_noise.end(), std::make_move_iterator( v.begin() ), std::make_move_iterator( v.end() ));
}

void PPFrame::move_to_noise(size_t blob_index) {
    ASSUME_NOT_FINALIZED;
    assert(blob_index < _blobs.size());
    
    // no update of pixels or maps is required
    _noise.insert(_noise.end(), std::make_move_iterator(_blobs.begin() + blob_index), std::make_move_iterator(_blobs.begin() + blob_index + 1));
    _blobs.erase(_blobs.begin() + blob_index);
}

void PPFrame::erase_anywhere(const pv::BlobPtr& blob) {
    ASSUME_NOT_FINALIZED;
    
    auto it = std::find(_blobs.begin(), _blobs.end(), blob);
    if(it != _blobs.end()) {
        _num_pixels -= blob->num_pixels();
        --_pixel_samples;
        _remove_from_map(blob->blob_id());
        _blobs.erase(it);
        
    } else if((it = std::find(_noise.begin(), _noise.end(), blob)) != _noise.end()) {
        _num_pixels -= blob->num_pixels();
        --_pixel_samples;
        _remove_from_map(blob->blob_id());
        _noise.erase(it);
    }
#ifndef NDEBUG
    else
        U_EXCEPTION("Blob %u not found anywhere.", blob->blob_id());
#endif
}

pv::BlobPtr PPFrame::erase_anywhere(uint32_t bdx) {
    ASSUME_NOT_FINALIZED;
    
    auto find = [bdx](const auto& blob){ return blob->blob_id() == bdx; };
    auto it = std::find_if(_blobs.begin(), _blobs.end(), find);
    if(it != _blobs.end()) {
        auto blob = *it;
        _num_pixels -= blob->num_pixels();
        --_pixel_samples;
        _remove_from_map(bdx);
        _blobs.erase(it);
        return blob;
        
    } else if((it = std::find_if(_noise.begin(), _noise.end(), find)) != _noise.end()) {
        auto blob = *it;
        _num_pixels -= blob->num_pixels();
        --_pixel_samples;
        _remove_from_map(bdx);
        _noise.erase(it);
        return blob;
    }
#ifndef NDEBUG
    else
        U_EXCEPTION("Blob %u not found anywhere.", bdx);
#endif
    return nullptr;
}

void PPFrame::add_regular(const pv::BlobPtr & blob) {
    ASSUME_NOT_FINALIZED;
    
    _add_to_map(blob);
    _blobs.emplace_back(blob);
    _num_pixels += blob->num_pixels();
    ++_pixel_samples;
}

void PPFrame::add_regular(std::vector<pv::BlobPtr>&& v) {
    ASSUME_NOT_FINALIZED;
    
    for(auto &b:v) {
        _add_to_map(b);
        _num_pixels += b->num_pixels();
    }
    _pixel_samples += v.size();
    _blobs.insert(_blobs.end(), std::make_move_iterator( v.begin() ), std::make_move_iterator( v.end() ));
}

pv::BlobPtr PPFrame::erase_regular(uint32_t bdx) {
    ASSUME_NOT_FINALIZED;
    
    auto it = _bdx_to_ptr.find(bdx);
    if(it == _bdx_to_ptr.end()) {
        return nullptr; // not found
    }
    
    auto bit = std::find(_blobs.begin(), _blobs.end(), it->second);
    if(bit != _blobs.end()) {
        auto ptr = *bit;
        _num_pixels -= ptr->num_pixels();
        --_pixel_samples;
        _remove_from_map(bdx);
        _blobs.erase(bit);
        return ptr;
    }
    
    return nullptr;
}

pv::BlobPtr PPFrame::find_bdx(uint32_t bdx) const {
    auto it = _bdx_to_ptr.find(bdx);
    if(it != _bdx_to_ptr.end()) {
        return it->second;
    }
    return nullptr;
}

const pv::BlobPtr& PPFrame::bdx_to_ptr(uint32_t bdx) const {
    return _bdx_to_ptr.at(bdx);
}

void PPFrame::clear_blobs() {
    ASSUME_NOT_FINALIZED;
    
    _blobs.clear();
    _noise.clear();
    _num_pixels = 0;
    _pixel_samples = 0;
    _bdx_to_ptr.clear();
}

void PPFrame::add_blobs(std::vector<pv::BlobPtr>&& blobs,
                        std::vector<pv::BlobPtr>&& noise,
                        size_t pixels,
                        size_t samples)
{
    ASSUME_NOT_FINALIZED;
    assert(samples == blobs.size() + noise.size());
    _num_pixels += pixels;
    _pixel_samples += samples;
    
    for(auto &b : blobs)
        _add_to_map(b);
    for(auto &b : noise)
        _add_to_map(b);
    
    _blobs.insert(_blobs.end(), std::make_move_iterator(blobs.begin()), std::make_move_iterator(blobs.end()));
    _noise.insert(_noise.end(), std::make_move_iterator(noise.begin()), std::make_move_iterator(noise.end()));
}

void PPFrame::finalize() {
    ASSUME_NOT_FINALIZED;
    _finalized = true;
}

void PPFrame::init_from_blobs(std::vector<pv::BlobPtr>&& vec) {
    ASSUME_NOT_FINALIZED;
    
    add_regular(std::move(vec));
    _original_blobs = _blobs; // also save to copy to original.
}

PPFrame::~PPFrame() { }

void PPFrame::clear() {
    _finalized = false;
    _blobs.clear();
    _noise.clear();
    cached_individuals.clear();
    _blob_grid.clear();
    _original_blobs.clear();
    blob_cliques.clear();
    fish_cliques.clear();
    split_blobs.clear();
    _bdx_to_ptr.clear();
    _num_pixels = 0;
    _pixel_samples = 0;
}

void PPFrame::fill_proximity_grid() {
    ASSUME_NOT_FINALIZED;
    
    for(auto &b : _blobs) {
        auto N = b->hor_lines().size();
        auto ptr = b->hor_lines().data();
        const auto end = ptr + N;
        
        auto &size = b->bounds().size();
        const size_t step_size = 2;
        const size_t step_size_x = (size_t)max(1, size.width * 0.1);
        
        if(N >= step_size * 2) {
            insert_line(_blob_grid, ptr, b->blob_id(), step_size_x);
            
            for(ptr = ptr + 1; ptr < end-1; ++ptr) {
                if(ptr->y % step_size == 0) {
                    insert_line(_blob_grid, ptr, b->blob_id(), step_size_x);
                }
            }
            
            insert_line(_blob_grid, end-1, b->blob_id(), step_size_x);
            
        } else {
            for(; ptr != end; ++ptr)
                insert_line(_blob_grid, ptr, b->blob_id(), step_size_x);
        }
    }
}

}
