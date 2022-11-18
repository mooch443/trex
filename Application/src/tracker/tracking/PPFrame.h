#pragma once

#include <types.h>
#include <pv.h>
#include <misc/bid.h>
#include <misc/idx_t.h>
#include <tracking/IndividualCache.h>

namespace track {

class PPFrame {
    GETTER_NCONST(pv::Frame, frame)
    GETTER_SETTER(Frame_t, index)
public:
    //! Time in seconds
    double time;
    
    //! Original timestamp
    timestamp_t timestamp;
    
    //! Original frame index
    //long_t index;
    bool _finalized = false;
    
private:
    std::vector<pv::BlobPtr> _tags;
    std::vector<pv::BlobPtr> _single_blobs;
    GETTER(std::vector<pv::BlobPtr>, blobs)
    GETTER(std::vector<pv::BlobPtr>, original_blobs)
    GETTER(std::vector<pv::BlobPtr>, noise)
    
    GETTER_I(size_t, num_pixels, 0)
    GETTER_I(size_t, pixel_samples, 0)
    
    GETTER_NCONST(std::vector<IndividualCache>, individual_cache)
    
public:
    const IndividualCache* cached(Idx_t) const;
    
    //std::map<Idx_t, IndividualCache> cached_individuals;
    ska::bytell_hash_map<pv::bid, UnorderedVectorSet<Idx_t>> clique_for_blob;
    ska::bytell_hash_map<pv::bid, UnorderedVectorSet<pv::bid>> clique_second_order;
    UnorderedVectorSet<pv::bid> split_blobs;
    
protected:
    ska::bytell_hash_map<pv::bid, pv::BlobPtr> _bdx_to_ptr;
    GETTER(grid::ProximityGrid, blob_grid)
    
public:
    int label(const pv::BlobPtr&) const;
    
    /**
     * Blob related functions below.
     */
    
    //! Adds one blob to _blobs.
    void add_regular(const pv::BlobPtr&);
    //! Adds one blob to _noise.
    void add_noise(const pv::BlobPtr&);
    
    //! Adds a vector of blobs to _blobs.
    void add_regular(std::vector<pv::BlobPtr>&& v);
    //! Adds a vector of blobs to _noise.
    void add_noise(std::vector<pv::BlobPtr>&& v);
    
    //! Simply moves one blob from _blobs to _noise.
    /// Has the advantage of not requiring recalculate of num_pixels.
    void move_to_noise(size_t blob_index);
    
    //! Tries to find the given blob in any of the arrays and removes it.
    void erase_anywhere(const pv::BlobPtr& blob);
    pv::BlobPtr erase_anywhere(pv::bid bdx);
    
    //! If the bdx can be found, this removes it from the _blobs array
    /// and returns a pv::BlobPtr. Otherwise nullptr is returned.
    pv::BlobPtr erase_regular(pv::bid bdx);
    
    //! If the bdx can be found in any of the arrays, this will return
    /// something != nullptr.
    pv::BlobPtr find_bdx(pv::bid bdx) const;
    
    //! Tries to find a blob in the original blobs.
    pv::BlobPtr find_original_bdx(pv::bid bdx) const;
    
    //! Will return the pv::BlobPtr assigned with the given bdx.
    /// If the bdx cannot be found, this will throw!
    const pv::BlobPtr& bdx_to_ptr(pv::bid bdx) const;

    void set_tags(std::vector<pv::BlobPtr>&&);
    std::vector<pv::BlobPtr>& tags() { return _tags; }
    
    //! Only remove blobs and update pixels arrays.
    void clear_blobs();
    
    //! Adds both from blobs and noise, assuming that pixels and samples are already known.
    void add_blobs(std::vector<pv::BlobPtr>&& blobs, std::vector<pv::BlobPtr>&& noise, size_t pixels, size_t samples);
    
    void fill_proximity_grid();
    void finalize();
    void init_from_blobs(std::vector<pv::BlobPtr>&& vec);
    
    PPFrame();
    ~PPFrame();
    
    void clear();
    
private:
    void _assume_not_finalized(const char*, int);
    bool _add_to_map(const pv::BlobPtr&);
    void _remove_from_map(pv::bid);
};

}
