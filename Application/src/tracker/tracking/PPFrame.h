#pragma once

#include <commons.pc.h>

#include <pv.h>
#include <misc/bid.h>
#include <misc/idx_t.h>
#include <misc/vec2.h>
#include <tracking/IndividualCache.h>
#include <misc/ProximityGrid.h>
#include <tracking/TrackingSettings.h>
#include <misc/ThreadPool.h>

#ifndef NDEBUG
#define TREX_ENABLE_HISTORY_LOGS true
#else
#define TREX_ENABLE_HISTORY_LOGS false
#endif

namespace track {
using namespace cmn;

template<class T> struct is_bytell : public std::false_type {};
template<class T, class Compare, class Alloc>
struct is_bytell<ska::bytell_hash_map<T, Compare, Alloc>> : public std::true_type {};

template<typename U, typename T>
concept VoidTransformer = std::invocable<T, U&> && std::is_same<std::invoke_result_t<T, U&>, void>::value;

template<typename U, typename T>
concept Predicate = std::invocable<T, U&> && std::is_same<std::invoke_result_t<T, U&>, bool>::value;

template<typename U, typename T>
concept IndexedTransformer = std::invocable<T, size_t, U&> && std::is_same<std::invoke_result_t<T, size_t, U&>, void>::value;

template<typename U, typename T>
concept Transformer = VoidTransformer<U, T> || Predicate<U, T> || IndexedTransformer<U, T>;


class PPFrame {
public:
    //robin_hood::unordered_map<long_t, std::set<pv::bid>> fish_mappings;
    robin_hood::unordered_map<pv::bid, std::set<Idx_t>> blob_mappings;
    robin_hood::unordered_map<Idx_t, ska::bytell_hash_map<pv::bid, Match::prob_t>> paired;
    robin_hood::unordered_map<Idx_t, Vec2> last_positions;
    
    std::atomic<uint64_t> _split_objects{0}, _split_pixels{0};
    
#if TREX_ENABLE_HISTORY_LOGS
    inline static std::shared_ptr<std::ofstream> history_log;
    inline static std::mutex log_mutex;
#endif
    
    template<typename... Args>
    static inline void Log([[maybe_unused]] Args&& ...args) {
#if TREX_ENABLE_HISTORY_LOGS
        if(!history_log)
            return;
        write_log(format<FormatterType::NONE>(std::forward<Args>(args)...));
#else
        
        return;
#endif
    }
    
    static void CloseLogs();
    static void UpdateLogs();
    
private:
    static void write_log(std::string str);
    
public:
    using cache_map_t = robin_hood::unordered_node_map<Idx_t, IndividualCache>;
    
    //! Time in seconds
    double time;
    
    GETTER_SETTER(double, loading_time)
    
public:
    //! Original timestamp
    timestamp_t timestamp;
    
    //! Original frame index
    GETTER_SETTER(Frame_t, index)

public:
    bool _finalized = false;
    source_location _finalized_at;
public:
    Settings::manual_matches_t::mapped_type fixed_matches;
    
private:
    std::vector<pv::BlobPtr> _tags;
    //GETTER(std::vector<pv::bid>, blobs)
    //GETTER(std::vector<pv::bid>, original_blobs)
    //GETTER(std::vector<pv::bid>, noise)
    std::vector<pv::BlobPtr> _blob_owner;
    std::vector<pv::BlobPtr> _noise_owner;
    
    GETTER_I(size_t, num_pixels, 0)
    GETTER_I(size_t, pixel_samples, 0)
    
    GETTER_NCONST(cache_map_t, individual_cache)
    
    GETTER(std::vector<Idx_t>, previously_active_identities)
    
public:
    const IndividualCache* cached(Idx_t) const;
    void init_cache(GenericThreadPool* pool);
    
private:
    void set_cache(Idx_t, IndividualCache&&);
    
protected:
    GETTER(grid::ProximityGrid, blob_grid)
    
public:
    auto& unsafe_access_all_blobs() { return _blob_owner; }
    
    int label(const pv::bid&) const;
    bool has_fixed_matches() const;
    size_t number_objects() const { return _blob_owner.size() + _noise_owner.size(); }
    size_t N_blobs() const { return _blob_owner.size(); }
    size_t N_noise() const { return _noise_owner.size(); }
    
    /**
     * Blob related functions below.
     */
    
    //! Adds one blob to _blobs.
    void add_regular(pv::BlobPtr&&);
    //! Adds one blob to _noise.
    void add_noise(pv::BlobPtr&&);
    
    //! Adds a vector of blobs to _blobs.
    void add_regular(std::vector<pv::BlobPtr>&& v);
    //! Adds a vector of blobs to _noise.
    void add_noise(std::vector<pv::BlobPtr>&& v);
    
    //! Simply moves one blob from _blobs to _noise.
    /// Has the advantage of not requiring recalculate of num_pixels.
    void move_to_noise(size_t blob_index);
    
    //! Tries to find the given blob in any of the arrays and removes it.
    //bool erase_anywhere(pv::bid bdx);
    pv::BlobPtr extract(pv::bid bdx);
    
    template<typename T, typename K = remove_cvref_t<T>>
    auto extract_from_range(std::vector<pv::BlobPtr>& objects,
                            std::vector<pv::BlobPtr>&& owner,
                            T&& vector,
                            bool physical_remove = true)
    {
        for(auto it = owner.begin(); it != owner.end(); ) {
            auto&& own = *it;
            if(!own) {
                ++it;
                continue;
            }
            
            auto bdx = own->blob_id();
            
            bool found;
            
            if constexpr(cmn::is_map<K>::value && !is_bytell<K>::value) {
                found = vector.contains(bdx);
            } else if constexpr(is_bytell<K>::value) {
                found = vector.count(bdx);
            } else {
                if(vector.empty())
                    break;
                
                auto vit = std::find(vector.begin(), vector.end(), bdx);
                found = vit != vector.end();
                if(found)
                    vector.erase(vit);
            }
            
            if(!found) {
                ++it;
            } else {
                //! we found the blob, so remove it everywhere...
                if(!_finalized) {
                    _blob_grid.erase(bdx);
                    
                    _num_pixels -= own->num_pixels();
                    _pixel_samples--;
                }
                
                // move object out and delete
                objects.emplace_back(std::move(own));
                
                // update iterator
                if(physical_remove)
                    it = owner.erase(it);
                else
                    ++it;
            }
        }
        
    }
    
    template<typename T, typename K = remove_cvref_t<T>>
        requires (cmn::is_container<K>::value) || (cmn::is_map<K>::value) || (cmn::is_set<K>::value) || std::same_as<K, UnorderedVectorSet<pv::bid>>
    std::vector<pv::BlobPtr> extract_from_blobs(T&& vector) {
        std::vector<pv::BlobPtr> objects;
        objects.reserve(vector.size());
        extract_from_range(objects, std::move(_blob_owner), std::forward<T>(vector));
        _check_owners();
        return objects;
    }
    
    template<typename T, typename K = remove_cvref_t<T>>
        requires (cmn::is_container<K>::value) || (cmn::is_map<K>::value) || (cmn::is_set<K>::value) || std::same_as<K, UnorderedVectorSet<pv::bid>>
    std::vector<pv::BlobPtr> extract_from_noise(T&& vector) {
        std::vector<pv::BlobPtr> objects;
        objects.reserve(vector.size());
        extract_from_range(objects, std::move(_noise_owner), std::forward<T>(vector));
        _check_owners();
        return objects;
    }
    
    template<typename T, typename K = remove_cvref_t<T>>
        requires (cmn::is_container<K>::value) || (cmn::is_map<K>::value) || (cmn::is_set<K>::value) || std::same_as<K, UnorderedVectorSet<pv::bid>>
    std::vector<pv::BlobPtr> extract_from_all(T&& vector) {
        std::vector<pv::BlobPtr> objects;
        objects.reserve(vector.size());
        extract_from_range(objects, std::move(_blob_owner), std::forward<T>(vector));
        extract_from_range(objects, std::move(_noise_owner), std::forward<T>(vector));
        _check_owners();
        return objects;
    }
    
    template<typename T, typename K = remove_cvref_t<T>>
        requires (cmn::is_container<K>::value)
                || (cmn::is_map<K>::value)
                || (cmn::is_set<K>::value)
                || (std::same_as<K, UnorderedVectorSet<pv::bid>>)
    std::vector<pv::BlobPtr> extract_from_blobs_unsafe(T&& vector) {
        std::vector<pv::BlobPtr> objects;
        objects.reserve(vector.size());
        extract_from_range(objects, std::move(_blob_owner), std::forward<T>(vector), false);
        _check_owners();
        return objects;
    }
    
    pv::BlobPtr create_copy(pv::bid bdx) const;
    
    //! If the bdx can be found, this removes it from the _blobs array
    /// and returns a pv::BlobPtr. Otherwise nullptr is returned.
    //bool erase_regular(pv::bid bdx);
    
    //! If the bdx can be found in any of the arrays, this will return
    /// something != nullptr.
    pv::BlobWeakPtr bdx_to_ptr(pv::bid bdx) const;
    bool has_bdx(pv::bid bdx) const;
    
    //! Tries to find a blob in the original blobs.
    //pv::BlobPtr find_original_bdx(pv::bid bdx) const;
    
    //! Will return the pv::BlobPtr assigned with the given bdx.
    /// If the bdx cannot be found, this will throw!
    //const pv::BlobPtr& bdx_to_ptr(pv::bid bdx) const;

    void set_tags(std::vector<pv::BlobPtr>&&);
    auto& tags() { return _tags; }
    
    //! Only remove blobs and update pixels arrays.
    void clear_blobs();
    
    //! Adds both from blobs and noise, assuming that pixels and samples are already known.
    void add_blobs(std::vector<pv::BlobPtr>&& blobs,
                   std::vector<pv::BlobPtr>&& noise,
                   size_t pixels, size_t samples);
    
    void fill_proximity_grid();
    void finalize(source_location loc = source_location::current());
    void init_from_blobs(std::vector<pv::BlobPtr>&& vec);
    
    template<typename T>
        requires Transformer<pv::Blob, T>
    void transform_all(T&& F) const {
        size_t i = 0;
        
        for(auto &own : _blob_owner) {
            if(!own)
                continue;
            
            if constexpr(VoidTransformer<pv::Blob, T>) {
                F(*own);
            } else if constexpr(Predicate<pv::Blob, T>) {
                if(!F(*own))
                    break;
            } else if constexpr(IndexedTransformer<pv::Blob, T>) {
                F(i++, *own);
            } else {
                static_assert(sizeof(T) == 0, "Transformer type not implemented.");
            }
        }
        
        for(auto &own : _noise_owner) {
            if(!own)
                continue;
            
            if constexpr(VoidTransformer<pv::Blob, T>) {
                F(*own);
            } else if constexpr(Predicate<pv::Blob, T>) {
                if(!F(*own))
                    break;
            } else if constexpr(IndexedTransformer<pv::Blob, T>) {
                F(i++, *own);
            } else {
                static_assert(sizeof(T) == 0, "Transformer type not implemented.");
            }
        }
    }
    
    template<typename T>
        requires Transformer<pv::Blob, T>
    void transform_noise(T&& F) const {
        size_t i = 0;
        for(auto &own : _noise_owner) {
            if(!own)
                continue;
            
            if constexpr(VoidTransformer<pv::Blob, T>) {
                F(*own);
            } else if constexpr(Predicate<pv::Blob, T>) {
                if(!F(*own))
                    break;
            } else if constexpr(IndexedTransformer<pv::Blob, T>) {
                F(i++, *own);
            } else {
                static_assert(sizeof(T) == 0, "Transformer type not implemented.");
            }
        }
    }
    
    template<typename T>
        requires Transformer<pv::bid, T>
    void transform_blob_ids(T&& F) const {
        size_t i = 0;
        for(auto &blob : _blob_owner) {
            if(!blob)
                continue;
            
            if constexpr(VoidTransformer<pv::bid, T>) {
                F(blob->blob_id());
            } else if constexpr(Predicate<pv::bid, T>) {
                if(!F(blob->blob_id()))
                    break;
            } else if constexpr(IndexedTransformer<pv::bid, T>) {
                F(i++, blob->blob_id());
            } else {
                static_assert(sizeof(T) == 0, "Transformer type not implemented.");
            }
        }
    }
    
    template<typename F>
        requires Transformer<pv::Blob, F>
    void transform_blobs(F&& fn) const {
        size_t i = 0;
        for(auto &own : _blob_owner) {
            if(!own)
                continue;
            
            if constexpr(VoidTransformer<pv::Blob, F>) {
                fn(*own);
            } else if constexpr(Predicate<pv::Blob, F>) {
                if(!fn(*own))
                    break;
            } else if constexpr(IndexedTransformer<pv::Blob, F>) {
                fn(i++, *own);
            } else {
                static_assert(sizeof(F) == 0, "Transformer type not implemented.");
            }
        }
    }
    
    template<typename F>
        requires Predicate<pv::Blob, F>
    void move_to_noise_if(F && fn) {
        for(auto it = _blob_owner.begin(); it != _blob_owner.end(); ) {
            auto &&own = *it;
            if(!own)
                continue;
            
            if(fn(*own)) {
                _noise_owner.emplace_back(std::move(own));
                it = _blob_owner.erase(it);
                //_blob_owner.erase(std::find(_blobs.begin(), _blobs.end(), own.blob->blob_id()));
            } else
                ++it;
        }
        
        _check_owners();
    }
    
    bool is_regular(pv::bid bdx) const;
    
    PPFrame();
    ~PPFrame();

    PPFrame(const PPFrame&) = delete;
    PPFrame(PPFrame&&) noexcept = delete;
    PPFrame& operator=(const PPFrame&) = delete;
    PPFrame& operator=(PPFrame&&) noexcept = delete;
    
    void clear();
    
private:
    void _assume_not_finalized(const char*, int);
    pv::bid _add_ownership(bool regular, pv::BlobPtr&&);
    pv::BlobPtr _extract_from(std::vector<pv::BlobPtr>&& range, pv::bid bdx);
    void _check_owners();
};

}
