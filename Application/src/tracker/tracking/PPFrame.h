#pragma once

#include <commons.pc.h>

#include <pv.h>
#include <misc/bid.h>
#include <misc/idx_t.h>
#include <tracking/IndividualCache.h>
#include <misc/ProximityGrid.h>
#include <misc/TrackingSettings.h>
#include <misc/ThreadPool.h>

//#ifndef NDEBUG
#define TREX_ENABLE_HISTORY_LOGS true
//#else
//#define TREX_ENABLE_HISTORY_LOGS false
//#endif

namespace track {
using namespace cmn;

template<typename T, typename... Args>
concept AnyTransformer =
    (std::invocable<T, Args...>)
    ||  (std::invocable<T, Args&...>)
    ||  (std::invocable<T, Args&&...>);

template<typename T, typename... Args>
concept VoidTransformer =
    (std::invocable<T, Args...>
        && std::is_same<std::invoke_result_t<T, Args...>, void>::value)
    ||  (std::invocable<T, Args&...>
            && std::is_same<std::invoke_result_t<T, Args&...>, void>::value)
    ||  (std::invocable<T, Args&&...>
            && std::is_same<std::invoke_result_t<T, Args&&...>, void>::value);

template<typename T, typename... Args>
concept Predicate =
        (std::invocable<T, Args&...>
            && std::is_same<std::invoke_result_t<T, Args&...>, bool>::value)
    ||  (std::invocable<T, Args&&...>
            && std::is_same<std::invoke_result_t<T, Args&&...>, bool>::value);

template<typename T, typename... Args>
concept IndexedTransformer =
    (std::invocable<T, size_t, Args...>
        && std::is_same<std::invoke_result_t<T, size_t, Args...>, void>::value)
    ||  (std::invocable<T, size_t, Args&...>
            && std::is_same<std::invoke_result_t<T, size_t, Args&...>, void>::value)
    ||  (std::invocable<T, size_t, Args&&...>
            && std::is_same<std::invoke_result_t<T, size_t, Args&&...>, void>::value);

template<typename T, typename... Args>
concept Transformer = VoidTransformer<T, Args...>
                   || Predicate<T, Args...>
                   || IndexedTransformer<T, Args...>;

class PPFrame {
public:
    template<typename K, typename V>
    using map_t = robin_hood::unordered_flat_map<K, V>;
    map_t<pv::bid, std::set<Idx_t>> blob_mappings;
    map_t<Idx_t, map_t<pv::bid, Match::prob_t>> paired;
    map_t<Idx_t, Vec2> last_positions;
    
    std::atomic<uint64_t> _split_objects{0}, _split_pixels{0};
    
#if TREX_ENABLE_HISTORY_LOGS
    static std::shared_ptr<std::ofstream>& history_log();
    static LOGGED_MUTEX_TYPE& log_mutex();
#endif
    
    template<typename... Args>
    static inline void Log([[maybe_unused]] Args&& ...args) {
#if TREX_ENABLE_HISTORY_LOGS
        if(!history_log())
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
    double time{0};
    
    CacheHints hints;
    GETTER_SETTER_I(double, loading_time, 0);
    
public:
    //! Original timestamp
    timestamp_t timestamp;
    
    //! Original frame index
    GETTER_SETTER(Frame_t, index);
    //! Original frame index in the video
    GETTER_SETTER(Frame_t, source_index);

public:
    bool _finalized = false;
    source_location _finalized_at;
public:
    Settings::manual_matches_t::mapped_type fixed_matches;
    
private:
    std::vector<pv::BlobPtr> _tags;
    //GETTER(std::vector<pv::bid>, blobs);
    //GETTER(std::vector<pv::bid>, original_blobs);
    //GETTER(std::vector<pv::bid>, noise);
    std::vector<pv::BlobPtr> _blob_owner;
    std::vector<pv::BlobPtr> _noise_owner;
    robin_hood::unordered_flat_map<pv::bid, pv::BlobWeakPtr> _blob_map;
    robin_hood::unordered_flat_map<pv::bid, pv::BlobWeakPtr> _noise_map;
    robin_hood::unordered_flat_set<pv::bid> _big_ids;
    
    GETTER_I(size_t, num_pixels, 0);
    GETTER_I(size_t, pixel_samples, 0);
    GETTER_SETTER(Size2, resolution);
    
    GETTER_NCONST(cache_map_t, individual_cache);
    
    GETTER(std::vector<Idx_t>, previously_active_identities);
    
public:
    const IndividualCache* cached(Idx_t) const;
    
    enum class NeedGrid {
        Need, NoNeed
    };
    void init_cache(GenericThreadPool* pool, NeedGrid);
    
private:
    void set_cache(Idx_t, IndividualCache&&);
    
protected:
    grid::ProximityGrid _blob_grid;
    std::mutex _blob_grid_mutex;
    
public:
    const grid::ProximityGrid& blob_grid();
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
    
    enum class VectorHandling {
        Compress,
        OneToOne
    };
    
    enum class RemoveHandling {
        RemoveFromOwner,
        Leave
    };
    
    template<VectorHandling compress,
             RemoveHandling remove = RemoveHandling::RemoveFromOwner,
             typename T, typename K = remove_cvref_t<T>,
             typename Value = typename K::value_type,
             typename Positions,
             typename... Owners>
    void _extract_from_single_range(T& vector,
                                    std::vector<pv::BlobPtr>& objects,
                                    std::vector<pv::BlobPtr>& owner,
                                    Positions&& positions = {})
    {
#ifndef NDEBUG
        std::set<pv::bid> bdxes;
        auto name = Meta::name<T>();
        Log("extract_from_single_range<", name.c_str(),">: ", owner, " objects:",objects);
#endif
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
                
                if constexpr(_clean_same<typename K::value_type, pv::bid>) {
                    if constexpr(set_type<K>) {
                        if constexpr(not std::is_const_v<std::remove_reference_t<T>>) {
                            size_t count = vector.erase(bdx);
                            found = count != 0;
                        } else {
                            found = vector.contains(bdx);
                        }
                        
                    } else {
                        auto vit = std::find(vector.begin(), vector.end(), bdx);
                        found = vit != vector.end();
                        if constexpr(not std::is_const_v<std::remove_reference_t<T>>) {
                            if(found)
                                vector.erase(vit);
                        }
                    }
                    
                } else {
                    auto vit = std::find_if(vector.begin(), vector.end(), [bdx](const auto& tuple) -> bool
                    {
                        auto work = [bdx](auto... args) -> bool {
                            return find_argtype_apply([bdx](pv::bid a) -> bool {
                                return a == bdx;
                            }, args...);
                        };
                        
                        return apply_to_tuple(tuple, work);
                    });

                    found = vit != vector.end();
                    if constexpr(not std::is_const_v<std::remove_reference_t<T>>) {
                        if(found)
                            vector.erase(vit);
                    }
                }
            }
            
            if(!found) {
                ++it;
                
            } else {
#ifndef NDEBUG
                if(bdxes.contains(bdx)) {
                    throw U_EXCEPTION("Already have ", bdx, " in the map!");
                } else {
                    bdxes.insert(bdx);
                }
#endif
                
                //! we found the blob, so remove it everywhere...
                if(!_finalized) {
                    if(not _blob_grid.empty()) {
                        std::scoped_lock guard(_blob_grid_mutex);
                        _blob_grid.erase(bdx);
                    }
                    
                    _num_pixels -= own->num_pixels();
                    _pixel_samples--;
                }
                
                // move object out and delete
                if constexpr(compress == VectorHandling::OneToOne)
                    objects[positions.at(bdx)] = std::move(own);
                else {
                    objects.emplace_back(std::move(own));
                }
                
                // update iterator
                if constexpr(remove == RemoveHandling::RemoveFromOwner) {
                    it = owner.erase(it);
                    
                } else
                    ++it;
                
                if(&owner == &_blob_owner) {
                    _blob_map.erase(bdx);
                } else
                    _noise_map.erase(bdx);
            }
        }
    }
    
    // variant that ensures correct ordering
    template<VectorHandling compress,
             RemoveHandling remove = RemoveHandling::RemoveFromOwner,
             typename T, typename K = remove_cvref_t<T>,
             typename Value = typename K::value_type,
             typename... Owners>
        requires (container_type<K>
                  && compress == VectorHandling::OneToOne
                  && (is_tuple_v<Value>
                      || _clean_same<Value, pv::bid>))
    void extract_from_range(T&& vector,
                            std::vector<pv::BlobPtr>& objects,
                            Owners&... owners)
    {
        std::unordered_map<pv::bid, size_t> positions;
        for(size_t i = 0; i < vector.size(); ++i) {
            if constexpr(is_tuple_v<Value>) {
                //! if we are dealing with a tuple, we first need
                //! to find and extract the bit that is the bid
                auto work = [](auto... args) {
                    return find_argtype_apply([](pv::bid a) {
                        return a;
                    }, args...);
                };
                
                auto bdx = apply_to_tuple(vector.at(i), work);
                positions[bdx] = i;
                
            } else {
                //! otherwise, assuming a vector of bids,
                //! this becomes a bit easier:
                positions[vector.at(i)] = i;
            }
        }
        
        assert(objects.empty());
        objects.resize(vector.size());
        
        ( _extract_from_single_range<VectorHandling::OneToOne, remove>(vector, objects, owners, positions), ... );
    }
    
    template<VectorHandling compress,
             RemoveHandling remove = RemoveHandling::RemoveFromOwner,
             typename T, typename K = remove_cvref_t<T>,
             typename... Owners>
        requires (compress == VectorHandling::Compress)
                && (not (container_type<K>
                    && (is_tuple_v<typename K::value_type>
                        || _clean_same<typename K::value_type, pv::bid>)))
    void extract_from_range(T&& vector,
                            std::vector<pv::BlobPtr>& objects,
                            Owners&... owners)
    {
        ( _extract_from_single_range<VectorHandling::Compress, remove>(std::forward<T>(vector), objects, owners, (void*)nullptr), ...);
    }
    
    template<VectorHandling compress = VectorHandling::Compress,
             RemoveHandling remove = RemoveHandling::RemoveFromOwner,
             typename T, typename K = remove_cvref_t<T>>
        requires (cmn::is_container<K>::value) || (cmn::is_map<K>::value) || (cmn::is_set<K>::value) || std::same_as<K, UnorderedVectorSet<pv::bid>>
    std::vector<pv::BlobPtr> extract_from_blobs(T&& vector) {
        std::vector<pv::BlobPtr> objects;
        objects.reserve(vector.size());
        extract_from_range<compress, remove>(std::forward<T>(vector), objects, _blob_owner);
        _check_owners();
        return objects;
    }
    
    template<VectorHandling compress = VectorHandling::Compress,
             RemoveHandling remove = RemoveHandling::RemoveFromOwner,
             typename T, typename K = remove_cvref_t<T>>
        requires (cmn::is_container<K>::value) || (cmn::is_map<K>::value) || (cmn::is_set<K>::value) || std::same_as<K, UnorderedVectorSet<pv::bid>>
    std::vector<pv::BlobPtr> extract_from_noise(T&& vector) {
        std::vector<pv::BlobPtr> objects;
        objects.reserve(vector.size());
        extract_from_range<compress, remove>(std::forward<T>(vector), objects,  _noise_owner);
        _check_owners();
        return objects;
    }
    
    template<VectorHandling compress = VectorHandling::Compress,
             RemoveHandling remove = RemoveHandling::RemoveFromOwner,
             typename T, typename K = remove_cvref_t<T>>
        requires (cmn::is_container<K>::value) || (cmn::is_map<K>::value) || (cmn::is_set<K>::value) || std::same_as<K, UnorderedVectorSet<pv::bid>>
    std::vector<pv::BlobPtr> extract_from_all(T&& vector) {
        std::vector<pv::BlobPtr> objects;
        objects.reserve(vector.size());
        extract_from_range<compress, remove>(std::forward<T>(vector), objects, _blob_owner, _noise_owner);
        _check_owners();
        return objects;
    }
    
    template<VectorHandling compress = VectorHandling::Compress,
             typename T, typename K = remove_cvref_t<T>>
        requires (cmn::is_container<K>::value)
                || (cmn::is_map<K>::value)
                || (cmn::is_set<K>::value)
                || (std::same_as<K, UnorderedVectorSet<pv::bid>>)
    std::vector<pv::BlobPtr> extract_from_blobs_unsafe(T&& vector) {
        std::vector<pv::BlobPtr> objects;
        objects.reserve(vector.size());
        extract_from_range<compress, RemoveHandling::Leave>(std::forward<T>(vector), objects, _blob_owner);
        _check_owners();
        return objects;
    }
    
    pv::BlobPtr create_copy(pv::bid bdx) const;
    
    //! If the bdx can be found, this removes it from the _blobs array
    /// and returns a pv::BlobPtr. Otherwise nullptr is returned.
    //bool erase_regular(pv::bid bdx);
    
    //! If the bdx can be found in any of the arrays, this will return
    /// something != nullptr.
    pv::BlobWeakPtr bdx_to_ptr(pv::bid bdx) const noexcept;
    bool has_bdx(pv::bid bdx) const noexcept;
    
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
                   robin_hood::unordered_flat_set<pv::bid>&& big_ids,
                   size_t pixels, size_t samples);
    
    void fill_proximity_grid(const Size2&);
    void finalize(source_location loc = source_location::current());
    void init_from_blobs(std::vector<pv::BlobPtr>&& vec);
    
    template<typename T>
        requires Transformer<T, pv::Blob>
    void transform_all(T&& F) const {
        size_t i = 0;
        
        for(auto &own : _blob_owner) {
            if(!own)
                continue;
            
            if constexpr(VoidTransformer<T, pv::Blob>) {
                F(*own);
            } else if constexpr(Predicate<T, pv::Blob>) {
                if(!F(*own))
                    break;
            } else if constexpr(IndexedTransformer<T, pv::Blob>) {
                F(i++, *own);
            } else {
                static_assert(sizeof(T) == 0, "Transformer type not implemented.");
            }
        }
        
        for(auto &own : _noise_owner) {
            if(!own)
                continue;
            
            if constexpr(VoidTransformer<T, pv::Blob>) {
                F(*own);
            } else if constexpr(Predicate<T, pv::Blob>) {
                if(!F(*own))
                    break;
            } else if constexpr(IndexedTransformer<T, pv::Blob>) {
                F(i++, *own);
            } else {
                static_assert(sizeof(T) == 0, "Transformer type not implemented.");
            }
        }
    }
    
    template<typename T>
        requires Transformer<T, pv::Blob>
    void transform_noise(T&& F) const {
        size_t i = 0;
        for(auto &own : _noise_owner) {
            if(!own)
                continue;
            
            if constexpr(VoidTransformer<T, pv::Blob>) {
                F(*own);
            } else if constexpr(Predicate<T, pv::Blob>) {
                if(!F(*own))
                    break;
            } else if constexpr(IndexedTransformer<T, pv::Blob>) {
                F(i++, *own);
            } else {
                static_assert(sizeof(T) == 0, "Transformer type not implemented.");
            }
        }
    }
    
    template<typename T>
        requires Transformer<T, pv::bid>
    void transform_blob_ids(T&& F) const {
        size_t i = 0;
        for(auto &blob : _blob_owner) {
            if(!blob)
                continue;
            
            if constexpr(VoidTransformer<T, pv::bid>) {
                F(blob->blob_id());
            } else if constexpr(Predicate<T, pv::bid>) {
                if(!F(blob->blob_id()))
                    break;
            } else if constexpr(IndexedTransformer<T, pv::bid>) {
                F(i++, blob->blob_id());
            } else {
                static_assert(sizeof(T) == 0, "Transformer type not implemented.");
            }
        }
    }
    
    template<typename K, typename T, typename Target = pv::Blob>
        requires Transformer<T, Target> && set_type<K>
    void transform_noise_ids(const K& ids, T&& F) const {
        size_t i = 0;
        for(auto &id : ids) {
            auto it = _noise_map.find(id);
            if(it != _noise_map.end()) {
                auto &blob = *it->second;
                
                if constexpr(VoidTransformer<T, Target>) {
                    F(blob);
                } else if constexpr(Predicate<T, Target>) {
                    if(!F(blob))
                        break;
                } else if constexpr(IndexedTransformer<T, Target>) {
                    F(i++, blob);
                } else {
                    static_assert(sizeof(T) == 0, "Transformer type not implemented.");
                }
            }
        }
    }
    
    template<typename F>
        requires Transformer<F, pv::Blob>
    void transform_blobs(F&& fn) const {
        size_t i = 0;
        for(auto &own : _blob_owner) {
            if(!own)
                continue;
            
            if constexpr(VoidTransformer<F, pv::Blob>) {
                fn(*own);
            } else if constexpr(Predicate<F, pv::Blob>) {
                if(!fn(*own))
                    break;
            } else if constexpr(IndexedTransformer<F, pv::Blob>) {
                fn(i++, *own);
            } else {
                static_assert(sizeof(F) == 0, "Transformer type not implemented.");
            }
        }
    }
    
    template<typename F>
        requires Predicate<F, pv::Blob>
    void move_to_noise_if(F && fn) {
        for(auto it = _blob_owner.begin(); it != _blob_owner.end(); ) {
            auto &&own = *it;
            if(!own) {
                ++it;
                continue;
            }
            
            if(fn(*own)) {
                _noise_map[own->blob_id()] = own.get();
                _blob_map.erase(own->blob_id());
                
                _noise_owner.emplace_back(std::move(own));
                it = _blob_owner.erase(it);
                
                //_blob_owner.erase(std::find(_blobs.begin(), _blobs.end(), own.blob->blob_id()));
            } else
                ++it;
        }
        
        _check_owners();
    }
    
    bool is_regular(pv::bid bdx) const;
    
    PPFrame() noexcept = default;
    PPFrame(const Size2&);

    PPFrame(const PPFrame&) = delete;
    PPFrame(PPFrame&&) noexcept = delete;
    PPFrame& operator=(const PPFrame&) = delete;
    PPFrame& operator=(PPFrame&&) noexcept = delete;
    
    void clear();
    
    static std::string class_name() { return "PPFrame"; }
    std::string toStr() const;
    
private:
    void _assume_not_finalized(const char*, int);
    pv::bid _add_ownership(bool regular, pv::BlobPtr&&);
    pv::BlobPtr _extract_from(std::vector<pv::BlobPtr>&& range, pv::bid bdx);
    void _check_owners();
};

}
