#pragma once

#include <commons.pc.h>
#include <misc/idx_t.h>
#include <misc/frame_t.h>
#include <tracking/TrackingSettings.h>
#include <tracking/PPFrame.h>
#include <tracking/Stuffs.h>

namespace Output {
class TrackingResults;
class ResultsFormat;
}

namespace track {

class Individual;

// collect all the currently active individuals
class IndividualManager {
    const Frame_t _frame;
    GETTER(set_of_individuals_t, current)
    mutable std::shared_mutex current_mutex;
    
private:
    robin_hood::unordered_flat_set<pv::bid> _blob_assigned;
    robin_hood::unordered_flat_set<Idx_t> _fish_assigned;
    
    mutable std::shared_mutex assign_mutex;
    
protected:
    void clear_blob_assigned() noexcept;
    void clear_fish_assigned() noexcept;
    void _assign(Idx_t, pv::bid);
    
    [[nodiscard]] bool blob_assigned(pv::bid) const;
    [[nodiscard]] bool fish_assigned(Idx_t) const;
    [[nodiscard]] bool fish_assigned(const Individual*) const;
    
    [[nodiscard]] Idx_t id_of_fish(const Individual*) const noexcept;

private:
    std::queue<std::tuple<Individual*, BasicStuff*, pv::BlobPtr>> need_postures;
    std::atomic<size_t> _assigned_count{0u};
    
public:
    size_t assigned_count() const noexcept;
    static void clear_pixels() noexcept;
    
protected:
    void assign_blob_individual(const AssignInfo&, Individual*, pv::BlobPtr&& blob);
public:
    template<typename... Args, template<typename...> class Vector,
             typename F, typename ErrorF>
        requires is_container<Vector<Args...>>::value
              && AnyTransformer<F, pv::bid, Individual*>
              && AnyTransformer<ErrorF, pv::bid, Individual*, const char*>
    void assign_to_inactive(AssignInfo&& info,
                Vector<pv::bid, Args...>& map,
                F&& apply,
                ErrorF&& error)
    {
        //const auto& cref = map;
        //auto blobs = info.frame->extract_from_all(cref);
        //assert(blobs.size() == map.size());
        
        //auto bit = blobs.begin();
        for(auto it = map.begin(); it != map.end();) {
            //auto&& blob = *bit;
            
            std::scoped_lock scoped(_global_mutex(), current_mutex);
            auto result = retrieve_inactive();
            auto bdx = *it;
            
            if(not result) {
                error(bdx, nullptr, result.error());
                
            } else if(   not fish_assigned(result.value())
                      && not blob_assigned(bdx)
                      && info.frame->has_bdx(bdx))
            {
                assign_blob_individual(info, result.value(), info.frame->extract(bdx));
                apply(bdx, result.value());
                
                it = map.erase(it);
                continue; // do not increase iterator
                
            } else {
                error(bdx, result.value(), blob_assigned(bdx) ? "Blob was already assigned." : (fish_assigned(result.value()) ? "Individual was already assigned." : "Frame does not contain bdx."));
            }
            
            ++it;
        }
    }
    
    template<bool safe = true, class Map,
             typename F>
        requires AnyTransformer<F, pv::bid, Idx_t, Individual*>
              && is_map<std::remove_cvref_t<Map>>::value
    void assign(AssignInfo&& info,
                Map&& map,
                F&& apply)
    {
        assign(std::move(info), std::move(map), std::move(apply), [](pv::bid, Idx_t, Individual*, const char*){});
    }
    
    template<bool safe = true, class Map,
             typename F, typename ErrorF>
        requires AnyTransformer<F, pv::bid, Idx_t, Individual*>
              && VoidTransformer<ErrorF, pv::bid, Idx_t, Individual*, const char*>
              && is_map<std::remove_cvref_t<Map>>::value
    void assign(AssignInfo&& info,
                Map&& map,
                F&& apply,
                ErrorF&& error)
    {
        //std::vector<pv::bid> blobs;
        //std::vector<std::tuple<Idx_t, Individual*>> individuals;
        robin_hood::unordered_flat_map<pv::bid, Individual*> blob_map;
        //blobs.reserve(map.size());
        //individuals.reserve(map.size());
        
        decltype(_blob_assigned) assigned_bdx;
        decltype(_fish_assigned) assigned_fdx;
        
        {
            std::shared_lock guard(assign_mutex);
            assigned_bdx = _blob_assigned;
            assigned_fdx = _fish_assigned;
        }
        
        for(auto&& [bdx, fdx] : map) {
            if(not bdx.valid()) {
                error(bdx, fdx, nullptr, "Blob ID invalid.");
                continue; // dont assign this blob
            }
            
            if(not fdx.valid()) {
                error(bdx, fdx, nullptr, "Individual ID invalid.");
                continue; // dont assign this fish
            }
            
            if constexpr(Predicate<F, pv::bid, Idx_t, Individual*>) {
                if(not apply(bdx, fdx, nullptr)) {
                    continue;
                }
            }
            
            //std::scoped_lock scoped(_global_mutex(), assign_mutex, current_mutex);
            auto result = retrieve_globally(fdx);
            
            if(not result) {
                error(bdx, fdx, nullptr, result.error());
                
            } else if(   not assigned_fdx.contains(fdx)
                      && not assigned_bdx.contains(bdx))
                      //&& info.frame->has_bdx(bdx))
            {
                assigned_bdx.insert(bdx);
                assigned_fdx.insert(fdx);
                
                blob_map[bdx] = result.value();
                //blobs.emplace_back(bdx);
                //individuals.emplace_back(fdx, result.value());
                
            } else {
                error(bdx, fdx, result.value(), blob_assigned(bdx) ? "Blob was already assigned." : (fish_assigned(result.value()) ? "Individual was already assigned." : "Frame does not contain bdx."));
            }
        }
        
        if(blob_map.empty())
            return;
        
        std::vector<pv::BlobPtr> ptrs;
        /*if constexpr(safe)
            ptrs = info.frame->extract_from_blobs<PPFrame::VectorHandling::OneToOne>(std::move(blobs));
        else
            ptrs = info.frame->extract_from_blobs_unsafe<PPFrame::VectorHandling::OneToOne>(std::move(blobs));*/
        if constexpr(safe)
            ptrs = info.frame->extract_from_blobs<PPFrame::VectorHandling::Compress, PPFrame::RemoveHandling::Leave>(blob_map);
        else
            ptrs = info.frame->extract_from_blobs_unsafe(blob_map);
        
        /*if(ptrs.size() != individuals.size()) {
            print("Got pointers: ", ptrs, " for map: ", blob_map, " with individuals: ", individuals, " and blobs: ", blobs);
        }
        
        assert(ptrs.size() == individuals.size());*/
        PPFrame::Log("Got pointers: ", ptrs, " for map: ", blob_map);
        for(size_t i=0; i<ptrs.size(); ++i) {
            if(ptrs.at(i) == nullptr)
                continue;
            
            //auto [fdx, fish] = individuals.at(i);
            auto bdx = ptrs.at(i)->blob_id();
            auto fish = blob_map.at(bdx);
            assign_blob_individual(info, fish, std::move(ptrs[i]));
            apply(bdx, id_of_fish(fish), fish);
        }
    }
    
    template<class Vector, typename F, typename ErrorF, typename Tuple_t = typename std::remove_cvref_t<Vector>::value_type>
        requires AnyTransformer<F, pv::bid, Idx_t>
              && VoidTransformer<ErrorF, pv::bid, Idx_t, const char*>
              && is_container<std::remove_cvref_t<Vector>>::value
              && is_tuple_v<Tuple_t>
    void assign(AssignInfo&& info,
                Vector&& vector,
                F&& apply,
                ErrorF&& error)
    {
        const auto& cref = vector;
        auto size = vector.size();
        auto blobs = info.frame->extract_from_all<PPFrame::VectorHandling::OneToOne>(cref);
        assert(blobs.size() == size);
        
        for(size_t i = 0; i<vector.size(); ++i) {
            auto &&tuple = vector[i];
            auto work = [](auto&&... args) -> Idx_t {
                return find_argtype_apply([](Idx_t fdx) {
                    return fdx;
                }, args...);
            };
            
            auto fdx = apply_to_tuple(std::forward<decltype(tuple)>(tuple), work);
            auto bdx = apply_to_tuple(std::forward<decltype(tuple)>(tuple), [](auto&&... args) -> pv::bid {
                return find_argtype_apply([](pv::bid bdx) {
                    return bdx;
                }, args...);
            });
            
            if constexpr(Predicate<F, pv::bid, Idx_t>) {
                if(not apply(bdx, fdx)) {
                    continue;
                }
            }
            
            auto &&blob = blobs[i];
            if(blob == nullptr) {
                error(bdx, fdx, "Cannot find the blob.");
                continue;
            }
            
            assert(bdx.valid());
            
            //std::scoped_lock scoped(_global_mutex(), assign_mutex, current_mutex);
            auto result = retrieve_globally(fdx);
            
            if(not result) {
                error(bdx, fdx, result.error());
                
            } else if(info.frame->has_bdx(bdx)
                      && not fish_assigned(fdx)
                      && not blob_assigned(bdx) )
            {
                assign_blob_individual(info, result.value(), std::move(blob));
                if constexpr(not Predicate<F, pv::bid, Idx_t>)
                    apply(bdx, fdx);
                
            } else {
                error(bdx, result ? fdx : Idx_t(), "Object was not found, or was already assigned.");
            }
        }
    }
    
public:
    using expected_individual_t = tl::expected<Individual*, const char*>;
    
    IndividualManager(const PPFrame&);
    ~IndividualManager();
    
    IndividualManager(const IndividualManager&) = delete;
    IndividualManager& operator=(const IndividualManager&) = delete;
    IndividualManager(IndividualManager&&) = delete;
    IndividualManager& operator=(IndividualManager&&) = delete;
    
protected:
    [[nodiscard]] bool is_active(Individual*) const noexcept;
    [[nodiscard]] bool is_inactive(Individual*) const noexcept;
    
    // only these two classes have access to these functions
    // since only they are allowed to guide the tracking process.
    // all other methods are simply "getters" and modifiers for
    // an existing tracking state.
    friend class Tracker;
    friend struct TrackingHelper;
    
    //! if possible (i.e. when `track_max_individuals == 0`)
    //! retrieves a currently unused individual and returns it
    expected_individual_t retrieve_inactive(Idx_t = {}) noexcept;
    
    //! tries to find an individual globally somewhere, and returns a pointer
    //! if it exists, as well as setting it to "active":
    expected_individual_t retrieve_globally(Idx_t) noexcept;
    
    void become_active(Individual*);
    
public:
    [[nodiscard]] static tl::expected<set_of_individuals_t*, const char*> active_individuals(Frame_t) noexcept;
    
    //! delete callback is called for each deleted individual, right before it is deleted
    static void remove_frames(Frame_t from, std::function<void(Individual*)>&& delete_callback = nullptr);
    static void clear() noexcept;
    
    [[nodiscard]] static expected_individual_t individual_by_id(Idx_t) noexcept;
    [[nodiscard]] static bool has_individual(Idx_t) noexcept;
    [[nodiscard]] static size_t num_individuals() noexcept;
    
    [[nodiscard]] static std::set<Idx_t> all_ids() noexcept;
    [[nodiscard]] static std::unordered_map<Idx_t, Individual*> copy() noexcept;
    
    template<typename F>
        requires Transformer<F, Individual*>
    static void transform_inactive(F&& fn) {
        //std::shared_lock im(_individual_mutex(), std::defer_lock),
        //                 gm(_global_mutex(), std::defer_lock);
        //std::scoped_lock slock(_individual_mutex(), _global_mutex());
        for(auto &[id, fish] : _inactive()) {
            if constexpr(Predicate<F, Individual*>) {
                if(not fn(fish))
                    break;
            } else {
                fn(fish);
            }
        }
    }
    
    template<typename F>
        requires Transformer<F, Individual*>
    void transform_active(F&& fn) const {
        //std::shared_lock im(_individual_mutex(), std::defer_lock),
        //                 gm(_global_mutex(), std::defer_lock);
        //std::scoped_lock slock(_individual_mutex(), _global_mutex());
        for(auto fish : _current) {
            if constexpr(Predicate<F, Individual*>) {
                if(not fn(fish))
                    break;
            } else {
                fn(fish);
            }
        }
    }
    
    template<typename F, typename R = std::invoke_result_t<F, Individual*>>
        requires AnyTransformer<F, Individual*>
    static auto transform_if_exists(Idx_t fdx, F&& fn) -> tl::expected<R, const char*> {
        //std::scoped_lock slock(_global_mutex(), _individual_mutex());
        auto it = individuals().find(fdx);
        if(it == individuals().end()) {
            return tl::unexpected("Cannot find individual ID.");
        }
        
        if constexpr(_clean_same<R, void>) {
            fn(it->second.get());
            return {};
        } else
            return fn(it->second.get());
    }
    
    template<typename F>
        requires Transformer<F, Idx_t, Individual*>
    static auto transform_all(F&& fn) {
        //std::scoped_lock slock(_global_mutex(), _individual_mutex());
        for(const auto& [fdx, fish] : individuals()) {
            if constexpr(Predicate<F, Idx_t, Individual*>) {
                if(not fn(fdx, fish.get())) {
                    return false;
                }
            } else {
                fn(fdx, fish.get());
            }
        }
        
        if constexpr(Predicate<F, Idx_t, Individual*>)
            return true;
    }
    
    template<typename F>
        requires Transformer<F, Idx_t, Individual*>
    static void transform_parallel(GenericThreadPool& pool, F&& fn) {
        //std::scoped_lock slock(_global_mutex(), _individual_mutex());
        distribute_indexes([&](auto, auto start, auto end, auto){
            for(auto it = start; it != end; ++it)
                std::invoke(std::forward<F>(fn), it->first, it->second.get());
        }, pool, individuals().begin(), individuals().end());
    }
    
    template<typename Key, typename Value, typename F, typename ErrorF, typename Map>   requires Transformer<F, Key, Value, Individual*>
               && VoidTransformer<ErrorF, Key, Value>
    static void _transform_ids_with_error(Map&& ids, F&& fn, ErrorF&& error) {
        for(const auto &[id, value] : ids) {
            //std::scoped_lock slock(_global_mutex(), _individual_mutex());
            individuals_map_t::const_iterator it;
            if constexpr(_clean_same<Idx_t, Key>)
                it = individuals().find(id);
            else
                it = individuals().find(value);
            //auto it = individuals().find(id);
            if(it != individuals().end()) {
                if constexpr(Predicate<F, Key, Value, Individual*>) {
                    if(not std::invoke(std::forward<F>(fn),
                                   std::forward<decltype(id)>(id),
                                   std::forward<decltype(value)>(value),
                                   it->second.get()))
                        break;
                } else {
                    std::invoke(std::forward<F>(fn),
                                std::forward<decltype(id)>(id),
                                std::forward<decltype(value)>(value),
                                it->second.get());
                }
            } else {
                std::invoke(std::forward<ErrorF>(error),
                            std::forward<decltype(id)>(id),
                            std::forward<decltype(value)>(value));
            }
        }
    }
    
    template<typename F, typename ErrorF>
        requires   Transformer<F, Idx_t, Individual*>
                && VoidTransformer<ErrorF, Idx_t>
    static void transform_ids_with_error(const set_or_container auto& ids, F&& fn, ErrorF&& error) {
        for(const auto &id : ids) {
            //std::scoped_lock slock(_global_mutex(), _individual_mutex());
            auto it = individuals().find(id);
            if(it != individuals().end()) {
                if constexpr(Predicate<F, Idx_t, Individual*>) {
                    if(not fn(std::forward<decltype(id)>(id),
                          it->second.get()))
                        break;
                } else {
                    fn(std::forward<decltype(id)>(id),
                       it->second.get());
                }
            } else {
                error(std::forward<decltype(id)>(id));
            }
        }
    }

    template<typename F, typename ErrorF, bool T, size_t S, typename K, typename V, typename Hash, typename KeyEqual>
    static void transform_ids_with_error(const robin_hood::detail::Table<T, S, K, V, Hash, KeyEqual>& ids, F&& fn, ErrorF&& error)
    {
        _transform_ids_with_error<K, V, F, ErrorF>(
             std::forward<decltype(ids)>(ids),
             std::forward<F>(fn),
             std::forward<ErrorF>(error));
    }
    
    template<typename F, typename Key, typename Value, typename... FurtherArgs, template <typename, typename, typename, typename...> class Map, typename ErrorF>
        requires map_type<Map<Key, Value, FurtherArgs...>>
              && Transformer<F, Key, Value, Individual*>
              && VoidTransformer<ErrorF, Key, Value>
    static void transform_ids_with_error(const Map<Key, Value, FurtherArgs...>& ids, F&& fn, ErrorF&& error) {
        _transform_ids_with_error<Key, Value>(
            std::forward<decltype(ids)>(ids),
            std::forward<F>(fn),
            std::forward<ErrorF>(error));
    }
    
    static void transform_ids(set_or_container auto&& ids, auto&& fn) {
        if constexpr( is_tuple_v< typename std::remove_reference_t< decltype(ids)>::value_type >)
        {
            for(auto &&tuple : ids) {
                //std::scoped_lock slock(_global_mutex(), _individual_mutex());
                auto get_id = [&fn](auto&&... args) mutable {
                    auto fish = find_argtype_apply([](Idx_t id) -> Individual* {
                        auto it = individuals().find(id);
                        if(it != individuals().end()) {
                            return it->second.get();
                            
                        } else {
                            return nullptr;
                        }
                        
                    }, std::forward<decltype(args)>(args)...);
                    
                    if(not fish) {
                        throw U_EXCEPTION("Cannot find object for tuple ", args..., " in map of individuals: ", extract_keys(individuals()));
                    }
                    fn(std::forward<decltype(args)>(args)..., fish);
                };
                
                apply_to_tuple(std::forward<decltype(tuple)>(tuple), get_id);
            }
        }
    }
    
    static void transform_ids(const set_or_container auto& ids, auto&& fn) {
        if constexpr( is_tuple_v< typename std::remove_reference_t< decltype(ids)>::value_type >)
        {
            for(auto &&tuple : ids) {
                //std::scoped_lock slock(_global_mutex(), _individual_mutex());
                auto get_id = [&fn](auto... args) {
                    auto fish = find_argtype_apply([](Idx_t id) -> Individual* {
                        auto it = individuals().find(id);
                        if(it != individuals().end()) {
                            /*if constexpr(Predicate<F, Idx_t, Individual*>) {
                                if(not fn(std::forward<decltype(id)>(id),
                                      it->second.get()))
                                    break;
                            } else {*/
                            //}
                            return it->second.get();
                            
                        } else {
                            return nullptr;
                        }
                        
                    }, std::forward<decltype(args)>(args)...);
                    
                    if(not fish) {
                        throw U_EXCEPTION("Cannot find object for tuple ", args..., " in map of individuals: ", extract_keys(individuals()));
                    }
                    fn(std::forward<decltype(args)>(args)..., fish);
                };
                
                apply_to_tuple(std::forward<decltype(tuple)>(tuple), get_id);
            }
            
        } else {
            transform_ids_with_error(
                 std::forward<decltype(ids)>(ids),
                 std::forward<decltype(fn)>(fn),
                 [](auto&& id) {
                     throw U_EXCEPTION("Cannot find object with id ", id, " in map of individuals: ", extract_keys(individuals()));
                 });
        }
    }
    
    static void transform_ids(const map_type auto& ids, auto&& fn) {
        transform_ids_with_error(
            std::forward<decltype(ids)>(ids),
            std::forward<decltype(fn)>(fn),
            [](auto&& id, auto&&) {
                throw U_EXCEPTION("Cannot find object with id ", id, " in map of individuals: ", extract_keys(individuals()));
            });
    }
    
private:
    //! called when an individual is not assigned in the current
    //! frame, but was assigned in the previous frame
    friend class Output::TrackingResults;
    friend class Output::ResultsFormat;
    friend class Tracker;
    
    [[nodiscard]] static active_individuals_map_t& _all_frames();
    [[nodiscard]] static set_of_individuals_t*& _last_active();
    [[nodiscard]] static inactive_individuals_t& _inactive();
    [[nodiscard]] static Individual* make_individual(Idx_t);
    [[nodiscard]] static const individuals_map_t& individuals();
    [[nodiscard]] static std::shared_mutex& _individual_mutex();
    [[nodiscard]] static std::shared_mutex& _global_mutex();
    
    struct Protect {
        std::scoped_lock<std::shared_mutex, std::shared_mutex> guard;
        Protect() : guard(_individual_mutex(), _global_mutex()) {
            
        }
    };
};

}
