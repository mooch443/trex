#pragma once

#include <commons.pc.h>
#include <misc/idx_t.h>
#include <misc/frame_t.h>
#include <tracking/TrackingSettings.h>
#include <tracking/PPFrame.h>

namespace Output {
class TrackingResults;
class ResultsFormat;
}

namespace track {

template<typename T>
concept set_or_container = container_type<T> || set_type<T>;

template <typename T>
struct is_tuple final {
    using type = std::false_type;
};
template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> final {
using type = std::true_type;
};

template <typename T>
using is_tuple_t = typename is_tuple<T>::type;

template <typename T>
constexpr auto is_tuple_v = is_tuple_t<T>{};

class Individual;

// collect all the currently active individuals
class IndividualManager {
    const Frame_t _frame;
    GETTER(set_of_individuals_t, current)
    mutable std::shared_mutex current_mutex;
    
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
    
    void become_active(Idx_t);
    
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
        std::scoped_lock slock(_individual_mutex(), _global_mutex());
        for(auto fish : _inactive()) {
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
        std::scoped_lock slock(_individual_mutex(), _global_mutex());
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
        std::scoped_lock slock(_global_mutex(), _individual_mutex());
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
    
    template<typename F, typename R = std::conditional_t<Predicate<F, Idx_t, Individual*>, bool, void>>
        requires Transformer<F, Idx_t, Individual*>
    static R transform_all(F&& fn) {
        std::scoped_lock slock(_global_mutex(), _individual_mutex());
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
        std::scoped_lock slock(_global_mutex(), _individual_mutex());
        distribute_indexes([&](auto, auto start, auto end, auto){
            for(auto it = start; it != end; ++it)
                std::invoke(std::forward<F>(fn), it->first, it->second.get());
        }, pool, individuals().begin(), individuals().end());
    }
    
    template<typename Key, typename Value, typename F, typename ErrorF, typename Map>   requires Transformer<F, Key, Value, Individual*>
               && VoidTransformer<ErrorF, Key, Value>
    static void _transform_ids_with_error(Map&& ids, F&& fn, ErrorF&& error) {
        for(const auto &[id, value] : ids) {
            std::scoped_lock slock(_global_mutex(), _individual_mutex());
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
            std::scoped_lock slock(_global_mutex(), _individual_mutex());
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
                std::scoped_lock slock(_global_mutex(), _individual_mutex());
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
                std::scoped_lock slock(_global_mutex(), _individual_mutex());
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
    [[nodiscard]] static std::vector<Individual*>& _inactive();
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
