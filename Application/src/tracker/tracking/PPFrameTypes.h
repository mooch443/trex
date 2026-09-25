#pragma once

#include <commons.pc.h>

namespace track {

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

enum class NeedGrid {
    Need,
    NoNeed
};

}
