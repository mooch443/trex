#pragma once

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#endif

#ifdef __llvm__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wimplicit-float-conversion"
#pragma clang diagnostic ignored "-Wfloat-conversion"
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif

#ifdef WIN32
#include <windows.h>
#endif

#include <ctime>
#include <iomanip>
#include <locale>
#include <type_traits>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <thread>
#include <functional>
#include <memory>
#include <chrono>
#include <set>
#include <unordered_set>
#include <queue>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <future>
#include <ostream>
#include <array>
#include <cstdint>
#include <exception>
#include <stdarg.h>
#include <concepts>

#if defined(__clang__)
    #if (__clang_major__) <= 13

namespace std {

template<class From, class To>
    concept convertible_to =
        std::is_convertible_v<From, To> &&
        requires {
            static_cast<To>(std::declval<From>());
        };
template<class T>
    concept floating_point = std::is_floating_point_v<T>;
template<class T>
    concept integral = std::is_integral_v<T>;
template <class T>
    concept signed_integral = std::integral<T> && std::is_signed_v<T>;
template <class T>
    concept unsigned_integral = std::integral<T> && !std::signed_integral<T>;
}

    #endif
#endif

#ifdef WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

using long_t = int32_t;

#ifndef _MSC_VER
#define CV_STATIC_ANALYSIS 0
#define CV_ErrorNoReturn(code, msg) cv::errorNoReturn( code, msg, CV_Func, __FILE__, __LINE__ )
#endif

#include <opencv2/opencv.hpp>

#include <commons/common/cpputils/cpputils.h>
#include <commons/common/cpputils/debug/Debug.h>
#include <commons/common/cpputils/utilsexception.h>
#include <commons/common/cpputils/debug/Printable.h>
#include <cnpy.h>

#ifdef __llvm__
#pragma clang diagnostic pop
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif
