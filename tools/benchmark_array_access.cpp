// Standalone benchmark: copied commons helpers, ICU, no TRex build or layouts.
// From Application/build with the trex Conda environment active:
//   xcrun clang++ -std=c++23 -O3 -g -DNDEBUG -fno-lto -fno-omit-frame-pointer -I"$CONDA_PREFIX/include" -c ../../tools/benchmark_array_access.cpp -o benchmark_array_access.o
//   xcrun clang++ benchmark_array_access.o -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" -licuuc -o benchmark_array_access.out
//   xcrun dsymutil benchmark_array_access.out -o benchmark_array_access.out.dSYM
//   ./benchmark_array_access.out > array_access.csv
// Optional first argument: --old-only or --new-only for profiling one accessor.
// Then: milliseconds per sample (default 10), rounds (default 7).
//   ./benchmark_array_access.out --new-only 100 7 > array_access_new.csv
// Single-accessor modes validate, calibrate, and measure only that accessor.
// Whole-parser comparison: --parse-only; profiling: --parse-head-only or --parse-new-only.
//   ./benchmark_array_access.out --parse-only 20 7 > array_parts.csv
// Parser modes measure complete vector construction/destruction, once per fixture.
// Their CSV rows use index=-1; ns_per_lookup is nanoseconds per complete parse.
//
// HEAD baseline models commons d02acf3: split all tokens into string_views using
// std::stack<char>, then select one. The other variants independently change
// token selection and nesting storage. These are frozen, simplified kernels:
// original commons ICU trimming, valid indices, no GUI resolution, decoding, or HTML handling.
// "copies" also copies the input and selected output on every lookup.
// "access_array_part" separately compares the complete old (std::stack) and new
// (std::string with cached top) accessor functions with string_view inputs and results.
// The old accessor and concept kernels use frozen old_accessor::utils helpers.
// Optimize cmn::utils and new_accessor together for the complete new implementation.
// "parse_array_parts" compares the complete commons d02acf3 HEAD parser and
// new_parser, using the original and optimized helpers respectively.

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stack>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <unicode/uchar.h>
#include <unicode/utf8.h>

using Clock = std::chrono::steady_clock;
volatile std::uint64_t checksum_sink = 0;

struct StringStack {
    std::string data;
    bool empty() const { return data.empty(); }
    char top() const { return data.back(); }
    void push(char c) { data.push_back(c); }
    void pop() { data.pop_back(); }
};

// Standalone copies of the commons helpers used by new_accessor. View inputs
// retain view results, including when trimming Unicode whitespace.
namespace cmn {
template<typename Str>
concept StringLike = std::is_same_v<std::remove_cvref_t<Str>, std::string> ||
                      std::is_same_v<std::remove_cvref_t<Str>, const char*> ||
                      std::is_same_v<std::remove_cvref_t<Str>, std::string_view> ||
                      std::is_array_v<std::remove_cvref_t<Str>>;

template<typename T, typename U>
concept _clean_same = std::same_as<T, typename std::remove_cv<U>::type>;

namespace utils {
template<typename T>
struct is_const_lvalue_ref : std::false_type {};

template<typename T>
struct is_const_lvalue_ref<const T&> : std::true_type {};

template<StringLike Str>
[[nodiscard]] constexpr std::string_view string_like_view(const Str& str) noexcept {
    using StrDecayed = std::remove_cvref_t<Str>;

    if constexpr(std::is_same_v<StrDecayed, const char*>) {
        return str == nullptr ? std::string_view{} : std::string_view{str};
    } else if constexpr(std::is_array_v<StrDecayed>) {
        static_assert(std::is_same_v<std::remove_cv_t<std::remove_extent_t<StrDecayed>>, char>,
                      "StringLike arrays must be char arrays.");
        constexpr auto extent = std::extent_v<StrDecayed>;
        size_t len = 0u;
        for(; len < extent && str[len] != '\0'; ++len);
        return std::string_view(str, len);
    } else {
        static_assert(_clean_same<StrDecayed, std::string> || _clean_same<StrDecayed, std::string_view>);
        return std::string_view{str};
    }
}

inline bool isWhitespaceOrAdditional(UChar32 c) {
    if (static_cast<uint32_t>(c) < 0x80)
        return c == ' ' || (c >= '\t' && c <= '\r');
    
    return u_hasBinaryProperty(c, UCHAR_WHITE_SPACE) ||
        c == 0x200B || // ZERO WIDTH SPACE
        c == 0x2060 || // WORD JOINER
        c == 0xFEFF;   // ZERO WIDTH NO-BREAK SPACE (BOM)
}

enum class TrimDirection { Left, Right };

template<TrimDirection Direction>
auto trim_impl(StringLike auto&& s) {
    using Str = decltype(s);
    using StrDecayed = std::remove_cvref_t<Str>;
    auto sv = utils::string_like_view(s);

    if (sv.empty()) {
        if constexpr (std::is_same_v<StrDecayed, std::string>) {
            return std::forward<decltype(s)>(s);
        } else {
            return std::string_view{};
        }
    }

    const char* data = sv.data();
    int32_t length = static_cast<int32_t>(sv.size());
    int32_t start = 0;
    int32_t end = length;

    if constexpr (Direction == TrimDirection::Left) {
        // Left trim
        int32_t offset = start;
        while (offset < end) {
            UChar32 c;
            U8_NEXT(data, offset, end, c);
            if (c < 0) {
                // Invalid character, stop trimming
                break;
            }
            if (isWhitespaceOrAdditional(c)) {
                // Whitespace character, move start forward
                start = offset;
            } else {
                // Non-whitespace character found, stop
                break;
            }
        }
    } else {
        // Right trim
        int32_t offset = end;
        while (offset > start) {
            UChar32 c;
            U8_PREV(data, start, offset, c);
            if (c < 0) {
                // Invalid character, stop trimming
                break;
            }
            if (isWhitespaceOrAdditional(c)) {
                // Whitespace character, move end backward
                end = offset;
            } else {
                // Non-whitespace character found, stop
                break;
            }
        }
    }

    if constexpr ((std::is_rvalue_reference_v<Str&&> ||
                   is_const_lvalue_ref<Str>::value) &&
                  std::is_same_v<StrDecayed, std::string>) {
        // Create a new string and return it
        return std::string(data + start, end - start);
    } else if constexpr (std::is_same_v<StrDecayed, std::string>) {
        // Modify the original string in place
        s.erase(end, length - end);      // Erase from 'end' to the end
        s.erase(0, start);               // Erase from the start to 'start'
        return std::forward<Str>(s);
    } else {
        // For string_view and const char*
        if (start >= end) {
            return std::string_view{};
        }
        return std::string_view(data + start, end - start);
    }
}

auto ltrim(StringLike auto && s) {
    return trim_impl<TrimDirection::Left>(std::forward<decltype(s)>(s));
}

auto rtrim(StringLike auto && s) {
    return trim_impl<TrimDirection::Right>(std::forward<decltype(s)>(s));
}

auto trim(StringLike auto && s) {
    return rtrim(ltrim(std::forward<decltype(s)>(s)));
}

} // namespace utils

template<typename Str>
concept is_viewable = std::is_array_v<std::remove_cvref_t<Str>>
        || std::is_pointer_v<std::remove_cvref_t<Str>>
        || std::same_as<std::remove_cvref_t<Str>, std::string_view>
        || (std::is_lvalue_reference_v<Str> &&
            _clean_same<std::remove_reference_t<Str>, std::string>);
} // namespace cmn

// The baseline helpers are isolated from all candidate helper optimizations.
namespace old_accessor {
using cmn::is_viewable;

namespace utils {
using cmn::StringLike;
using cmn::_clean_same;

template<typename T>
struct is_const_lvalue_ref : std::false_type {};

template<typename T>
struct is_const_lvalue_ref<const T&> : std::true_type {};

template<StringLike Str>
[[nodiscard]] constexpr std::string_view string_like_view(const Str& str) noexcept {
    using StrDecayed = std::remove_cvref_t<Str>;

    if constexpr(std::is_same_v<StrDecayed, const char*>) {
        return str == nullptr ? std::string_view{} : std::string_view{str};
    } else if constexpr(std::is_array_v<StrDecayed>) {
        static_assert(std::is_same_v<std::remove_cv_t<std::remove_extent_t<StrDecayed>>, char>,
                      "StringLike arrays must be char arrays.");
        constexpr auto extent = std::extent_v<StrDecayed>;
        size_t len = 0u;
        for(; len < extent && str[len] != '\0'; ++len);
        return std::string_view(str, len);
    } else {
        static_assert(_clean_same<StrDecayed, std::string> || _clean_same<StrDecayed, std::string_view>);
        return std::string_view{str};
    }
}

inline bool isWhitespaceOrAdditional(UChar32 c) {
    return u_hasBinaryProperty(c, UCHAR_WHITE_SPACE) ||
        c == 0x200B || // ZERO WIDTH SPACE
        c == 0x2060 || // WORD JOINER
        c == 0xFEFF;   // ZERO WIDTH NO-BREAK SPACE (BOM)
}

enum class TrimDirection { Left, Right };

template<TrimDirection Direction>
auto trim_impl(StringLike auto&& s) {
    using Str = decltype(s);
    using StrDecayed = std::remove_cvref_t<Str>;
    auto sv = utils::string_like_view(s);

    if (sv.empty()) {
        if constexpr (std::is_same_v<StrDecayed, std::string>) {
            return std::forward<decltype(s)>(s);
        } else {
            return std::string_view{};
        }
    }

    const char* data = sv.data();
    int32_t length = static_cast<int32_t>(sv.size());
    int32_t start = 0;
    int32_t end = length;

    if constexpr (Direction == TrimDirection::Left) {
        // Left trim
        int32_t offset = start;
        while (offset < end) {
            UChar32 c;
            U8_NEXT(data, offset, end, c);
            if (c < 0) {
                // Invalid character, stop trimming
                break;
            }
            if (isWhitespaceOrAdditional(c)) {
                // Whitespace character, move start forward
                start = offset;
            } else {
                // Non-whitespace character found, stop
                break;
            }
        }
    } else {
        // Right trim
        int32_t offset = end;
        while (offset > start) {
            UChar32 c;
            U8_PREV(data, start, offset, c);
            if (c < 0) {
                // Invalid character, stop trimming
                break;
            }
            if (isWhitespaceOrAdditional(c)) {
                // Whitespace character, move end backward
                end = offset;
            } else {
                // Non-whitespace character found, stop
                break;
            }
        }
    }

    if constexpr ((std::is_rvalue_reference_v<Str&&> ||
                   is_const_lvalue_ref<Str>::value) &&
                  std::is_same_v<StrDecayed, std::string>) {
        // Create a new string and return it
        return std::string(data + start, end - start);
    } else if constexpr (std::is_same_v<StrDecayed, std::string>) {
        // Modify the original string in place
        s.erase(end, length - end);      // Erase from 'end' to the end
        s.erase(0, start);               // Erase from the start to 'start'
        return std::forward<Str>(s);
    } else {
        // For string_view and const char*
        if (start >= end) {
            return std::string_view{};
        }
        return std::string_view(data + start, end - start);
    }
}

auto ltrim(StringLike auto && s) {
    return trim_impl<TrimDirection::Left>(std::forward<decltype(s)>(s));
}

auto rtrim(StringLike auto && s) {
    return trim_impl<TrimDirection::Right>(std::forward<decltype(s)>(s));
}

auto trim(StringLike auto && s) {
    return rtrim(ltrim(std::forward<decltype(s)>(s)));
}

} // namespace utils

auto access_array_part(cmn::StringLike auto&& str, size_t index,
                       const char delimiter = ',')
{
    using Str = decltype(str);

    // ---- choose element type ------------------------------------------------
    // We can safely hand out string_views only when the caller guarantees
    // that the underlying buffer will out‑live the views.  That is true for
    //   * character arrays / literals
    //   * raw C‑strings
    //   * std::string_view itself
    //   * *l‑value* std::string objects
    // Everything else (temporary std::string, std::string&& …) requires copies.
    static constexpr bool _safe_to_view =
        is_viewable<Str>;

    using Return_t = std::conditional_t<_safe_to_view, std::string_view, std::string>;

    // ---- split implementation ----------------------------------------------
    std::stack<char> brackets;
    char   prev        = 0;
    size_t token_start = 0;
    size_t current_index = 0;

    std::string_view sv = utils::string_like_view(std::forward<Str>(str));
    if(sv.empty())
        throw std::invalid_argument("Trying to access index into an empty string.");

    const auto flush_token = [&](size_t end_index) {
        return utils::trim(
            std::string_view(sv.data() + token_start, end_index - token_start));
    };

    for (size_t i = 0; i < sv.size(); ++i) {
        char c = sv[i];
        bool in_string = !brackets.empty() &&
                         (brackets.top() == '\'' || brackets.top() == '"');

        if (in_string) {
            if (prev != '\\' && c == brackets.top())
                brackets.pop();
        } else {
            switch (c) {
                case '(': case '[': case '{': case '"': case '\'':
                    brackets.push(c); break;
                case ']':
                    if (!brackets.empty() && brackets.top() == '[') brackets.pop();
                    break;
                case '}':
                    if (!brackets.empty() && brackets.top() == '{') brackets.pop();
                    break;
                case ')':
                    if (!brackets.empty() && brackets.top() == '(') brackets.pop();
                    break;
            }

            if (brackets.empty() && c == delimiter) {
                if(index == current_index) {
                    return Return_t( flush_token(i) );
                } else {
                    ++current_index;
                }
                token_start = i + 1;
            }
        }

        prev = (prev == '\\' && c == '\\') ? 0 : c;
    }

    // final token (handles trailing‑delimiter case, too)
    if(current_index == index) {
        return Return_t( flush_token(sv.size()) );
    } else {
        ++current_index;
    }

    throw std::invalid_argument("Array index out of bounds in access_array_part");
}
} // namespace old_accessor

using old_accessor::utils::trim;

namespace new_accessor {
using cmn::is_viewable;
namespace utils = cmn::utils;

template<typename First, typename ... T>
constexpr bool is_in(First &&first, T && ... t)
{
    return ((first == t) || ...);
}

constexpr std::string_view flush_token(const char* token_start, const char* end_index) {
    return utils::trim(std::string_view(token_start, end_index - token_start));
}

template<const char delimiter = ','>
    /// we are assuming this delimiter is not one of the bracket or quote characters,
    /// otherwise future optimizations might break, and its also weird
    requires (not is_in(delimiter, '{', '}', '(', ')', '[', ']', '\'', '"'))
constexpr auto access_array_part(cmn::StringLike auto&& str, size_t index)
{
    using Str = decltype(str);

    // ---- choose element type ------------------------------------------------
    // We can safely hand out string_views only when the caller guarantees
    // that the underlying buffer will out‑live the views.  That is true for
    //   * character arrays / literals
    //   * raw C‑strings
    //   * std::string_view itself
    //   * *l‑value* std::string objects
    // Everything else (temporary std::string, std::string&& …) requires copies.
    static constexpr bool _safe_to_view =
        is_viewable<Str>;

    using Return_t = std::conditional_t<_safe_to_view, std::string_view, std::string>;

    // ---- split implementation ----------------------------------------------
    std::string brackets;
    char last_bracket = 0;
    char   prev        = 0;
    size_t current_index = 0;
    std::string_view sv = utils::string_like_view(std::forward<Str>(str));
    const size_t N = sv.size();
    const char* token_start = sv.data();

    if(N == 0)
        throw std::invalid_argument("Trying to access index into an empty string.");

    const char* end_ptr = sv.data() + N;
    for (auto ptr = sv.data(); ptr < end_ptr; ++ptr) {
    //for (size_t i = 0; i < N; ++i) {
        //char c = sv[i];
        char c = *ptr;
        /*bool in_string = !brackets.empty() &&
                         (brackets.back() == '\'' || brackets.back() == '"');*/
        bool in_string = last_bracket
                            && is_in(last_bracket, '\'', '"');

        if (in_string) {
            if (prev != '\\'
                && c == last_bracket)
            {
                brackets.pop_back();
                last_bracket = brackets.empty() ? 0 : brackets.back();
            }
            
        } else if (c == delimiter) {
            if(last_bracket == 0) {
                if(index == current_index) {
                    return Return_t( flush_token(token_start, ptr) );
                } else {
                    ++current_index;
                }
                token_start = ptr + 1;
            }
            
        } else if(is_in(c, '(', '[', '{', '"', '\'')) {
            char opposite = c;
            if(c == '[') { opposite = ']'; }
            else if(c == '{') { opposite = '}'; }
            else if(c == '(') { opposite = ')'; }
            
            brackets.push_back(opposite);
            last_bracket = opposite;
            
        } else if(c == last_bracket && c) {
            brackets.pop_back();
            last_bracket = brackets.empty() ? 0 : brackets.back();
        }

        prev = (prev == '\\' && c == '\\') ? 0 : c;
    }

    // final token (handles trailing‑delimiter case, too)
    if(current_index == index) {
        return Return_t( flush_token(token_start, end_ptr) );
    } else {
        ++current_index;
    }

    throw std::invalid_argument("Array index out of bounds in access_array_part");
}
} // namespace new_accessor

namespace head_parser {
using cmn::is_viewable;
namespace utils = old_accessor::utils;

auto parse_array_parts(cmn::StringLike auto&& str,
                       const char delimiter = ',')
{
    using Str = decltype(str);

    // ---- choose element type ------------------------------------------------
    // We can safely hand out string_views only when the caller guarantees
    // that the underlying buffer will out‑live the views.  That is true for
    //   * character arrays / literals
    //   * raw C‑strings
    //   * std::string_view itself
    //   * *l‑value* std::string objects
    // Everything else (temporary std::string, std::string&& …) requires copies.
    static constexpr bool _safe_to_view =
        is_viewable<Str>;

    using Return_t = std::conditional_t<_safe_to_view, std::string_view, std::string>;

    // ---- split implementation ----------------------------------------------
    std::vector<Return_t> ret;
    std::stack<char> brackets;
    char   prev        = 0;
    size_t token_start = 0;

    std::string_view sv = utils::string_like_view(std::forward<Str>(str));
    if(sv.empty())
        return ret;

    const auto flush_token = [&](size_t end_index) {
        auto token_view = utils::trim(
            std::string_view(sv.data() + token_start, end_index - token_start));
        if constexpr (_safe_to_view)
            ret.emplace_back(token_view);
        else
            ret.emplace_back(std::string(token_view));
    };

    for (size_t i = 0; i < sv.size(); ++i) {
        char c = sv[i];
        bool in_string = !brackets.empty() &&
                         (brackets.top() == '\'' || brackets.top() == '"');

        if (in_string) {
            if (prev != '\\' && c == brackets.top())
                brackets.pop();
        } else {
            switch (c) {
                case '(': case '[': case '{': case '"': case '\'':
                    brackets.push(c); break;
                case ']':
                    if (!brackets.empty() && brackets.top() == '[') brackets.pop();
                    break;
                case '}':
                    if (!brackets.empty() && brackets.top() == '{') brackets.pop();
                    break;
                case ')':
                    if (!brackets.empty() && brackets.top() == '(') brackets.pop();
                    break;
            }

            if (brackets.empty() && c == delimiter) {
                flush_token(i);
                token_start = i + 1;
            }
        }

        prev = (prev == '\\' && c == '\\') ? 0 : c;
    }

    // final token (handles trailing‑delimiter case, too)
    flush_token(sv.size());

    return ret;
}
} // namespace head_parser

namespace new_parser {
using cmn::is_viewable;
using new_accessor::is_in;
namespace utils = cmn::utils;

template<const char delimiter = ','>
    /// we are assuming this delimiter is not one of the bracket or quote characters,
    /// otherwise future optimizations might break, and its also weird
    requires (not is_in(delimiter, '{', '}', '(', ')', '[', ']', '\'', '"'))
constexpr auto parse_array_parts(cmn::StringLike auto&& str)
{
    using Str = decltype(str);

    // ---- choose element type ------------------------------------------------
    // We can safely hand out string_views only when the caller guarantees
    // that the underlying buffer will out‑live the views.  That is true for
    //   * character arrays / literals
    //   * raw C‑strings
    //   * std::string_view itself
    //   * *l‑value* std::string objects
    // Everything else (temporary std::string, std::string&& …) requires copies.
    static constexpr bool _safe_to_view =
        is_viewable<Str>;

    using Return_t = std::conditional_t<_safe_to_view, std::string_view, std::string>;

    // ---- split implementation ----------------------------------------------
    std::vector<Return_t> ret;
    std::string brackets;
    char last_bracket = 0;
    char   prev        = 0;
    std::string_view sv = utils::string_like_view(std::forward<Str>(str));
    const size_t N = sv.size();
    const char* token_start = sv.data();

    if(sv.empty())
        return ret;

    static constexpr auto flush_token = [](std::vector<Return_t>& ret, const char* token_start, const char* end_index) {
        auto token_view = utils::trim( std::string_view(token_start, end_index - token_start) );
        if constexpr (_safe_to_view)
            ret.emplace_back(token_view);
        else
            ret.emplace_back(std::string(token_view));
    };

    const char* end_ptr = sv.data() + N;
    for (auto ptr = sv.data(); ptr < end_ptr; ++ptr) {
    //for (size_t i = 0; i < N; ++i) {
        //char c = sv[i];
        char c = *ptr;
        /*bool in_string = !brackets.empty() &&
                         (brackets.back() == '\'' || brackets.back() == '"');*/
        bool in_string = last_bracket
                            && is_in(last_bracket, '\'', '"');

        if (in_string) {
            if (prev != '\\'
                && c == last_bracket)
            {
                brackets.pop_back();
                last_bracket = brackets.empty() ? 0 : brackets.back();
            }
            
        } else if (c == delimiter) {
            if(last_bracket == 0) {
                flush_token(ret, token_start, ptr);
                token_start = ptr + 1;
            }
            
        } else if(is_in(c, '(', '[', '{', '"', '\'')) {
            char opposite = c;
            if(c == '[') { opposite = ']'; }
            else if(c == '{') { opposite = '}'; }
            else if(c == '(') { opposite = ')'; }
            
            brackets.push_back(opposite);
            last_bracket = opposite;
            
        } else if(c && c == last_bracket) {
            brackets.pop_back();
            last_bracket = brackets.empty() ? 0 : brackets.back();
        }

        prev = (prev == '\\' && c == '\\') ? 0 : c;
    }

    // final token (handles trailing‑delimiter case, too)
    flush_token(ret, token_start, end_ptr);

    return ret;
}
} // namespace new_parser



template<typename Stack, bool Accessor>
std::string_view select_part(std::string_view text, std::size_t index) {
    Stack brackets;
    std::vector<std::string_view> parts;
    char prev = 0;
    std::size_t start = 0, current = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        const bool quoted = !brackets.empty()
            && (brackets.top() == '\'' || brackets.top() == '"');
        if (quoted) {
            if (prev != '\\' && c == brackets.top())
                brackets.pop();
        } else {
            switch (c) {
                case '(': case '[': case '{': case '\'': case '"':
                    brackets.push(c); break;
                case ')':
                    if (!brackets.empty() && brackets.top() == '(') brackets.pop();
                    break;
                case ']':
                    if (!brackets.empty() && brackets.top() == '[') brackets.pop();
                    break;
                case '}':
                    if (!brackets.empty() && brackets.top() == '{') brackets.pop();
                    break;
            }
            if (c == ',' 
                && brackets.empty()) 
            {
                if constexpr (Accessor) {
                    if (current == index)
                        return trim(std::string_view(text.data() + start, i - start));
                } else {
                    parts.push_back(trim(std::string_view(text.data() + start, i - start)));
                }
                ++current;
                start = i + 1;
            }
        }
        prev = (prev == '\\' && c == '\\') ? 0 : c;
    }
    if constexpr (Accessor) {
        if (!text.empty() && current == index)
            return trim(std::string_view(text.data() + start, text.size() - start));
        throw std::out_of_range("array index");
    } else {
        if (!text.empty())
            parts.push_back(trim(std::string_view(text.data() + start, text.size() - start)));
        return parts.at(index);
    }
}

// GCC/Clang compiler barrier keeps the complete result observable and prevents
// a repeated lookup from being hoisted out of the timing loop.
std::uint64_t observe(std::string_view value) {
    const char* data = value.data();
    const auto size = value.size();
    asm volatile("" : : "g"(data), "g"(size) : "memory");
    return size + (value.empty() ? 0 : static_cast<unsigned char>(value.front()));
}

// Expose all returned token views without adding a traversal to the measurement.
std::uint64_t observe(const std::vector<std::string_view>& value) {
    const auto* data = value.data();
    const auto size = value.size();
    asm volatile("" : : "g"(data), "g"(size) : "memory");
    return size;
}

template<typename Stack, bool Accessor, bool Copies>
double measure(std::string_view input, std::size_t index, std::size_t iterations) {
    std::uint64_t checksum = 0;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        if constexpr (Copies) {
            std::string source(input);
            std::string output(select_part<Stack, Accessor>(source, index));
            checksum += observe(output);
        } else {
            checksum += observe(select_part<Stack, Accessor>(input, index));
        }
    }
    const auto elapsed = Clock::now() - start;
    checksum_sink = checksum;
    return std::chrono::duration<double, std::nano>(elapsed).count();
}

using Measure = double (*)(std::string_view, std::size_t, std::size_t);

[[gnu::noinline]] double measure_old_accessor(std::string_view input, std::size_t index, std::size_t iterations) {
    std::uint64_t checksum = 0;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
        checksum += observe(old_accessor::access_array_part(input, index));
    const auto elapsed = Clock::now() - start;
    checksum_sink = checksum;
    return std::chrono::duration<double, std::nano>(elapsed).count();
}

[[gnu::noinline]] double measure_new_accessor(std::string_view input, std::size_t index, std::size_t iterations) {
    std::uint64_t checksum = 0;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
        checksum += observe(new_accessor::access_array_part(input, index));
    const auto elapsed = Clock::now() - start;
    checksum_sink = checksum;
    return std::chrono::duration<double, std::nano>(elapsed).count();
}

[[gnu::noinline]] double measure_head_parser(std::string_view input, std::size_t iterations) {
    std::uint64_t checksum = 0;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
        checksum += observe(head_parser::parse_array_parts(input));
    const auto elapsed = Clock::now() - start;
    checksum_sink = checksum;
    return std::chrono::duration<double, std::nano>(elapsed).count();
}

[[gnu::noinline]] double measure_new_parser(std::string_view input, std::size_t iterations) {
    std::uint64_t checksum = 0;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
        checksum += observe(new_parser::parse_array_parts(input));
    const auto elapsed = Clock::now() - start;
    checksum_sink = checksum;
    return std::chrono::duration<double, std::nano>(elapsed).count();
}

constexpr std::array<const char*, 4> names{
    "head", "vector_string", "accessor_stack", "accessor_string"
};
template<bool Copies>
constexpr std::array<Measure, 4> methods{
    measure<std::stack<char>, false, Copies>, measure<StringStack, false, Copies>,
    measure<std::stack<char>, true, Copies>, measure<StringStack, true, Copies>
};

struct Fixture {
    std::string name;
    std::vector<std::string> tokens;
    std::string input() const {
        std::string result;
        for (const auto& token : tokens) {
            if (!result.empty()) result += ',';
            result += " " + token + " ";
        }
        return result;
    }
};

template<char delimiter = ','>
void validate_parser(std::string_view input, const std::vector<std::string>& expected, std::string_view mode) {
    static_assert(std::is_same_v<decltype(head_parser::parse_array_parts(input, delimiter)), std::vector<std::string_view>>);
    static_assert(std::is_same_v<decltype(new_parser::parse_array_parts<delimiter>(input)), std::vector<std::string_view>>);
    const std::size_t first_method = mode == "parse-new" ? 1 : 0;
    const std::size_t method_count = mode == "all" || mode == "parse" ? 2 : 1;
    for (std::size_t method = first_method; method < first_method + method_count; ++method) {
        const auto result = method == 0
            ? head_parser::parse_array_parts(input, delimiter)
            : new_parser::parse_array_parts<delimiter>(input);
        if (result.size() != expected.size()
            || !std::equal(result.begin(), result.end(), expected.begin()))
        {
            throw std::runtime_error(std::string(method == 0 ? "HEAD" : "New")
                + " parser disagrees with fixture [" + std::string(input) + "]");
        }
    }
}

void validate(std::string_view input, const std::vector<std::string>& expected, std::string_view mode = "all") {
    if (mode == "all" || mode.starts_with("parse")) {
        validate_parser(input, expected, mode);
        if (mode != "all")
            return;
    }
    static_assert(std::is_same_v<decltype(old_accessor::access_array_part(input, 0)), std::string_view>);
    static_assert(std::is_same_v<decltype(new_accessor::access_array_part(input, 0)), std::string_view>);
    constexpr std::array validation_names{
        names[0], names[1], names[2], names[3], "old_accessor", "new_accessor"
    };
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (mode != "all") {
            const auto result = mode == "old"
                ? old_accessor::access_array_part(input, i)
                : new_accessor::access_array_part(input, i);
            if (result != expected[i])
                throw std::runtime_error(std::string(mode) + " accessor at index "
                    + std::to_string(i) + ": expected [" + expected[i]
                    + "], got [" + std::string(result) + "]");
            continue;
        }
        const std::array results{
            select_part<std::stack<char>, false>(input, i),
            select_part<StringStack, false>(input, i),
            select_part<std::stack<char>, true>(input, i),
            select_part<StringStack, true>(input, i),
            old_accessor::access_array_part(input, i),
            new_accessor::access_array_part(input, i)
        };
        for (std::size_t method = 0; method < results.size(); ++method) {
            if (results[method] != expected[i])
                throw std::runtime_error(std::string(validation_names[method]) + " at index "
                    + std::to_string(i) + ": expected [" + expected[i]
                    + "], got [" + std::string(results[method]) + "]");
        }
    }
}

int main(int argc, char** argv) try {
    std::string_view mode = "all";
    int first_argument = 1;
    if (argc > 1 && std::string_view(argv[1]) == "--old-only") {
        mode = "old";
        ++first_argument;
    } else if (argc > 1 && std::string_view(argv[1]) == "--new-only") {
        mode = "new";
        ++first_argument;
    } else if (argc > 1 && std::string_view(argv[1]) == "--parse-only") {
        mode = "parse";
        ++first_argument;
    } else if (argc > 1 && std::string_view(argv[1]) == "--parse-head-only") {
        mode = "parse-head";
        ++first_argument;
    } else if (argc > 1 && std::string_view(argv[1]) == "--parse-new-only") {
        mode = "parse-new";
        ++first_argument;
    }
    const int milliseconds = argc > first_argument ? std::stoi(argv[first_argument]) : 10;
    const int rounds = argc > first_argument + 1 ? std::stoi(argv[first_argument + 1]) : 7;
    if (argc > first_argument + 2 || milliseconds < 1 || milliseconds > 1000 || rounds < 3 || rounds > 101)
        throw std::invalid_argument("Usage: benchmark_array_access.out [--old-only|--new-only|--parse-only|--parse-head-only|--parse-new-only] [sample_ms:1..1000] [rounds:3..101]");

    std::vector<Fixture> fixtures{
        {"pair", {"10", "20"}},
        {"bounds", {"10", "20", "640", "480"}}
    };
    for (int count : {64, 512}) {
        Fixture f{"flat" + std::to_string(count), {}};
        for (int i = 0; i < count; ++i) f.tokens.push_back(std::to_string(1000 + i));
        fixtures.push_back(std::move(f));
    }
    Fixture quoted{"quoted16", {}}, nested{"nested16", {}};
    for (int i = 0; i < 4; ++i) {
        for (auto token : {R"("left,right")", R"('single,quoted')", R"("escaped \", comma")", R"("slash\\")"})
            quoted.tokens.emplace_back(token);
        for (auto token : {"[1,2]", R"({key:"a,b"})", "(x,[y,z])", "tail"})
            nested.tokens.emplace_back(token);
    }
    fixtures.push_back(std::move(quoted));
    fixtures.push_back(std::move(nested));
    fixtures.push_back({"long4", std::vector<std::string>(4, '"' + std::string(256, 'x') + '"')});
    std::string deep;
    for (int i = 0; i < 64; ++i) deep += "[({"[i % 3];
    deep += "x,y";
    for (int i = 63; i >= 0; --i) deep += "])}"[i % 3];
    fixtures.push_back({"depth64", {deep, "tail", deep, "end"}});

    // Validate all tokens, including escaped quotes and nesting beyond SSO.
    for (const auto& fixture : fixtures) validate(fixture.input(), fixture.tokens, mode);
    validate("\u3000first\u200B,\uFEFFsecond\u2060,\u00A0third\u00A0", {"first", "second", "third"}, mode);
    if (mode == "all" || mode.starts_with("parse")) {
        validate_parser("", {}, mode);
        validate_parser(",,", {"", "", ""}, mode);
        validate_parser("a,,c,", {"a", "", "c", ""}, mode);
        validate_parser(std::string_view{"a\0b,c", 5}, {std::string("a\0b", 3), "c"}, mode);
        validate_parser<':'>(R"(key: [1,2]: "a:b": (c:d))",
                             {"key", "[1,2]", R"("a:b")", "(c:d)"}, mode);
    }
    const std::string_view padded = " \u3000value\uFEFF ";
    const auto trimmed = mode == "old" || mode == "parse-head"
        ? old_accessor::utils::trim(padded)
        : cmn::utils::trim(padded);
    if (trimmed != "value" || trimmed.data() != padded.data() + 4)
        throw std::runtime_error("Trimming must preserve a view into the input");
    std::cerr << "Concept benchmark; HEAD model = commons d02acf3, commons ICU trim.\n"
              << "Full access_array_part comparison: old=std::stack with original helpers, new=std::string with cached top and optimized helpers.\n"
              << "Full parse_array_parts comparison: HEAD=commons d02acf3 with original helpers, new=current parser with optimized helpers.\n"
              << "Run mode: " << mode << '\n'
              << "Compiler: " << __VERSION__ << '\n'
              << "Samples: " << milliseconds << " ms, " << rounds << " rounds.\n";
    std::cout << "case,bytes,index,mode,variant,round,iterations,ns_per_lookup\n"
              << std::fixed << std::setprecision(3);
    for (const auto& fixture : fixtures) {
        const auto input = fixture.input();
        if (mode == "all" || mode.starts_with("parse")) {
            const std::array parser_methods{measure_head_parser, measure_new_parser};
            const std::array<const char*, 2> parser_names{"head", "new"};
            const std::size_t first_method = mode == "parse-new" ? 1 : 0;
            const std::size_t method_count = mode == "all" || mode == "parse" ? 2 : 1;
            std::array<std::size_t, 2> iterations{};
            for (std::size_t method = first_method; method < first_method + method_count; ++method) {
                auto& n = iterations[method];
                n = 64;
                while (parser_methods[method](input, n) < milliseconds * 1e6) {
                    if (n >= (std::size_t{1} << 28))
                        throw std::runtime_error("Could not calibrate parser benchmark");
                    n *= 2;
                }
            }
            for (int round = 0; round < rounds; ++round) {
                for (std::size_t offset = 0; offset < method_count; ++offset) {
                    const auto method = first_method + (offset + round) % method_count;
                    const auto n = iterations[method];
                    const auto ns = parser_methods[method](input, n) / n;
                    std::cout << fixture.name << ',' << input.size()
                              << ",-1,parse_array_parts," << parser_names[method] << ','
                              << round << ',' << n << ',' << ns << '\n';
                }
            }
            if (mode != "all") {
                std::cerr << "Measured " << fixture.name << '\n';
                continue;
            }
        }
        std::vector<std::size_t> indices{0, fixture.tokens.size() / 2, fixture.tokens.size() - 1};
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        for (const auto index : indices) {
            for (bool copies : {false, true}) {
                if (mode != "all")
                    break;
                const auto& runners = copies ? methods<true> : methods<false>;
                std::array<std::size_t, 4> iterations{};
                // Independent calibration keeps fast variants measurable too.
                for (std::size_t method = 0; method < names.size(); ++method) {
                    auto& n = iterations[method];
                    n = 64;
                    while (runners[method](input, index, n) < milliseconds * 1e6) {
                        if (n >= (std::size_t{1} << 28))
                            throw std::runtime_error("Could not calibrate benchmark");
                        n *= 2;
                    }
                }
                // Rotate execution order to reduce warmup/thermal order bias.
                for (int round = 0; round < rounds; ++round) {
                    for (std::size_t offset = 0; offset < names.size(); ++offset) {
                        const auto method = (offset + round) % names.size();
                        const auto n = iterations[method];
                        const auto ns = runners[method](input, index, n) / n;
                        std::cout << fixture.name << ',' << input.size() << ',' << index << ','
                                  << (copies ? "copies" : "selection") << ',' << names[method] << ','
                                  << round << ',' << n << ',' << ns << '\n';
                    }
                }
            }
            const std::array<Measure, 2> accessor_methods{measure_old_accessor, measure_new_accessor};
            const std::array<const char*, 2> accessor_names{"old", "new"};
            const std::size_t first_method = mode == "new" ? 1 : 0;
            const std::size_t method_count = mode == "all" ? 2 : 1;
            std::array<std::size_t, 2> iterations{};
            for (std::size_t method = first_method; method < first_method + method_count; ++method) {
                auto& n = iterations[method];
                n = 64;
                while (accessor_methods[method](input, index, n) < milliseconds * 1e6) {
                    if (n >= (std::size_t{1} << 28))
                        throw std::runtime_error("Could not calibrate accessor benchmark");
                    n *= 2;
                }
            }
            for (int round = 0; round < rounds; ++round) {
                for (std::size_t offset = 0; offset < method_count; ++offset) {
                    const auto method = first_method + (offset + round) % method_count;
                    const auto n = iterations[method];
                    const auto ns = accessor_methods[method](input, index, n) / n;
                    std::cout << fixture.name << ',' << input.size() << ',' << index
                              << ",access_array_part," << accessor_names[method] << ','
                              << round << ',' << n << ',' << ns << '\n';
                }
            }
        }
        std::cerr << "Measured " << fixture.name << '\n';
    }
    std::cerr << "Checksum: " << checksum_sink << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
