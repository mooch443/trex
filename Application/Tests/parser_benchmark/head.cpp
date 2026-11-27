#include <commons.pc.h>
#include "LegacyParsers.h"
#include <gui/DynamicGUI.h>
#include <gui/dyn/ResolveVariable.h>
#include <gui/dyn/Action.h>
#include <misc/frame_t.h>
#include <misc/ranges.h>
#include <misc/default_settings.h>

// Commons d02acf307f5dca62c1b63368c907ac24ff32c858.
// Parser snapshots use isolated namespaces and shared Context/State data objects.
// JSON type/value access is adapted to the current Glaze API.
namespace cmn::benchmark_head {
namespace utils {
using namespace cmn::utils;
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




void fast_fromstr(std::string& output, std::string_view sv) {
    const std::size_t N = sv.size();
    if (N < 2) {                               // too short to be quoted
        output.append(sv);
        return;
    }

    const char quote = sv[0];
    const bool quoted = ((quote == '"' || quote == '\'') && sv[N-1] == quote);
    size_t current_size = output.size();

    if (!quoted) {                             // no outer quotes → verbatim
        output.resize(current_size + N);
        std::memcpy(output.data() + current_size, sv.data(), N);
        return;
    }

    // Payload lies strictly between the two quotes
    output.reserve(current_size + N - 2);     // worst‑case growth (every '\' drops)

    const char* chunk = sv.data() + 1;         // first byte inside the quotes
    const char* end   = sv.data() + N - 1;     // one‑past‑last payload byte

    while (chunk < end) {
        // Find the next backslash in the remaining payload
        const char* bs = static_cast<const char*>(std::memchr(chunk, '\\', end - chunk));
        if (!bs || bs + 1 >= end) {
            // No more escape sequences (or stray trailing '\') – copy the rest verbatim
            size_t L = end - chunk;
            output.resize(current_size + L);
            std::memcpy(output.data() + current_size, chunk, L);
            break;
        }

        // Copy slice before the backslash
        size_t L = bs - chunk;
        output.resize(current_size + L);
        std::memcpy(output.data() + current_size, chunk, L);
        current_size += L;

        // Copy the character after the backslash, effectively un‑escaping it
        output.push_back(*(bs + 1));
        ++current_size;

        // Advance past the escape sequence
        chunk = bs + 2;
    }
}


}
namespace util {
using namespace cmn::util;
template<StringLike Str>
auto truncate(Str&& str) {
    using StrDecayed = std::remove_cvref_t<Str>;
    std::string_view sv = utils::string_like_view(std::forward<Str>(str));

    if ((utils::beginsWith(sv, '{') && utils::endsWith(sv, '}'))
        || (utils::beginsWith(sv, '[') && utils::endsWith(sv, ']'))) {

        std::string_view trimmed = utils::trim(sv.substr(1, sv.size() - 2));

        if constexpr (std::is_rvalue_reference_v<Str&&>
                      || (std::is_lvalue_reference_v<Str> && std::is_const_v<std::remove_reference_t<Str>>))
        {
            if constexpr (std::is_same_v<StrDecayed, std::string>) {
                return std::string(trimmed);
            } else
                return trimmed;
        } else
            return trimmed;
    }

    throw illegal_syntax("Cannot parse array " + std::string(sv) + ".");
}


template<typename Str>
concept is_viewable = std::is_array_v<std::remove_cvref_t<Str>>
        || _is_dumb_pointer<std::remove_cvref_t<Str>>
        || std::same_as<std::remove_cvref_t<Str>, std::string_view>
        || (std::is_lvalue_reference_v<Str> &&
            _clean_same<std::remove_reference_t<Str>, std::string>);

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


}

#if !defined(NDEBUG) && false
#define CTIMER_ENABLED true
#else
#define CTIMER_ENABLED false
#endif

#if CTIMER_ENABLED
#endif

struct CTimer {
#if CTIMER_ENABLED
    struct Timing {
        double count{0};
        uint64_t samples{0};
        
        std::string toStr() const {
            return cmn::dec<4>(count / double(samples) * 1000).toStr()+"ms total = "+ cmn::dec<4>(count).toStr()+"s";
        }
    };
    
    static inline std::unordered_map<std::string, Timing> timings;
    std::string_view _name;
    Timer timer;
#else
    constexpr CTimer(std::string_view) {}
#endif
#if CTIMER_ENABLED
    ~CTimer() {
        auto elapsed = timer.elapsed();
        auto &field = timings[std::string(_name)];
        field.count += elapsed;
        field.samples++;
        
        if(field.samples % 10000 == 0) {
            cmn::Print("! [", _name, "] ", field);
            field.count /= double(field.samples);
            field.samples = 1;
        }
    }
#endif
};
namespace dyn {
using namespace cmn::gui;
using namespace cmn::gui::dyn;
using Pattern = cmn::benchmark_head::pattern::UnresolvedStringPattern;
struct PreVarProps {
    std::string original;
    std::string_view name;
    std::vector<std::string_view> subs;
    std::vector<std::string_view> parameters;
    
    bool optional{false};
    bool html{false};
    
    PreVarProps() noexcept = default;
    
    // Copy constructor
    PreVarProps(const PreVarProps& other);
    
    // Move constructor
    PreVarProps(PreVarProps&& other) noexcept;
    
    std::string toStr() const;
    static consteval std::string_view class_name() { return "PreVarProps"; }
    VarProps parse(const Context& context, State& state) const;
};


// Function to skip over nested structures so they don't interfere with our parsing
auto skipNested(const auto& trimmedStr, std::size_t& pos, char openChar, char closeChar) {
    int balance = 1;
    pos++; // Skip the opening char
    while (pos < trimmedStr.size() and balance > 0) {
        if (trimmedStr[pos] == openChar) {
            balance++;
        } else if (trimmedStr[pos] == closeChar) {
            balance--;
        }
        pos++;
    }
    //return trimmedStr.substr(startPos, pos - startPos);  // Return the nested structure as a string
};

template<typename R>
R create_parse_result(std::string_view trimmedStr) {
    if (not trimmedStr.empty() and trimmedStr.front() == '{' and trimmedStr.back() == '}') {
        if constexpr(std::same_as<R, PreVarProps>) {
            trimmedStr = trimmedStr.substr(1, trimmedStr.size() - 2);
        } else {
            R action;
            action.original = std::string(trimmedStr);
            action.name = std::string_view(action.original);
            return action;
        }
    }

    std::size_t pos = 0;
    
    R action;
    action.original = std::string(trimmedStr);
    trimmedStr = std::string_view(action.original);
    
    while (pos < trimmedStr.size() and trimmedStr[pos] not_eq ':') {
        ++pos;
    }

    action.name = trimmedStr.substr(0, pos);
    bool inQuote = false;
    char quoteChar = '\0';
    int braceLevel = 0;  // Keep track of nested braces
    std::size_t token_start = ++pos;  // Skip the first ':'

    for (; pos < trimmedStr.size(); ++pos) {
        char c = trimmedStr[pos];

        if (c == ':' and not inQuote and braceLevel == 0) {
            assert(pos >= token_start);
            action.parameters.emplace_back(trimmedStr.substr(token_start, pos - token_start));
            token_start = pos + 1;
        } else if ((c == '\'' or c == '\"') and (not inQuote or c == quoteChar)) {
            inQuote = not inQuote;
            if (inQuote) {
                quoteChar = c;
            }
        } else if (c == '{' and not inQuote) {
            // Assuming skipNested advances 'pos'
            skipNested(trimmedStr, pos, '{', '}');
            --pos;
        } else if (c == '[' and not inQuote) {
            ++braceLevel;
        } else if (c == ']' and not inQuote) {
            --braceLevel;
        }
    }

    if (inQuote) {
        throw std::invalid_argument("Invalid format: Missing closing quote");
    }
    
    if (token_start < pos) {
        action.parameters.emplace_back(trimmedStr.substr(token_start, pos - token_start));
    }
    
    return action;
}

PreVarProps extractControls(const std::string_view& variable);

template<cmn::StringLike Str>
bool convert_to_bool(Str&& p) noexcept {
    auto sv = utils::string_like_view(std::forward<Str>(p));
    if (   sv == "false"
        || sv == "[]"
        || sv == "{}"
        || sv == "0"
        || sv == "null"
        || sv == "''"
        || sv == "\"\""
        || sv == "0.0"
        || sv == ""
        || sv == "Map<size:0>"
        )
      return false;

    return true;
}




using cmn::gui::dyn::get_modifier_from_object;

template<typename ApplyF, typename ErrorF, typename Result = typename cmn::detail::return_type<ApplyF>::type>
inline auto resolve_variable(const std::string_view& word, const Context& context, State& state, ApplyF&& apply, ErrorF&& error) -> Result {
    auto props = extractControls(word);
    
    try {
        if(props.name == "if") {
            //[[maybe_unused]] CTimer ctimer("if");
            VarProps p{
                .name = std::string(props.name),
                .optional = props.optional,
                .html = props.html,
            };
            
            p.parameters.reserve(props.parameters.size());
            for(auto &pr : props.parameters) {
                p.parameters.emplace_back((std::string)pr);
            }
            
            if(p.parameters.size() >= 2) {
                bool condition = convert_to_bool(parse_text_snapshot(p.parameters.at(0), context, state));
                //Print("Condition ", props.parameters.at(0)," => ", condition);
                if(condition)
                    return Meta::fromStr<Result>(parse_text_snapshot(p.parameters.at(1), context, state));
                else if(p.parameters.size() == 3)
                    return Meta::fromStr<Result>(parse_text_snapshot(p.parameters.at(2), context, state));
                else
                    return "";
            }

        } else if(const bool is_and = props.name == "&&";
                  is_and || props.name == "||")
        {
            //[[maybe_unused]] CTimer ctimer("logical");
            bool result = is_and;

            // Resolve operands in order so a decisive value protects the rest.
            for(const auto& parameter : props.parameters) {
                const bool value = convert_to_bool(parse_text_snapshot(parameter, context, state));
                if((is_and && not value) || (not is_and && value)) {
                    result = value;
                    break;
                }
            }

            if constexpr(StringLike<Result>)
                return Result(result ? "true" : "false");
            else
                return Meta::fromStr<Result>(result ? std::string_view("true") : std::string_view("false"));
            
        } else if(props.name == "for") {
            //[[maybe_unused]] CTimer ctimer("for");
            VarProps p{
                .name = std::string(props.name),
                .optional = props.optional,
                .html = props.html,
            };
            
            p.parameters.reserve(props.parameters.size());
            for(auto& pr : props.parameters) {
                p.parameters.emplace_back((std::string)pr);
            }
            
            if(p.parameters.size() < 2) {
                throw InvalidArgumentException("A for loop needs at least two parameters, received ", p.parameters.size(), ".");
            }
            
            std::string variable = "i";
            std::string list_expr = p.parameters.at(0);
            std::string fn_expr = p.parameters.at(1);
            if(p.parameters.size() >= 3) {
                variable = parse_text_snapshot(p.parameters.at(0), context, state);
                if(variable.empty()) {
                    variable = "i";
                }
                list_expr = p.parameters.at(1);
                fn_expr = p.parameters.at(2);
            }
            
            auto handler = state._current_object_handler.lock();
            if(not handler) {
                handler = std::make_shared<CurrentObjectHandler>();
                state._current_object_handler = std::weak_ptr(handler);
            }
            
            std::stringstream ss;
            ss << "[";
            
            auto trunc = parse_text_snapshot(list_expr, context, state);
            size_t index = 0;
            std::string idx_str;
            if(not trunc.empty() && is_in(utils::trim(trunc).front(), '[', '{')) {
                auto parts = util::parse_array_parts(util::truncate(utils::trim(std::string_view(trunc))));
                for(auto& item : parts) {
                    auto scope = handler->scope();
                    idx_str = Meta::toStr(index);
                    scope.set(variable, item);
                    scope.set("idx", idx_str);
                    if(index > 0) {
                        ss << ",";
                    }
                    ss << parse_text_snapshot(fn_expr, context, state);
                    ++index;
                }
                
            } else if(not trunc.empty() && is_in(trunc.front(), '\'', '"')) {
                auto text = Meta::fromStr<std::string>(trunc);
                for(char c : text) {
                    auto scope = handler->scope();
                    idx_str = Meta::toStr(index);
                    scope.set(variable, std::string_view(&c, 1));
                    scope.set("idx", idx_str);
                    if(index > 0) {
                        ss << ",";
                    }
                    ss << parse_text_snapshot(fn_expr, context, state);
                    ++index;
                }
                
            } else {
                throw InvalidArgumentException("For loop list must evaluate to an array or string, received ", no_quotes(trunc), ".");
            }
            
            ss << "]";
            return Meta::fromStr<Result>(ss.str());
            
        } else if(context.has(props.name, state)) {
            //[[maybe_unused]] CTimer ctimer("normal");
            
            auto variable = context.variable(props.name, state);
            if constexpr(std::invocable<ApplyF, VarBase_t&, const VarProps&>)
                return apply(*variable, props.parse(context, state));
            else
                static_assert(std::invocable<ApplyF, VarBase_t&, const VarProps&>);
            
        } else if(auto it = context.defaults.variables.find(props.name); it != context.defaults.variables.end()) {
            //[[maybe_unused]] CTimer ctimer("custom var");
            return Meta::fromStr<Result>(it->second->value<std::string>(context, state));
            
        } else if(auto lock = state._current_object_handler.lock();
                  lock != nullptr)
        {
            if(auto ptr = lock->retrieve_named(props.name);
               ptr != nullptr)
            {
                auto v = get_modifier_from_object(ptr.get(), props.parse(context, state));
                if(v.has_value()) {
                    return Meta::fromStr<Result>(v.value());
                }
            }
            
        }
        /// if its an unknown "thing", we error in the following code
    } catch(...) {
        // catch exceptions
    }
    
    if constexpr(std::invocable<ErrorF, bool>)
        return error(props.optional);
    else
        return error();
}





std::string PreVarProps::toStr() const {
    return "PreVarProps<"+std::string(name)+" parm="+Meta::toStr(parameters)+" subs="+Meta::toStr(subs)+">";
}



// Copy constructor
PreVarProps::PreVarProps(const PreVarProps& other)
    : original(other.original), // copy the string
      optional(other.optional),
      html(other.html)
{
    // Compute the offset in the original string for 'name'
    auto offset_name = other.name.data() - other.original.data();
    name = std::string_view(original.data() + offset_name, other.name.size());

    // Compute the offsets in the original string for 'subs' and 'parameters'
    subs.resize(other.subs.size());
    parameters.resize(other.parameters.size());
    for (size_t i = 0; i < other.subs.size(); ++i) {
        auto offset = other.subs[i].data() - other.original.data();
        subs[i] = std::string_view(original.data() + offset, other.subs[i].size());
    }
    for (size_t i = 0; i < other.parameters.size(); ++i) {
        auto offset = other.parameters[i].data() - other.original.data();
        parameters[i] = std::string_view(original.data() + offset, other.parameters[i].size());
    }
}

// Move constructor
PreVarProps::PreVarProps(PreVarProps&& other) noexcept {
    original = (other.original);
    optional = other.optional;
    html = other.html;
    
    // Compute the offset in the original string for 'name'
    auto offset_name = other.name.data() - other.original.data();
    name = std::string_view(original.data() + offset_name, other.name.size());

    // Compute the offsets in the original string for 'subs' and 'parameters'
    subs.resize(other.subs.size());
    parameters.resize(other.parameters.size());
    for (size_t i = 0; i < other.subs.size(); ++i) {
        auto offset = other.subs[i].data() - other.original.data();
        subs[i] = std::string_view(original.data() + offset, other.subs[i].size());
    }
    for (size_t i = 0; i < other.parameters.size(); ++i) {
        auto offset = other.parameters[i].data() - other.original.data();
        parameters[i] = std::string_view(original.data() + offset, other.parameters[i].size());
    }
}

VarProps PreVarProps::parse(const Context& context, State& state) const {
    VarProps props{
        .name = std::string(name),
        .optional = optional,
        .html = html
    };
    
    props.subs.reserve(subs.size());
    for(auto &s : subs) {
        props.subs.emplace_back(parse_text_snapshot(s, context, state));
    }
    
    if(props.name == "if" || props.name == "for")  {
        for(auto &p : parameters)
            props.parameters.emplace_back(p);
        
    } else {
        props.parameters.reserve(parameters.size());
        for(auto &p : parameters) {
            props.parameters.emplace_back(parse_text_snapshot(p, context, state));
        }
    }
    
    return props;
}





template<typename Number>
    requires std::integral<Number>
bool is_valid_number(std::string_view str) {
    if(str.empty())
        return false;
    
    for(char c : str) {
        if(not std::isdigit(c))
            return false;
    }
    return true;
}

std::string access_sub_field(const glz::generic& json, std::queue<std::string_view> subs) {
    if(not subs.empty()) {
        const auto& name = subs.front();
        subs.pop();
        
        if(json.is_object()) {
            if(not json.contains(name))
                throw InvalidArgumentException("Object ", json, " does not contain field ", name);
            return access_sub_field(json[name], subs);
        
        } else if(json.is_array()) {
            auto index = Meta::fromStr<uint16_t>(name);
            auto& obj = json.get_array()[index];
            return access_sub_field(obj, subs);
        }
    }
    
    if(json.is_string())
        return json.template get<std::string>();
    return Meta::toStr(json);
}

std::string _handle_subobjects(const VarBase_t& variable, const VarProps& modifiers, std::optional<std::queue<std::string_view>> subs = std::nullopt) {
    if(not subs) {
        subs = std::queue<std::string_view>{};
        for(auto &sub : modifiers.subs)
            subs->emplace(sub);
    }
    
    try {
        std::string ret;
        if(subs->empty()) {
            ret = variable.value_string(modifiers);
            
        } else if(variable.is<Size2>()) {
            if(subs->front() == "w")
                ret = Meta::toStr(variable.value<Size2>(modifiers).width);
            else if(subs->front() == "h")
                ret = Meta::toStr(variable.value<Size2>(modifiers).height);
            else
                throw InvalidArgumentException("Sub ",modifiers," of Size2 is not valid.");
            
        } else if(variable.is<Vec2>()) {
            if(subs->front() == "x")
                ret = Meta::toStr(variable.value<Vec2>(modifiers).x);
            else if(subs->front() == "y")
                ret = Meta::toStr(variable.value<Vec2>(modifiers).y);
            else
                throw InvalidArgumentException("Sub ",modifiers," of Vec2 is not valid.");
            
        } else if(variable.is<Range<Frame_t>>()) {
            if(subs->front() == "start")
                ret = Meta::toStr(variable.value<Range<Frame_t>>(modifiers).start);
            else if(subs->front() == "end")
                ret = Meta::toStr(variable.value<Range<Frame_t>>(modifiers).end);
            else
                throw InvalidArgumentException("Sub ",modifiers," of Range<Frame_t> is not valid.");
                
        } else if(variable.is<glz::generic>()) {
            return variable.access_value<glz::generic>([&](auto&& value){
                return access_sub_field(value, *subs);
            }, modifiers);
        }
        else if(variable.is<sprite::Map>()) {
            return variable.access_value<sprite::Map>([&](auto&& value) {
                if(auto key = subs->front();
                   value.has(key))
                {
                    subs->pop();
                    
                    auto prop = value.at(key);
                    assert(prop.valid());
                    if(prop->template is_type<glz::generic>()) {
                        return access_sub_field(prop->template value<glz::generic>(), *subs);
                    } else if(prop->is_array()
                              && not subs->empty())
                    {
                        auto p = prop->access_array(Meta::fromStr<uint16_t>(subs->front()));
                        if(p
                           && p->valid())
                        {
                            return Meta::fromStr<std::string>(p->valueString());
                        }
                        
                    } else if(prop->template is_type<sprite::Map>()) {
                        throw InvalidArgumentException("Accessing a sprite::Map inside a sprite::Map is unsupported (", modifiers,").");
                    }
                    
                    return Meta::fromStr<std::string>(value.at(key)->valueString());
                } else
                    throw InvalidArgumentException("No key named ", modifiers, " in map.");
            }, modifiers);
            
        } else {
            ret = variable.value_string(modifiers);
            //throw InvalidArgumentException("Variable ", modifiers.name, " does not have arguments (requested ", modifiers.parameters,").");
            //auto str = modifiers.toStr();
            //Print(str.c_str(), " resolves to ", ret);
            
            /// support array access for string-arrays
            if(subs
               && not subs->empty()
               && not ret.empty())
            {
                if(ret.front() == '['
                   && ret.back() == ']')
                {
                    if(is_in(subs->front(), "x", "y")) {
                        /// we specifically allow x/y subtypes if the array has two elements
                        auto parts = util::parse_array_parts(util::truncate(ret));
                        if(parts.size() != 2)
                            throw InvalidArgumentException("Expected exactly 2 array elements for string 'array' ", ret, ". Got ", parts, " instead.");
                        if(subs->front() == "x") {
                            ret = parts[0];
                        } else
                            ret = parts[1];
                        
                    } else if(is_valid_number<uint16_t>(subs->front())) {
                        /// must be resolved to an integer!
                        auto index = Meta::fromStr<uint16_t>(subs->front());
                        
                        /// this is an array
                        ret = util::parse_array_parts(util::truncate(ret)).at(index);
                    }
                    
                } else {
                    //throw InvalidArgumentException("Suspected string 'array' ", ret, " is not an array (with subs ", *subs,").");
                }
            }
        }
        
        if(modifiers.html)
            return settings::htmlify(ret);
        return ret;
    } catch(const std::exception& ex) {
        if(not modifiers.optional)
            throw;//FormatExcept("Exception: ", ex.what(), " in variable: ", modifiers);
        return modifiers.optional ? "" : "null";
    }
}

template<typename T>
    requires (std::convertible_to<T, std::string_view> || std::same_as<T, Pattern>)
std::string _parse_text(const T& _pattern, const Context& context, State& state) {
    std::string_view pattern;
    if constexpr(std::same_as<T, Pattern>) {
        pattern = std::string_view(_pattern.original);
    } else
        pattern = std::string_view(_pattern);
    
    std::stringstream output;
    std::stack<std::size_t> nesting_start_positions;
    bool comment_out = false;

    for(std::size_t i = 0; i < pattern.size(); ++i) {
        char ch = pattern[i];
        if(nesting_start_positions.empty()) {
            switch(ch) {
                case '\\':
                    if(not comment_out) {
                        comment_out = true;
                    } else {
                        comment_out = false;
                        output << '\\';
                    }
                    break;
                case '{':
                    if(!comment_out) nesting_start_positions.push(i + 1);
                    else {
                        output << ch;
                    }
                    comment_out = false;
                    break;
                case '}':
                    if(!comment_out) throw InvalidSyntaxException("Mismatched closing brace at position ", i);
                    else {
                        output << ch;
                    }
                    comment_out = false;
                    break;
                default:
                    if(comment_out) {
                        FormatWarning("Invalid escape sequence at position ", i, ": ", ch, ". Only braces need to be escaped.");
                        comment_out = false;
                    }
                    output << ch;
            }
        } else {
            if(comment_out) {
                //output << ch;
                comment_out = false;
                
            } else if(ch == '\\') {
                comment_out = true;
                
            } else if(ch == '}') {
                if(nesting_start_positions.empty()) {
                    throw InvalidSyntaxException("Mismatched closing brace at position ", i);
                }
                
                std::size_t start_pos = nesting_start_positions.top();
                nesting_start_positions.pop();

                if(nesting_start_positions.empty()) {
                    std::string_view current_word = pattern.substr(start_pos, i - start_pos);
                    if(current_word.empty()) {
                        throw InvalidSyntaxException("Empty braces at position ", i);
                    }
                    
                    [[maybe_unused]] CTimer timer(current_word);
                    std::string resolved_word;
                    if(auto value = state.cached_variable_value(current_word, state._current_object_handler.lock());
                       value.has_value())
                    {
                        resolved_word = value.value();
                        
                    } else {
                        resolved_word = cmn::benchmark_head::dyn::resolve_variable(current_word, context, state, [](const VarBase_t& variable, const VarProps& modifiers) -> std::string {
                            return cmn::benchmark_head::dyn::_handle_subobjects(variable, modifiers);
                        }, [current_word](bool optional) -> std::string {
                            if(not optional)
                                throw InvalidSyntaxException("Cannot find variable or expression ", current_word,".");
                            //if(not optional)
                            //    FormatWarning("Cannot find variable or expression ", current_word,".");
                            //return optional ? "" : "null";
                            return "";
                        });
                        
                        state.set_cached_variable_value(state._current_object_handler.lock(), current_word, resolved_word);
                    }
                    if(nesting_start_positions.empty()) {
                        output << resolved_word;
                    } else {
                        //nesting.top() += resolved_word;
                    }
                }
                
            } else if(ch == '{') {
                nesting_start_positions.push(i + 1);
            } else {
                if(nesting_start_positions.empty()) {
                    throw InvalidSyntaxException("Mismatched opening brace at position ", i);
                }
                //nesting.top() += ch;
            }
        }
    }

    if(not nesting_start_positions.empty()) {
        throw InvalidSyntaxException("Mismatched opening brace");
    }
    
    if(comment_out) {
        // Trailing backslash without a following character
        throw InvalidSyntaxException("Trailing backslash");
    }

    return output.str();
}

std::string parse_text_snapshot(const std::string_view& pattern, const Context& context, State& state) {
    return _parse_text(pattern, context, state);
}
/*std::string parse_text_snapshot(const Pattern& pattern, const Context& context, State& state) {
    return _parse_text(pattern, context, state);
}*/

PreVarProps extractControls(const std::string_view& variable) {
    auto props = create_parse_result<PreVarProps>(variable);
    
    bool html{false}, optional{false};
    std::size_t controlsSize = 0;
    for (char c : props.name) {
        if(c == '#' || c == '.') {
            if (c == '#') {
                html = true;
            }
            else if (c == '.') {
                optional = true;
            }
            controlsSize++;
        } else {
            // Once we hit a character that's not a control character,
            // we break out of the loop.
            break;
        }
    }

    // Parsing the name and sub-variables, omitting the controls
    auto r = utils::split(std::string_view(props.name).substr(controlsSize), '.');
    
    // Assigning the action name without the control characters.
    if (r.empty())
        throw InvalidArgumentException("No variables found: ", props.name);

    props.optional = optional;
    props.html = html;
    /*VarProps props{
        .name{std::string(r.front())},
        .optional = optional,
        .html = html
    };*/
    
    const size_t N = r.size();
    props.subs.resize(N - 1);
    for (size_t i=1, N = r.size(); i<N; ++i) {
        props.subs[i - 1] = r[i];
    }

    //Print("Initial parameters = ", props.parameters);
    //for(auto &p : props.parameters) {
        // parse parameters here
        // if the parameter seems to be a string (quotes '"), use parse_text_snapshot(text, context) function to parse it
        // if the parameter seems to be a variable, it needs to be resolved:
    //    p = _parse_text(p, context, state);
    //}
    
    if(r.front() != std::string_view(props.name))
        props.name = r.front();
    return props;
}


template<typename T, bool allow_null_ambiguity = false>
T convert_number(std::string_view a) {
    if constexpr(are_the_same<bool, T>) {
        return convert_to_bool(a);
    }
    
    T A{};
    
    /// this is to support "null" == 0 explicitly (and some variations)
    if constexpr(allow_null_ambiguity) {
        bool truth_value = convert_to_bool(a);
        
        if constexpr(std::is_floating_point_v<std::remove_cvref_t<T>>) {
            if(truth_value)
                A = T(Meta::fromStr<double>(a));
            
        } else if constexpr(std::is_integral_v<std::remove_cvref_t<T>>) {
            if(truth_value)
                A = T(Meta::fromStr<int64_t>(a));
            
        } else {
            if (utils::beginsWith(a, '[') && utils::endsWith(a, ']')) {
                A = Meta::fromStr<T>(a);
                
            } else if(truth_value) {
                A = T(Meta::fromStr<double>(a));
            }
        }
        
    } else if constexpr(std::is_floating_point_v<std::remove_cvref_t<T>>) {
        A = T(Meta::fromStr<double>(a));
            
    } else if constexpr(std::is_integral_v<std::remove_cvref_t<T>>) {
        A = T(Meta::fromStr<int64_t>(a));
        
    } else {
        if (utils::beginsWith(a, '[') && utils::endsWith(a, ']')) {
            A = Meta::fromStr<T>(a);
        } else {
            A = T(Meta::fromStr<double>(a));
        }
    }
    
    return A;
}

template<typename T, bool allow_null_ambiguity = false, typename K = T>
K map_vectors(const VarProps& props, auto&& apply) {
    REQUIRE_EXACTLY(2, props);
    
    T A{}, B{};
    
    try {
        std::string_view a = utils::trim(std::string_view(props.parameters.front()));
        std::string_view b = utils::trim(std::string_view(props.parameters.back()));
        
        A = convert_number<T, allow_null_ambiguity>(a);
        B = convert_number<T, allow_null_ambiguity>(b);

    } catch(const std::exception& ex) {
        throw InvalidSyntaxException("Cannot parse ", props,": ", ex.what());
        return K{};
    }
    
    return apply(A, B);
}


void configure(Context& context) {
    auto& variables = context.system_variables();
    for(auto&& entry : std::initializer_list<Context::VariablePair>{
        VarFunc("int", [](const VarProps& props) -> int64_t {
                REQUIRE_EXACTLY(1, props);
                
                auto v = Meta::fromStr<double>(props.parameters.front());
                if(std::isnan(v))
                    return static_cast<int64_t>(std::numeric_limits<int64_t>::quiet_NaN());
                else
                    return static_cast<int64_t>(v);
            }),
        VarFunc("dec", [](const VarProps& props) -> std::string {
                REQUIRE_EXACTLY(2, props);
                float decimals = Meta::fromStr<uint8_t>(props.parameters.front());
                
                auto str = props.parameters.back();
                auto it = std::find(str.begin(), str.end(), '.');
                if(it != str.end()) {
                    size_t offset = 0;
                    while(++it != str.end() && offset++ < decimals) {}
                    str.erase(it, str.end());
                }
                
                return str;
            }),
        VarFunc("not", [](const VarProps& props) -> bool {
                REQUIRE_EXACTLY(1, props);
                
                std::string_view p(props.parameters.front());
                try {
                    return not convert_to_bool(p);
                    //return not Meta::fromStr<bool>(p);
                    
                } catch(const std::exception& ex) {
                    throw InvalidArgumentException("Cannot parse boolean ", p, ": ", ex.what());
                }
            }),
        VarFunc("equal", [](const VarProps& props) -> bool {
                REQUIRE_EXACTLY(2, props);
                
                std::string_view p0(props.parameters.front());
                std::string_view p1(props.parameters.back());
                try {
                    return p0 == p1;
                    
                } catch(const std::exception& ex) {
                    throw InvalidArgumentException("Cannot parse boolean ", p0, " == ",p1,": ", ex.what());
                }
            }),
        VarFunc("&&", [](const VarProps& props) -> bool {
                for (auto &c : props.parameters) {
                    if(not convert_to_bool(c))
                        return false;
                }
                return true;
            }),
        VarFunc("||", [](const VarProps& props) -> bool {
                for (auto &c : props.parameters) {
                    if(convert_to_bool(c))
                        return true;
                }
                return false;
            }),
        VarFunc("*", [](const VarProps& props) {
                return map_vectors<double>(props, [](auto&A, auto&B){return A * B;});
            }),
        VarFunc("at", [](const VarProps& props) -> std::string {
                REQUIRE_EXACTLY(2, props);
                
                auto trunc = std::string_view(props.parameters.at(1));
                if(is_in(utils::trim(trunc).front(), '{')) {
                    auto key = Meta::fromStr<std::string>(props.parameters.front());
                    auto parts = util::parse_array_parts(util::truncate(utils::trim(trunc)));
                    
                    for(auto &part : parts) {
                        auto key_value = util::parse_array_parts(utils::trim(part), ':');
                        if(key_value.size() != 2) {
                            throw InvalidArgumentException("Map component ", part, " does not consist of key:value pair. Instead we have ", key_value,".");
                        }
                        
                        if(key == Meta::fromStr<std::string>(key_value.front())) {
                            return std::string(key_value.back());
                        }
                    }
                    
                    throw InvalidArgumentException("Cannot find key ", no_quotes(key)," in ", trunc);
                    
                } else {
                    auto index = Meta::fromStr<size_t>(props.parameters.front());
                    if(is_in(utils::trim(trunc).front(), '[', '{')) {
                        auto parts = util::parse_array_parts(util::truncate(utils::trim(trunc)));
                        return std::string(parts.at(index));
                    } else if(is_in(trunc.front(), '\'', '"')) {
                        return std::string(1, Meta::fromStr<std::string>(trunc).at(index));
                    }
                }
                
                throw InvalidArgumentException("Needs to be an indexable type, got ", props, " instead.");
            })
    }) {
        variables[entry.first] = entry.second;
    }
}

}
namespace pattern {


template<typename Number>
    requires std::integral<Number>
bool is_valid_number(std::string_view str) {
    if(str.empty())
        return false;
    
    for(char c : str) {
        if(not std::isdigit(c))
            return false;
    }
    return true;
}

using namespace cmn::benchmark_head::dyn;
UnpreparedPatterns parse_words(std::string_view pattern);

UnresolvedStringPattern::UnresolvedStringPattern(UnresolvedStringPattern&& other)
    : original(std::move(other.original)),
      objects(std::move(other.objects)),
      typical_length(std::move(other.typical_length)),
      all_patterns(std::move(other.all_patterns))
{
    other.all_patterns.clear();
}

UnresolvedStringPattern::UnresolvedStringPattern(const UnresolvedStringPattern& other)
{
    *this = other;
}

UnresolvedStringPattern& UnresolvedStringPattern::operator=(UnresolvedStringPattern&& other) {
    for(auto pattern: all_patterns) {
        delete pattern;
    }
    original = std::move(other.original);
    objects = std::move(other.objects);
    typical_length = std::move(other.typical_length);
    all_patterns = std::move(other.all_patterns);
    
    other.all_patterns.clear();
    return *this;
}

UnresolvedStringPattern& UnresolvedStringPattern::operator=(const UnresolvedStringPattern& other)
{
    if (this == &other) return *this;

    // Delete old patterns
    for (auto pattern : all_patterns) {
        delete pattern;
    }
    all_patterns.clear();
    
    // this will contain the text for all the string_views
    if(other.original)
        original = std::make_unique<std::string>(*other.original);
    else
        original = nullptr;

    // Copy simple members
    objects = other.objects;
    typical_length = other.typical_length;

    const char* old_base = other.original ? other.original->data() : nullptr;
    const char* old_base_end = other.original ? other.original->data() + other.original->size() : nullptr;
    const char* new_base = original      ? original->data()      : nullptr;

    // Deep copy all_patterns and build a pointer mapping
    all_patterns.reserve(other.all_patterns.size());
    std::unordered_map<const Prepared*, Prepared*> prepared_map;
    for (const auto* pattern : other.all_patterns) {
        if (pattern) {
            Prepared* new_p = new Prepared(*pattern);
            all_patterns.push_back(new_p);
            prepared_map[pattern] = new_p;
        } else {
            all_patterns.push_back(nullptr);
        }
    }

    // Recursively fix all POINTER, PREPARED, and SV references in objects
    std::function<void(PreparedPattern&)> fix_ptrs;
    std::function<void(Prepared&)>        fix_prepared;   // non‑const; we mutate pointers

    fix_ptrs = [&](PreparedPattern& pat) {
        switch (pat.type) {
            case PreparedPattern::SV:
                // Remap the string_view so it points into this->original, not other's.
                if (old_base
                    && new_base
                    && pat.value.sv.data() >= old_base
                    && pat.value.sv.data() < old_base_end)
                {
                    std::ptrdiff_t offset = pat.value.sv.data() - old_base;
                    pat.value.sv = std::string_view(new_base + offset, pat.value.sv.size());
                }
                break;

            case PreparedPattern::POINTER:
                if (pat.value.ptr) {
                    auto it = prepared_map.find(pat.value.ptr);
                    assert(it != prepared_map.end());
                    if (it != prepared_map.end())
                    {             // pointer still old → remap
                        pat.value.ptr = it->second;
                    }
                    //fix_prepared(*pat.value.ptr);                // always recurse
                }
                break;

            case PreparedPattern::PREPARED:
                if (pat.value.prepared) {
                    auto it = prepared_map.find(pat.value.prepared);
                    assert(it != prepared_map.end());
                    if (it != prepared_map.end())
                    {              // still pointing at “other” ⇒ remap
                        pat.value.prepared = it->second;
                    }
                    fix_prepared(*pat.value.prepared);           // recurse
                }
                break;

            default:
                /* SV handled above; nothing to do for other enum values */
                break;
        }
    };

    fix_prepared = [&](Prepared& prep)
    {
        if (old_base
            && new_base
            && prep.original.data() >= old_base
            && prep.original.data() < old_base_end)
        {
            std::ptrdiff_t offset = prep.original.data() - old_base;
            prep.original = std::string_view(new_base + offset, prep.original.size());
        }
        
        for (auto& paramVec : prep.parameters)
            for (auto& child : paramVec)
                fix_ptrs(child);
        
        for(auto& sub : prep.subs) {
            for(auto &child: sub)
                fix_ptrs(child);
        }
    };

    for (auto& obj : objects)
        fix_ptrs(obj);

    return *this;
}

Unprepared create_parse_result(std::string_view trimmedStr) {
    if (not trimmedStr.empty() and trimmedStr.front() == '{' and trimmedStr.back() == '}')
    {
        trimmedStr = trimmedStr.substr(1, trimmedStr.size() - 2);
    }

    std::size_t pos = 0;
    
    Unprepared action;
    action.original = trimmedStr;
    
    while (pos < trimmedStr.size() and trimmedStr[pos] not_eq ':') {
        ++pos;
    }

    action.name = trimmedStr.substr(0, pos);
    bool inQuote = false;
    char quoteChar = '\0';
    int braceLevel = 0;  // Keep track of nested braces
    std::size_t token_start = ++pos;  // Skip the first ':'

    for (; pos < trimmedStr.size(); ++pos) {
        char c = trimmedStr[pos];

        if (c == ':' and not inQuote and braceLevel == 0) {
            action.parameters.emplace_back(trimmedStr.substr(token_start, pos - token_start));
            token_start = pos + 1;
        } else if ((c == '\'' or c == '\"') and (not inQuote or c == quoteChar)) {
            inQuote = not inQuote;
            if (inQuote) {
                quoteChar = c;
            }
        } else if (c == '{' and not inQuote) {
            // Assuming skipNested advances 'pos'
            gui::dyn::skipNested(trimmedStr, pos, '{', '}');
            --pos;
        } else if (c == '[' and not inQuote) {
            ++braceLevel;
        } else if (c == ']' and not inQuote) {
            --braceLevel;
        }
    }

    if (inQuote) {
        throw std::invalid_argument("Invalid format: Missing closing quote");
    }
    
    if (token_start < pos) {
        action.parameters.emplace_back(trimmedStr.substr(token_start, pos - token_start));
    }
    
    return action;
}

Unprepared extract_unprepared(std::string_view variable) {
    Unprepared props = create_parse_result(variable);
    
    bool html{false}, optional{false};
    std::size_t controlsSize = 0;
    for (char c : props.name) {
        if(c == '#' || c == '.') {
            if (c == '#') {
                html = true;
            }
            else if (c == '.') {
                optional = true;
            }
            controlsSize++;
        } else {
            // Once we hit a character that's not a control character,
            // we break out of the loop.
            break;
        }
    }

    // Parsing the name and sub-variables, omitting the controls
    auto r = utils::split(std::string_view(props.name).substr(controlsSize), '.');
    
    // Assigning the action name without the control characters.
    if (r.empty())
        throw InvalidArgumentException("No variables found: ", props.name);

    props.optional = optional;
    props.html = html;
    
    const size_t N = r.size();
    props.subs.resize(N - 1);
    for (size_t i=1, N = r.size(); i<N; ++i) {
        props.subs[i - 1] = parse_words(r[i]);
        //props.subs[i - 1] = r[i];
    }

    if(r.front() != std::string_view(props.name))
        props.name = r.front();
    return props;
}

UnpreparedPatterns parse_words(std::string_view pattern) {
    UnpreparedPatterns result;
    std::stack<std::size_t> nesting_start_positions;
    bool comment_out = false;
    std::optional<std::size_t> last_start;

    for(std::size_t i = 0; i < pattern.size(); ++i) {
        char ch = pattern[i];
        if(nesting_start_positions.empty()) {
            if(not last_start
               && not comment_out
               && ch != '{')
            {
                last_start = i;
            }
            
            switch(ch) {
                case '\\':
                    if(not comment_out) {
                        comment_out = true;
                        if(last_start && *last_start < i)
                            result.push_back(pattern.substr(*last_start, i - *last_start));
                        last_start = i + 1;
                        
                    } else {
                        comment_out = false;
                    }
                    break;
                case '{':
                    if(not comment_out){
                        nesting_start_positions.push(i + 1);
                        if(last_start)
                            result.push_back(pattern.substr(*last_start, i - *last_start));
                        last_start.reset();
                    }
                    comment_out = false;
                    break;
                case '}':
                    if(!comment_out)
                        throw InvalidSyntaxException("Mismatched closing brace at position ", i);
                    comment_out = false;
                    break;
                default:
                    if(comment_out) {
                        FormatWarning("Invalid escape sequence at position ", i, ": ", ch, ". Only braces need to be escaped.");
                        comment_out = false;
                    }
            }
        } else {
            if(comment_out) {
                comment_out = false;
                
            } else if(ch == '\\') {
                comment_out = true;
                /*if(last_start && *last_start < i)
                     result.push_back(pattern.substr(*last_start, i - *last_start));
                last_start = i + 1;*/
                
            } else if(ch == '}') {
                if(nesting_start_positions.empty()) {
                    throw InvalidSyntaxException("Mismatched closing brace at position ", i);
                }
                
                std::size_t start_pos = nesting_start_positions.top();
                nesting_start_positions.pop();

                if(nesting_start_positions.empty()) {
                    std::string_view current_word = pattern.substr(start_pos, i - start_pos);
                    if(current_word.empty()) {
                        throw InvalidSyntaxException("Empty braces at position ", i);
                    }
                    
                    result.push_back(extract_unprepared(current_word));
                    last_start = i + 1;
                }
                
            } else if(ch == '{') {
                nesting_start_positions.push(i + 1);
                if(last_start)
                    result.push_back(pattern.substr(*last_start, i - *last_start));
                last_start.reset();
            } else {
                if(nesting_start_positions.empty()) {
                    throw InvalidSyntaxException("Mismatched opening brace at position ", i);
                }
                //nesting.top() += ch;
            }
        }
    }

    if(not nesting_start_positions.empty()) {
        throw InvalidSyntaxException("Mismatched opening brace");
    }
    
    if(comment_out) {
        // Trailing backslash without a following character
        throw InvalidSyntaxException("Trailing backslash");
    }

    if(last_start
       && *last_start < pattern.length())
    {
        result.push_back(pattern.substr(*last_start, pattern.length() - *last_start));
    }
    
    return result;
}

Prepared* Prepared::get(const Unprepared& unprepared) {
    Prepared *prepared = new Prepared{
        .original = unprepared.original
    };
    
    prepared->parameters.reserve(unprepared.parameters.size());
    for(auto &o : unprepared.parameters) {
        assert(std::holds_alternative<UnpreparedPatterns>(o));
        PreparedPatterns r;
        for(auto &p : std::get<UnpreparedPatterns>(o)) {
            std::visit([&r](auto& obj) {
                using T = std::decay_t<decltype(obj)>;
                if constexpr(std::same_as<T, std::string_view>) {
                    r.push_back(PreparedPattern::make_sv(obj));
                } else if constexpr(std::same_as<T, Unprepared>) {
                    r.push_back(PreparedPattern::make_prepared(Prepared::get(obj)));
                    
                } else {
                    static_assert(std::same_as<T, void>, "unknown type.");
                }
            }, p);
        }
        prepared->parameters.push_back(std::move(r));
    }
    
    prepared->subs.reserve(unprepared.subs.size());
    for(auto &o : unprepared.subs) {
        assert(std::holds_alternative<UnpreparedPatterns>(o));
        PreparedPatterns r;
        for(auto &p : std::get<UnpreparedPatterns>(o)) {
            std::visit([&r](auto& obj) {
                using T = std::decay_t<decltype(obj)>;
                if constexpr(std::same_as<T, std::string_view>) {
                    r.push_back(PreparedPattern::make_sv(obj));
                } else if constexpr(std::same_as<T, Unprepared>) {
                    r.push_back(PreparedPattern::make_prepared(Prepared::get(obj)));
                    
                } else {
                    static_assert(std::same_as<T, void>, "unknown type.");
                }
            }, p);
        }
        prepared->subs.push_back(std::move(r));
    }
    
    auto& props = prepared->resolved;
    props.name = unprepared.name;
    props.html = unprepared.html;
    props.optional = unprepared.optional;
    return prepared;
}

std::string Prepared::toStr() const {
    std::string result = (std::string)resolved.name;
    if(not resolved.subs.empty()) {
        for(size_t j = 0; j < resolved.subs.size(); ++j) {
            result.append(".");
            result.append(resolved.subs[j]);
        }
    }
    result.append("{");
    for(size_t j = 0; j < parameters.size(); ++j) {
        auto &parm = parameters[j];
        if(j > 0)
            result += ",";
        if(parm.size() > 1)
            result += "(";
        
        for(size_t i = 0; i < parm.size(); ++i) {
            if(i > 0)
                result += ",";
            
            result += parm[i].toStr();
        }
        
        if(parm.size() > 1)
            result += ")";
    }
    result += "}";
    return result;
}

UnresolvedStringPattern::~UnresolvedStringPattern() {
    for(auto prepared : all_patterns)
        delete prepared;
}

void PreparedPattern::resolve(std::string& c, UnresolvedStringPattern& pattern, const gui::dyn::Context& context, gui::dyn::State& state)
{
    auto is_cache_valid = [&state](const Prepared& prepared) -> bool {
        if(not prepared.cached().has_value()) {
            return false;
        }
        
        auto handler = state._current_object_handler.lock();
        if(not handler) {
            return true;
        }
        
        return prepared._cached_variable_versions.has_value()
            && prepared._cached_variable_versions->global == handler->variable_values_version()
            && prepared._cached_variable_versions->scoped == handler->scoped_variable_values_version();
    };
    
    switch(type) {
        case SV:
            c.append(value.sv.data(), value.sv.size());
            break;
        case PREPARED:
            if(auto& o = value.prepared;
               is_cache_valid(*o))
            {
                auto& sv = o->cached().value();
                c.append(sv.data(), sv.size());
            } else {
                o->resolve(pattern, c, context, state);
            }
            break;
        case POINTER:
            if(auto& o = *value.ptr;
               is_cache_valid(o))
            {
                auto& sv = o.cached().value();
                c.append(sv.data(), sv.size());
            } else {
                o.resolve(pattern, c, context, state);
            }
            break;
        default:
#ifndef NDEBUG
            throw InvalidArgumentException("Illegal type ", type, ".");
#endif
            break;
    }
}

bool PreparedPattern::operator==(const PreparedPattern& other) const noexcept {
    if(type == other.type) {
        switch (type) {
            case SV:
                return value.sv == other.value.sv;
            case PREPARED:
                return value.prepared == other.value.prepared;
            case POINTER:
                return value.ptr == other.value.ptr;
            default:
                break;
        }
    }
    return false;
}

bool PreparedPattern::operator<(const PreparedPattern& other) const noexcept {
    if(type != other.type)
        return type < other.type;
    switch (type) {
        case SV:
            return value.sv < other.value.sv;
        case PREPARED:
            return value.prepared->original < other.value.prepared->original;
        case POINTER:
            return value.ptr && other.value.ptr ? value.ptr->original < other.value.ptr->original : value.ptr < other.value.ptr;
            
        default:
            break;
    }
    return false;
}

std::string PreparedPattern::toStr() const {
    std::string str = "<";
    switch (type) {
        case SV:
            str.append(value.sv);
            break;
        case PREPARED:
            str.append(value.prepared->toStr());
            break;
        case POINTER:
            str.append(hex(value.ptr).toStr());
            break;
            
        default:
            break;
    }
    str += ">";
    return str;
}

template<typename ApplyF, typename ErrorF>
    requires std::invocable<ApplyF, std::string&, VarBase_t&, const VarProps&>
inline auto resolve_variable(std::string& output, const VarProps& props, const Context& context, State& state, ApplyF&& apply, ErrorF&& error)
{
    try {
        auto cached = props.subs.empty()
                        ? state.cached_variable_value(props.name, state._current_object_handler.lock())
                        : state.cached_variable_value(utils::concat_views(props.name, props.subs), state._current_object_handler.lock());
        if(cached)
        {
            output.append(cached.value());
            return;
            
        } else if(context.has(props.name, state))
        {
            auto variable = context.variable(props.name, state);
            apply(output, *variable, props);
            return;
            
        } else if(props.name == "if") {
            REQUIRE_AT_LEAST(2, props);
            //[[maybe_unused]] CTimer ctimer("if");
            bool condition = props.parameters.front() == "true";
            //Print("* condition ", props.parameters.front()," => ", condition);
            if(condition)
                utils::fast_fromstr(output, props.parameters[1]);
            //output += Meta::fromStr<std::string>(props.parameters[1]);
            else if(props.parameters.size() == 3)
                utils::fast_fromstr(output, props.parameters[2]);
            //output += Meta::fromStr<std::string>(props.parameters[2]);
            return;
        } else if(props.name == "for") {
#ifndef NDEBUG
            Print("For loop: ", props);
#endif
            throw InvalidArgumentException("Not expecting a for loop here: ", props);
            return;
            
        } else if(auto it = context.defaults.variables.find(props.name); it != context.defaults.variables.end()) {
            //Print("* defaults contain ", props);
            //Print("* => ", it->second->value<std::string>(context, state));
            //[[maybe_unused]] CTimer ctimer("custom var");
            if(not props.subs.empty()) {
                /// here we might need to check for x,y / array accessors
                auto &key = props.subs.front();
                if(auto index = is_in_index(key, "x", "y", "w", "h");
                   index < 4u) {
                    /// treated special
                    std::string tmp = it->second->value<std::string>(context, state);
                    
                    auto parts = util::parse_array_parts(util::truncate(tmp));
                    if(index >= parts.size()) {
                        throw InvalidArgumentException("String 'array' ", tmp, " does not have ",index," elements and cant be accessed using .x/.y/.w/.h");
                    }
                    utils::fast_fromstr(output, parts[index]);
                    return;
                    
                } else if(is_valid_number<uint16_t>(key)) {
                    std::string tmp = it->second->value<std::string>(context, state);
                    auto parts = util::parse_array_parts(util::truncate(tmp));
                    
                    auto index = Meta::fromStr<uint8_t>(key);
                    utils::fast_fromstr(output, parts.at(index));
                    return;
                }
            }
            
            utils::fast_fromstr(output, it->second->value<std::string>(context, state));
            return;
            
        } else if(auto lock = state._current_object_handler.lock();
                  lock != nullptr)
        {
            //Print("* ", props, " testing current handler");
            if(auto ptr = lock->retrieve_named(props.name);
               ptr != nullptr)
            {
                auto v = get_modifier_from_object(ptr.get(), props);
                if(v.has_value()) {
                    //Print("* ", props, " modifiers ", v.value());
                    utils::fast_fromstr(output, v.value());
                    //output += Meta::fromStr<std::string>(v.value());
                    return;
                }
            }
            
        }
        
    } catch(const std::exception& ex) {
        if constexpr(std::invocable<ErrorF, std::string&, bool, const std::string&>) {
            error(output, props.optional, std::string(ex.what()));
            return;
        }
    }
    
    if constexpr(std::invocable<ErrorF, std::string&, bool, const std::string&>) {
        error(output, props.optional, "Cannot find property "+props.toStr()+" in context: "+Meta::toStr(extract_keys(context.variables)));
        return;
    }
    else if constexpr(std::invocable<ErrorF, std::string&, bool>)
        error(output, props.optional);
    else
        error(output);
}


void resolve_parameter(std::string& c, UnresolvedStringPattern& pattern, PreparedPatterns& patterns, const cmn::gui::dyn::Context& context, cmn::gui::dyn::State& state) {
    for(auto& o : patterns) {
        o.resolve(c, pattern, context, state);
    }
}


void apply_html(std::string& output, const VarProps& modifiers, StringLike auto&& str) {
    if(modifiers.html)
        output += settings::htmlify(std::forward<decltype(str)>(str));
    else
        output += str;
};

template<typename String> void access_sub_field(std::string& output, const VarProps& modifiers, const glz::generic& json, std::span<String> subs);
template<typename String> void access_sub_field(std::string& output, const VarProps& modifiers, const Size2& json, std::span<String> subs);
template<typename String> void access_sub_field(std::string& output, const VarProps& modifiers, const Vec2& json, std::span<String> subs);

template<typename String>
void access_sub_field(std::string& output, const VarProps& modifiers, const Size2& size, std::span<String> subs)
{
    const auto& name = subs.front();
    //subs = subs.subspan(1); /// we are not continuing here anyway
    
    if(not is_in(name, "w", "h", "width", "height")) {
        if(not modifiers.optional)
            apply_html(output, modifiers, "null");
        return;
    }

    if(is_in(name, "w", "width"))
        apply_html(output, modifiers, Meta::toStr(size.width));
    else
        apply_html(output, modifiers, Meta::toStr(size.height));
}

template<typename String>
void access_sub_field(std::string& output, const VarProps& modifiers, const Vec2& pos, std::span<String> subs)
{
    const auto& name = subs.front();
    //subs = subs.subspan(1); /// we are not going further here anyway.
    
    if(not is_in(name, "x", "y")) {
        if(not modifiers.optional)
            apply_html(output, modifiers, "null");
        return;
    }

    if(is_in(name, "x"))
        apply_html(output, modifiers, Meta::toStr(pos.x));
    else
        apply_html(output, modifiers, Meta::toStr(pos.y));
}

template<typename Field, typename String>
void access_map_subfield(std::string& output, const VarProps& modifiers, std::span<String> subs, Field&& field) {
    if(field.is_array())  {
        if(not subs.empty()) {
            const auto& name = subs.front();
            if(is_valid_number<uint16_t>(name)) {
                auto idx = Meta::fromStr<uint16_t>(name);
                Print("* Accessing array index ", idx, " of ", field);
                subs = subs.subspan(1);
                
                auto ptr = field.access_array(idx);
                access_map_subfield(output, modifiers, subs, *ptr);
                return;
            } else {
                if(not modifiers.optional)
                    apply_html(output, modifiers, "null");
                return;
            }
        }
        
        /// just output the value then...

    } else if(field.template is_type<glz::generic>()) {
        // need another key
        if(not subs.empty()) {
            access_sub_field(output, modifiers, field.template value<glz::generic>(), subs);
            return;
        }
        
    } else if(field.template is_type<sprite::Map>()) {
        throw InvalidArgumentException("Not currently implemented to access a sprite::Map within a sprite::Map.");
    }
    
    apply_html(output, modifiers, Meta::fromStr<std::string>(field.valueString()));
}

template<typename String>
void access_sub_field(std::string& output, const VarProps& modifiers, const sprite::Map& map, std::span<String> subs)
{
    const auto& name = subs.front();
    subs = subs.subspan(1);
    
    if(not map.has(name)) {
        if(not modifiers.optional)
            apply_html(output, modifiers, "null");
        return;
        //throw InvalidArgumentException(map, " does not contain field ", name);
    }
    
    sprite::ConstReference field = map.at(name);
    if(field.valid())
        access_map_subfield(output, modifiers, std::move(subs), field.get());
    else if(not modifiers.optional)
        apply_html(output, modifiers, "null");
}

template<typename String>
void access_sub_field(std::string& output, const VarProps& modifiers, const glz::generic& json, std::span<String> subs)
{
    if(subs.empty()) {
        if(json.is_string()) {
            apply_html(output, modifiers, json.template get<std::string>());
            return;
        }
        
        apply_html(output, modifiers, Meta::toStr(json));
        return;
    }
    
    const auto& name = subs.front();
    subs = subs.subspan(1);
    
    if(json.is_array()) {
        const auto& array = json.get_array();
        if(size_t index = is_in_index(name, "x", "y", "w", "h");
           index < 4)
        {
            const auto &obj = array[index];
            if(obj.is_string()) {
                apply_html(output, modifiers, obj.get_string());
            } else
                apply_html(output, modifiers, Meta::toStr(obj));
            return;
        }
        else if(is_valid_number<uint16_t>(name)) {
            auto index = Meta::fromStr<uint16_t>(name);
            const auto &obj = array[index];
            if(obj.is_string()) {
                apply_html(output, modifiers, obj.get_string());
            } else
                apply_html(output, modifiers, Meta::toStr(obj));
            return;
        }
        
        throw InvalidArgumentException("Cannot access index ",name," in array ", json);
        
    } else if(json.is_object()) {
        auto &obj = json.get_object();
        auto it = obj.find(name);
        if(it == obj.end())
            throw InvalidArgumentException("Object ", json, " does not contain field ", name, " contains: ", extract_keys(obj));
        
        if(const glz::generic& field = it->second;
           not subs.empty())
        {
            access_sub_field(output, modifiers, field, subs);
            return;
        } else {
            if(field.is_string())
                apply_html(output, modifiers, field.get_string());
            else
                apply_html(output, modifiers, Meta::toStr(field));
            return;
        }
    }
    
    throw InvalidArgumentException("Unknown access type to ", json, " with ", subs);
}

void handle_sub_objects(std::string& output, cmn::gui::dyn::VarBase_t& variable, const VarProps& modifiers) {
    
    try {
        if(modifiers.subs.empty()) {
            apply_html(output, modifiers, variable.value_string(modifiers));
        } else if(variable.is<Size2>()) {
            variable.access_value<Size2>([&](auto&& value){
                std::span subs{ modifiers.subs };
                access_sub_field(output, modifiers, value, subs);
            }, modifiers);
            
        } else if(variable.is<Vec2>()) {
            variable.access_value<Vec2>([&](auto&& value){
                std::span subs{ modifiers.subs };
                access_sub_field(output, modifiers, value, subs);
            }, modifiers);
            
        } else if(variable.is<Range<Frame_t>>()) {
            if(modifiers.subs.front() == "start")
                apply_html(output, modifiers,
                           Meta::toStr(variable.value<Range<Frame_t>>(modifiers).start));
            else if(modifiers.subs.front() == "end")
                apply_html(output, modifiers,
                           Meta::toStr(variable.value<Range<Frame_t>>(modifiers).end));
            else
                throw InvalidArgumentException("Sub ",modifiers," of Range<Frame_t> is not valid.");
            
            
        } else if(variable.is<glz::generic>()) {
            variable.access_value<glz::generic>([&](auto&& value){
                std::span subs{ modifiers.subs };
                access_sub_field(output, modifiers, value, subs);
            }, modifiers);
        }
        else if(variable.is<sprite::Map>())
        {
            variable.access_value<sprite::Map>([&](auto&& value){
                std::span subs{ modifiers.subs };
                access_sub_field(output, modifiers, value, subs);
            }, modifiers);
            
        } else {
            if(not modifiers.subs.empty()) {
                /// need to check for special subs
                auto& key = modifiers.subs.front();
                if(auto index = is_in_index(key, "x", "y", "w", "h");
                   index < 4)
                {
                    std::string tmp = variable.value_string(modifiers);
                    auto parts = util::parse_array_parts(util::truncate(tmp));
                    if(index >= parts.size()) {
                        throw InvalidArgumentException("Expecting at least ",index," values in string 'array' ", tmp, " for index ", key," (=>",index,").");
                    }
                    
                    apply_html(output, modifiers, parts[index]);
                    return;
                    
                } else if(is_valid_number<uint16_t>(key)) {
                    auto index = Meta::fromStr<uint16_t>(key);
                    std::string tmp = variable.value_string(modifiers);
                    auto parts = util::parse_array_parts(util::truncate(tmp));
                    
                    apply_html(output, modifiers, parts.at(index));
                    //Print("* got unknown variable ", modifiers, " with subs => ", tmp);
                    return;
                } else {
                    /// here we trust the variable function to handle the modifiers
                    //throw InvalidArgumentException("Received subs ", modifiers.subs, " to variable of unknown/string type => ", variable.value_string(modifiers));
                }
            }
            apply_html(output, modifiers,
                       variable.value_string(modifiers));
        }
        
        //if(modifiers.html)
        //    return settings::htmlify(ret);
        return;
        
    } catch(const std::exception& ex) {
        if(not modifiers.optional)
            throw;//FormatExcept("Exception: ", ex.what(), " in variable: ", modifiers);
        output += modifiers.optional ? "" : "null";
    }
}

void Prepared::resolve(UnresolvedStringPattern& pattern, std::string& str, const gui::dyn::Context& context, gui::dyn::State& state)
{
    auto& props = resolved;
    auto& parms = props.parameters;
    size_t index = str.length();
    
    auto update_cache = [&](){
        if(has_children) {
            _cached_value = std::string_view(str).substr(index);
            if(auto handler = state._current_object_handler.lock();
               handler)
            {
                _cached_variable_versions = CachedVariableVersions{
                    .global = handler->variable_values_version(),
                    .scoped = handler->scoped_variable_values_version()
                };
            } else {
                _cached_variable_versions.reset();
            }
        }
    };
    
    props.subs.resize(subs.size());
    
    for(size_t i = 0; i < subs.size(); ++i) {
        props.subs[i].clear();
        for(auto &p : subs[i]) {
            p.resolve(props.subs[i], pattern, context, state);
        }
    }
    
    if(resolved.name == "if") {
        /// we have a special case here where we want lazy evaluation
        /// lets assume three parameters here:
        if(not is_in(parameters.size(), 2u, 3u))
            throw InvalidArgumentException("An if statement (", parameters, ") accepts exactly 2 or 3 parameters.");
        
        parms.resize(parameters.size());
        auto& condition = parms[0];
        condition.clear();
        if(props.optional) {
            try {
                resolve_parameter(condition, pattern, parameters[0], context, state);
            } catch(...) {
                FormatWarning("Failed to resolve optional if condition: ", props, " => ", parameters[0]);
                condition = "null";
            }
        } else {
            resolve_parameter(condition, pattern, parameters[0], context, state);
        }
        
        if(cmn::benchmark_head::dyn::convert_to_bool(condition)) {
            parms[0] = "true";
            parms[1].clear();
            resolve_parameter(parms[1], pattern, parameters[1], context, state);
            if(parms.size() == 3)
                parms[2] = "null";
        } else if(parms.size() == 3) {
            parms[0] = "false";
            parms[2].clear();
            resolve_parameter(parms[2], pattern, parameters[2], context, state);
            parms[1] = "null";
        }
        
    } else if(resolved.name == "&&" || resolved.name == "||") {
        const bool is_and = resolved.name == "&&";
        //bool condition = is_and ? true : false;
        //parms.clear();
        //parms.reserve(parameters.size());
        std::string value;

        // Resolve operands in order so a decisive value protects the rest.
        for(auto& parameter : parameters) {
            value.clear();
            resolve_parameter(value, pattern, parameter, context, state);
            const bool truth_value = cmn::benchmark_head::dyn::convert_to_bool(value);
            /*if(is_and)
                condition = condition && truth_value;
            else
                condition = condition || truth_value;*/
            
            if((is_and && not truth_value) || (not is_and && truth_value)) {
                str.append(truth_value ? "true" : "false");
                update_cache();
                return;
            }
        }
        
        if(is_and)
            str.append("true");
        else
            str.append("false");
        update_cache();
        return;

    } else if(resolved.name == "for") {
        if(not is_in(parameters.size(), 2u, 3u))
            throw InvalidArgumentException("An if statement (", parameters, ") only accepts 2 or 3 parameters.");
        
        size_t items_index = 0u;
        size_t fn_index = 1u;
        std::string variable_name = "i";
        
        if(parameters.size() == 3u) {
            items_index = 1u;
            fn_index = 2u;
            variable_name.clear();
            resolve_parameter(variable_name, pattern, parameters[0u], context, state);
        }
        
        parms.resize(1u);
        parms[0].clear();
        resolve_parameter(parms[0], pattern, parameters[items_index], context, state);
        
        if(not utils::beginsWith(parms[0], '[')
            || not utils::endsWith(parms[0], ']'))
        {
            throw InvalidArgumentException("An if statement (", parameters, ") requires an array-like second parameter, received ", parms[0]);
        }
        
        size_t index = str.length();
        auto items = util::truncate(parms[0]);
        
        str.append("[");
        
        State _state;
        _state._collectors = state._collectors;
        _state._current_object_handler = state._current_object_handler;
        //auto handler = std::make_shared<CurrentObjectHandler>();
        //state._current_object_handler = handler;
        auto handler = _state._current_object_handler.lock();
        if(not handler) {
            handler = std::make_shared<CurrentObjectHandler>();
            _state._current_object_handler = std::weak_ptr(handler);
        }
        
        size_t idx = 0;
        for(auto item : util::parse_array_parts(items)) {
            if(idx > 0)
                str.append(",");
            
            auto scope = handler->scope();
            auto idx_str = Meta::toStr(idx);
            scope.set(variable_name, item);
            scope.set("index", idx_str);
            resolve_parameter(str, pattern, parameters[fn_index], context, _state);
            
            ++idx;
        }
        
        str.append("]");
        
        if(has_children) {
            _cached_value = std::string_view(str).substr(index);
            if(auto handler = state._current_object_handler.lock();
               handler)
            {
                _cached_variable_versions = CachedVariableVersions{
                    .global = handler->variable_values_version(),
                    .scoped = handler->scoped_variable_values_version()
                };
            } else {
                _cached_variable_versions.reset();
            }
        }
        
        return;
        
    } else {
        const size_t N = parameters.size();
        parms.resize(N);
        for(size_t i = 0; i < N; ++i) {
            auto &p = parameters[i];
            parms[i].clear();
            resolve_parameter(parms[i], pattern, p, context, state);
        }
    }
    

    resolve_variable(str, props, context, state, [](std::string& output, cmn::gui::dyn::VarBase_t& variable, const gui::dyn::VarProps& modifiers)
    {
        handle_sub_objects(output, variable, modifiers);
        
    }, [&props, &str](std::string&, bool optional, const std::string& ex = "") {
        if(not optional)
            throw InvalidArgumentException("Failed to evaluate ", props, ": ", no_quotes(not ex.empty() ? ex : std::string("<null>")));
        if(not props.optional)
            str.append("null");
    });
    
    update_cache();
}

std::string UnresolvedStringPattern::realize(const gui::dyn::Context& context, gui::dyn::State& state) {
    [[maybe_unused]] CTimer ctimer(*original);
    std::string str;
    bool has_length = typical_length.has_value();
    if(has_length)
        str.reserve(typical_length.value() * 1.5);
    
    for(auto ptr : all_patterns) {
        ptr->reset();
    }
    
    for(auto& object : objects) {
        object.resolve(str, *this, context, state);
    }
    
    if(not has_length)
        typical_length = str.length();
    return str;
}

std::string UnresolvedStringPattern::toStr() const {
    // Produce a readable string representation of the pattern by
    // concatenating the textual form of each stored PreparedPattern.
    // We separate individual parts with a single space to match the
    // formatting style used elsewhere in the debug helpers.
    std::string result;
    result.reserve(objects.size() * 16);   // rough pre‑allocation

    bool first = true;
    for (const auto& obj : objects) {
        if (!first)
            result += ' ';
        result += obj.toStr();
        first = false;
    }
    return result;
}

UnresolvedStringPattern UnresolvedStringPattern::prepare_static(std::string_view _str) {
    auto result = UnresolvedStringPattern{};
    result.original = std::make_unique<std::string>(_str);
    std::string_view str{*result.original};
    
    result.objects.push_back(PreparedPattern::make_sv(str));
    
    return result;
}

UnresolvedStringPattern UnresolvedStringPattern::prepare(std::string_view _str) {
    auto result = UnresolvedStringPattern{};
    result.original = std::make_unique<std::string>(_str);
    std::string_view str{*result.original};
    
    auto unprepared = parse_words(str);
    std::queue<Unprepared*> patterns;
    for(auto& r : unprepared) {
        std::visit([&](auto& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr(std::same_as<T, Unprepared>) {
                patterns.push(&arg);
                
            } else if constexpr(std::same_as<T, std::string_view>) {
                //result.objects.push_back(arg);
            } else {
                static_assert(std::same_as<T, void>, "Unknown Type.");
            }
            
        }, r);
    }
    
    while(not patterns.empty()) {
        auto unprep = patterns.front();
        patterns.pop();
        
        for(size_t i = 0; i < unprep->parameters.size(); ++i) {
            assert(std::holds_alternative<std::string_view>(unprep->parameters[i]));
            unprep->parameters[i] = parse_words(std::get<std::string_view>(unprep->parameters[i]));
        }
        
        for(auto &v : unprep->parameters) {
            if(std::holds_alternative<std::string_view>(v))
                continue;
            
            for(auto &obj : std::get<UnpreparedPatterns>(v)) {
                std::visit([&patterns](auto& obj){
                    using T = std::decay_t<decltype(obj)>;
                    if constexpr(std::same_as<T, Unprepared>) {
                        patterns.push(&obj);
                    }
                }, obj);
            }
        }
    }
    
    for(auto &obj : unprepared) {
        std::visit([&result](auto& obj) {
            using T = std::decay_t<decltype(obj)>;
            if constexpr(std::same_as<T, std::string_view>) {
                result.objects.push_back(PreparedPattern::make_sv(obj));
            } else if constexpr(std::same_as<T, Unprepared>) {
                result.objects.push_back(PreparedPattern::make_prepared(Prepared::get(obj)));
                
            } else {
                static_assert(std::same_as<T, void>, "unknown type.");
            }
        }, obj);
    }
    
    /// second run, try to remove duplicates
    /// document current state:
    struct Preview {
        std::string_view name;
        bool operator<(const Preview& other) const {
            return name < other.name;
        }
    };
    
    std::deque<PreparedPattern*> q;
    std::map<Preview, size_t> counts;
    std::map<Preview, Prepared*> duplicates;
    
    for(auto& r : result.objects) {
        if(r.type == PreparedPattern::PREPARED)
            q.push_front(&r);
    }
    
    while(not q.empty()) {
        PreparedPattern* obj = q.front();
        q.pop_front();
        
        assert(obj->type == PreparedPattern::PREPARED);
        auto &prepared = obj->value.prepared;
        Preview preview{
            .name = prepared->original
        };
        counts[preview]++;
        
        auto it = duplicates.find(preview);
        if(it != duplicates.end()) {
            assert(not obj->value.prepared->has_children);
            delete obj->value.prepared;
            obj->value.ptr = (Prepared*)it->second;
            obj->type = PreparedPattern::POINTER;
            it->second->has_children = true;
            continue;
        }
        
        duplicates[preview] = prepared;
        result.all_patterns.push_back(prepared);
        
        for(auto &p : prepared->parameters) {
            for(auto &o : p) {
                if(o.type == PreparedPattern::PREPARED)
                    q.push_front(&o);
            }
        }

        for(auto &p : prepared->subs) {
            for(auto &o : p) {
                if(o.type == PreparedPattern::PREPARED)
                    q.push_front(&o);
            }
        }
    }
    
    /*{
        std::stringstream ss;
        ss << "[";
        
        for(auto& r : result.objects) {
            std::visit([&](auto&& obj){
                output_object(ss, std::forward<std::decay_t<decltype(obj)>>(obj));
            }, r);
        }
        
        ss << "]";
        Print(no_quotes(ss.str()));
    }*/
    
    /*for(auto &[key, value] : counts) {
        Print("* ", key.name, ": ", value);
    }*/
    
    return result;
}


}
}
