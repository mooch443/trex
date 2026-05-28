#pragma once

#include <commons.pc.h>
#include <expected>

namespace Python {

using validation_result_t = std::expected<void, std::string>;

template<typename T>
constexpr bool always_false_v = false;

template<typename T>
using clean_t = std::remove_cvref_t<T>;

template<typename T>
constexpr bool string_like_v = cmn::StringLike<clean_t<T>>;

template<typename ExpectedT>
using parsed_field_t = std::conditional_t<
    std::same_as<clean_t<ExpectedT>, std::string_view>,
    std::string,
    clean_t<ExpectedT>>;

template<typename T, cmn::StringLike Key>
validation_result_t read_typed_field(const glz::json_t& response,
                                     Key&& key,
                                     T& out_value)
{
    using value_t = clean_t<T>;
    const auto key_sv = cmn::utils::string_like_view(key);
    const auto key_str = std::string(key_sv);

    if(not response.contains(key_str)) {
        return std::unexpected("missing `" + key_str + "`");
    }

    const auto& value = response.at(key_str);
    if constexpr(std::same_as<value_t, bool>) {
        if(not value.is_boolean()) {
            return std::unexpected("expected boolean `" + key_str + "`");
        }
        out_value = value.template get<bool>();
    } else if constexpr(std::integral<value_t> || std::floating_point<value_t>) {
        if(not value.is_number()) {
            return std::unexpected("expected numeric `" + key_str + "`");
        }
        out_value = value.template get<value_t>();
    } else if constexpr(std::same_as<value_t, std::string>) {
        if(not value.is_string()) {
            return std::unexpected("expected string `" + key_str + "`");
        }
        out_value = value.template get<std::string>();
    } else {
        static_assert(always_false_v<T>, "Unsupported type in read_typed_field.");
    }

    return {};
}

template<typename L, typename R>
bool values_match(const L& lhs, const R& rhs, float epsilon = 1e-6f) {
    if constexpr(string_like_v<L> && string_like_v<R>) {
        return cmn::utils::string_like_view(lhs) == cmn::utils::string_like_view(rhs);
    } else if constexpr(std::floating_point<clean_t<L>> && std::floating_point<clean_t<R>>) {
        using wide_t = std::common_type_t<clean_t<L>, clean_t<R>>;
        return std::fabs(static_cast<wide_t>(lhs) - static_cast<wide_t>(rhs))
               <= static_cast<wide_t>(epsilon);
    } else {
        return lhs == rhs;
    }
}

template<cmn::StringLike ModuleName,
         cmn::StringLike SetterName,
         typename ExtraValidation = std::nullptr_t>
bool validate_setter_response(ModuleName&& module_name,
                              SetterName&& setter_name,
                              const std::optional<glz::json_t>& result,
                              ExtraValidation&& extra_validation = nullptr)
{
    const auto module_name_sv = cmn::utils::string_like_view(module_name);
    const auto setter_name_sv = cmn::utils::string_like_view(setter_name);

    if(not result.has_value()) {
        FormatWarning(module_name_sv, " ", setter_name_sv, " did not return a value.");
        return false;
    }
    if(not result->is_object()) {
        FormatWarning(module_name_sv, " ", setter_name_sv, " did not return an object.");
        return false;
    }
    if(not result->contains("ok")
       || not result->at("ok").is_boolean()
       || not result->at("ok").get<bool>())
    {
        FormatWarning(module_name_sv, " ", setter_name_sv, " returned success=false.");
        return false;
    }

    if constexpr(not std::same_as<clean_t<ExtraValidation>, std::nullptr_t>) {
        static_assert(std::invocable<ExtraValidation, const glz::json_t&>,
                      "extra_validation must be callable with (const glz::json_t&).");
        using extra_result_t = std::invoke_result_t<ExtraValidation, const glz::json_t&>;
        static_assert(std::same_as<clean_t<extra_result_t>, validation_result_t>,
                      "extra_validation must return std::expected<void, std::string>.");

        if(auto maybe_error = std::invoke(std::forward<ExtraValidation>(extra_validation), *result);
           not maybe_error.has_value())
        {
            FormatWarning(module_name_sv, " ", setter_name_sv,
                          " returned invalid response: ", maybe_error.error(), ".");
            return false;
        }
    }

    return true;
}

template<typename T,
         cmn::StringLike ModuleName,
         cmn::StringLike SetterName,
         cmn::StringLike Key>
bool validate_setter_response_with_value(ModuleName&& module_name,
                                         SetterName&& setter_name,
                                         const std::optional<glz::json_t>& result,
                                         Key&& key,
                                         const T& expected_value,
                                         float epsilon = 1e-6f)
{
    const auto key_str = std::string(cmn::utils::string_like_view(key));

    parsed_field_t<T> expected_stored{};
    if constexpr(std::same_as<clean_t<T>, std::string_view>) {
        expected_stored = std::string(expected_value);
    } else {
        expected_stored = parsed_field_t<T>(expected_value);
    }

    return validate_setter_response(
        std::forward<ModuleName>(module_name),
        std::forward<SetterName>(setter_name),
        result,
        [key_str, expected_stored, epsilon](const glz::json_t& response) -> validation_result_t {
            parsed_field_t<T> returned_value{};
            if(auto maybe_error = read_typed_field(response, key_str, returned_value);
               not maybe_error.has_value())
            {
                return std::unexpected(maybe_error.error());
            }

            if(not values_match(returned_value, expected_stored, epsilon)) {
                return std::unexpected("expected " + key_str + "=" + Meta::toStr(expected_stored)
                                     + " but got " + Meta::toStr(returned_value));
            }
            return {};
        });
}

} // namespace track::py
