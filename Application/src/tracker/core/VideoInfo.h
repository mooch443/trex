#pragma once

#include <commons.pc.h>
#include <file/PathArray.h>
#include <misc/frame_t.h>

namespace cmn {

struct VideoInfo {
    file::PathArray base;
    Size2 resolution;
    short framerate{0};
    bool finite{false};
    Frame_t length;
    Frame_t current_frame_index; // Pipeline index; invalid until supplied.

    glz::json_t to_json() const;
    std::string toStr() const;
};

}

template <>
struct glz::meta<cmn::VideoInfo> {
    using T = cmn::VideoInfo;
    static constexpr auto value = glz::object(
        "base", &T::base,
        "resolution", &T::resolution,
        "framerate", &T::framerate,
        "finite", &T::finite,
        "length", &T::length,
        "current_frame_index", &T::current_frame_index
    );
};

template <>
struct glz::to<glz::JSON, cmn::VideoInfo> {
    template <auto Opts>
    static void op(const cmn::VideoInfo& value, auto&&... args) {
        auto json = value.to_json();
        glz::serialize<glz::JSON>::op<Opts>(json, std::forward<decltype(args)>(args)...);
    }
};
