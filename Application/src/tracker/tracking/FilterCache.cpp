#include "FilterCache.h"
#include <gui/Transform.h>
#include <misc/Image.h>
#include <misc/idx_t.h>
#include <misc/frame_t.h>
#include <tracker/misc/default_config.h>
#include <grabber/misc/default_config.h>
#include <misc/PVBlob.h>
#include <misc/ranges.h>
#include <misc/Timer.h>
#include <tracking/Tracker.h>
#include <processing/Background.h>


using namespace default_config;

namespace track {
namespace image {

std::tuple<Image::Ptr, Vec2> normalize_image(
      const cv::Mat& mask,
      const cv::Mat& image,
      const gui::Transform &midline_transform,
      float midline_length,
      const Size2 &output_size,
      bool use_legacy)
{
    cv::Mat padded;
    
    if(midline_length < 0) {
        static Timer timer;
        if(timer.elapsed() > 1) { // dont spam messages
            FormatWarning("[calculate_normalized_diff_image] invalid midline_length");
            timer.reset();
        }
        return {nullptr, Vec2()};
    }
    
    if(!output_size.empty())
        padded = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC(image.channels()));
    else
        image.copyTo(padded);
    assert(padded.isContinuous());
    
    auto size = Size2(padded.size());
    auto scale = FAST_SETTING(individual_image_scale);
    //Vec2 pos = size * 0.5 + Vec2(midline_length * 0.4);
    
    gui::Transform tr;
    if(use_legacy) {
        tr.translate(size * 0.5);
        tr.scale(scale);
        tr.translate(Vec2(-midline_length * 0.5, 0));
        
    } else {
        tr.translate(size * 0.5);
        tr.scale(scale);
        tr.translate(Vec2(midline_length * 0.4));
    }
    tr.combine(midline_transform);
    
    auto t = tr.toCV();
    
    image.copyTo(image, mask);
    //tf::imshow("before", image);
    
    //TODO: if larger?
    using namespace grab::default_config;
    if(Background::meta_encoding() == meta_encoding_t::r3g3b2)
       cv::warpAffine(image, padded, t, (cv::Size)size, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
    else
       cv::warpAffine(image, padded, t, (cv::Size)size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    //resize_image(padded, SETTING(individual_image_scale).value<float>());
    
    //tf::imshow("after", padded);
    int left = 0, right = 0, top = 0, bottom = 0;
    
    if(!output_size.empty()) {
        if(padded.cols < output_size.width) {
            left = roundf(output_size.width - padded.cols);
            right = left / 2;
            left -= right;
        }
        
        if(padded.rows < output_size.height) {
            top = roundf(output_size.height - padded.rows);
            bottom = top / 2;
            top -= bottom;
        }
        
        if(left || right || top || bottom)
            cv::copyMakeBorder(padded, padded, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
        
        assert(padded.cols >= output_size.width && padded.rows >= output_size.height);
        if(padded.cols > output_size.width || padded.rows > output_size.height) {
            left = padded.cols - output_size.width;
            right = left / 2;
            left -= right;
            
            top = padded.rows - output_size.height;
            bottom = top / 2;
            top -= bottom;
            
            padded(Bounds(left, top, padded.cols - left - right, padded.rows - top - bottom)).copyTo(padded);
        }
    }
    
    if(!output_size.empty() && (padded.cols != output_size.width || padded.rows != output_size.height))
        throw U_EXCEPTION("Padded size differs from expected size (",padded.cols,"x",padded.rows," != ",output_size.width,"x",output_size.height,")");
    
    auto i = tr.getInverse();
    auto pt = i.transformPoint(left, top);
    return { Image::Make(padded), pt };
}

std::tuple<Image::Ptr, Vec2>
calculate_normalized_image(const gui::Transform &midline_transform,
                           const pv::BlobWeakPtr& blob,
                           float midline_length,
                           const Size2 &output_size,
                           bool use_legacy,
                           const Background* background)
{
    cv::Mat mask, image;
    if(!blob->pixels())
        throw std::invalid_argument("[calculate_normalized_diff_image] The blob has to contain pixels.");
    
    imageFromLines(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels().get(), 0, background, 0);
    
    return normalize_image(mask, image, midline_transform, midline_length, output_size, use_legacy);
}

std::tuple<Image::Ptr, Vec2>
calculate_normalized_diff_image(const gui::Transform &midline_transform,
                                const pv::BlobWeakPtr& blob,
                                float midline_length,
                                const Size2 &output_size,
                                bool use_legacy,
                                const Background* background)
{
    cv::Mat mask, image;
    if(not blob->is_binary() && not blob->pixels())
        throw std::invalid_argument("[calculate_normalized_diff_image] The blob has to contain pixels.");
    
    if(   background
       && Background::track_background_subtraction())
    {
        imageFromLines(blob->input_info(), blob->hor_lines(), &mask, NULL, &image, blob->pixels() ? blob->pixels().get() : nullptr, 0, background, 0);
    } else {
        imageFromLines(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels() ? blob->pixels().get() : nullptr, 0, nullptr, 0);
    }
    
    return normalize_image(mask, image, midline_transform, midline_length, output_size, use_legacy);
}

std::tuple<Image::Ptr, Vec2>
calculate_diff_image(pv::BlobWeakPtr blob,
                     const Size2& output_size,
                     const Background* background)
{
    cv::Mat mask, image;
    cv::Mat padded;
    
    if(not blob->is_binary() && not blob->pixels())
        throw std::invalid_argument("[calculate_diff_image] The blob has to contain pixels.");
    
    if(background
       && Background::track_background_subtraction())
    {
        imageFromLines(blob->input_info(), blob->hor_lines(), &mask, NULL, &image, blob->pixels() ? blob->pixels().get() : nullptr, 0, background, 0);
    } else {
        imageFromLines(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels() ? blob->pixels().get() : nullptr, 0, nullptr, 0);
    }
    
    image.copyTo(padded, mask);
    
    auto scale = FAST_SETTING(individual_image_scale);
    if(scale != 1)
        resize_image(padded, scale);
    
    Bounds bounds(blob->bounds().pos(), blob->bounds().size() + blob->bounds().pos());
    
    if(!output_size.empty()) {
        int left = 0, right = 0, top = 0, bottom = 0;
        if(padded.cols < output_size.width) {
            left = roundf(output_size.width - padded.cols);
            right = left / 2;
            left -= right;
        }
        
        if(padded.rows < output_size.height) {
            top = roundf(output_size.height - padded.rows);
            bottom = top / 2;
            top -= bottom;
        }
        
        if(left || right || top || bottom) {
            bounds.x -= left;
            bounds.y -= top;
            bounds.width += right;
            bounds.height += bottom;
            
            cv::copyMakeBorder(padded, padded, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
        }
        
        bounds << Size2(bounds.size() - bounds.pos());
        
        assert(padded.cols >= output_size.width && padded.rows >= output_size.height);
        if(padded.cols > output_size.width || padded.rows > output_size.height) {
            left = padded.cols - output_size.width;
            right = left / 2;
            left -= right;
            
            top = padded.rows - output_size.height;
            bottom = top / 2;
            top -= bottom;
            
            Bounds cut(left, top, padded.cols - left - right, padded.rows - top - bottom);
            
            bounds.x += cut.x;
            bounds.y += cut.y;
            bounds.width = cut.width;
            bounds.height = cut.height;
            
            padded(cut).copyTo(padded);
        }
    }
    
    if(!output_size.empty() && (padded.cols != output_size.width || padded.rows != output_size.height))
        throw U_EXCEPTION("Padded size differs from expected size (",padded.cols,"x",padded.rows," != ",output_size.width,"x",output_size.height,")");
    
    
    return { Image::Make(padded), bounds.pos() };
}

}

namespace constraints {
using namespace image;

std::string FilterCache::toStr() const {
    return "TFC<l:" + Meta::toStr(median_midline_length_px) + "+-" + Meta::toStr(midline_length_px_std) + " pts:" + Meta::toStr(median_number_outline_pts) + "+-" + Meta::toStr(outline_pts_std) + " angle:" + Meta::toStr(median_angle_diff) + ">";
}

static auto& filter_mutex() {
    static auto _filter_mutex = new LOGGED_MUTEX("FilterCache::_filter_mutex");
    return *_filter_mutex;
}
inline static std::map<Idx_t, std::map<Range<Frame_t>, std::shared_ptr<FilterCache>>> _filter_cache_std, _filter_cache_no_std;

inline Float2_t standard_deviation(const std::set<Float2_t> & v) {
    Float2_t sum = std::accumulate(v.begin(), v.end(), 0.0_F);
    Float2_t mean = sum / v.size();
    
    std::vector<Float2_t> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](Float2_t x) {
        return x - mean;
    });
    Float2_t sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0_F);
    
    return (Float2_t)std::sqrt(sq_sum / v.size());
}

std::tuple<Image::Ptr, Vec2> diff_image(
     const individual_image_normalization_t::Class &normalize,
     pv::BlobWeakPtr blob,
     const gui::Transform& midline_transform,
     float median_midline_length_px,
     const Size2& output_shape,
     const Background* background)
{
    if(normalize == individual_image_normalization_t::posture)
        return calculate_normalized_diff_image(midline_transform, blob, median_midline_length_px, output_shape, false, background);
    else if(normalize == individual_image_normalization_t::legacy)
        return calculate_normalized_diff_image(midline_transform, blob, median_midline_length_px, output_shape, true, background);
    else if (normalize == individual_image_normalization_t::moments)
    {
        blob->calculate_moments();
        
        gui::Transform tr;
        float angle = narrow_cast<float>(DEGREE(-blob->orientation() + float(M_PI) * 0.25f));
        
        tr.rotate(angle);
        tr.translate( -blob->bounds().size() * 0.5);
        //tr.translate(-offset());
        
        return calculate_normalized_diff_image(tr, blob, 0, output_shape, false, background);
    }
    else {
        auto && [img, pos] = calculate_diff_image(blob, output_shape, background);
        return std::make_tuple(std::move(img), pos);
    }
}

void FilterCache::clear() {
    auto guard = LOGGED_LOCK(filter_mutex());
    _filter_cache_std.clear();
    _filter_cache_no_std.clear();
}

bool cached_filter(Idx_t fdx, const Range<Frame_t>& tracklet, FilterCache & constraints, const bool with_std) {
    auto guard = LOGGED_LOCK(filter_mutex());
    const auto &cache = with_std ? _filter_cache_std : _filter_cache_no_std;
    auto fit = cache.find(fdx);
    if(fit != cache.end()) {
        auto sit = fit->second.find(tracklet);
        if(sit != fit->second.end()) {
            constraints = *sit->second;
            return true;
        }
    }
    return false;
}

std::shared_ptr<FilterCache>
local_midline_length(const Individual *fish,
                     Frame_t frame,
                     const bool calculate_std)
{
    auto tracklet = fish->get_tracklet(frame);
    if(tracklet.contains(frame)) {
        return local_midline_length(fish, tracklet.range, calculate_std);
    }
    
    return nullptr;
}

std::shared_ptr<FilterCache> local_midline_length(const Individual *fish,
                                                  const Range<Frame_t>& tracklet,
                                                  const bool calculate_std)
{
    std::shared_ptr<FilterCache> constraints = std::make_shared<FilterCache>();
    if(cached_filter(fish->identity().ID(), tracklet, *constraints, calculate_std))
        return constraints;
    
    /// limit the number of samples that can be taken
    /// to stop its impact on overall performance for very
    /// long and many tracklets (it will average out anyway).
    /// we will add a safety margin here in case we have to skip
    /// some frames and underestimated it:
    static constexpr uint32_t max_samples = 200;
    const uint32_t step_size = tracklet.empty() ? 1 : max(1u, uint32_t(tracklet.length().get() * 0.9) / max_samples);
    
    Median<Float2_t> median_midline, median_outline, median_angle_diff;
    std::set<Float2_t> midline_lengths, outline_stds;
    
    const PostureStuff* previous_midline = nullptr;
    
    if (FAST_SETTING(calculate_posture)) {
        fish->iterate_frames(tracklet, [&](Frame_t frame, const auto&, auto basic, auto posture) -> bool
        {
            if (!basic || !posture || basic->blob.split())
                return true;

            auto bounds = basic->blob.calculate_bounds();
            if (!Tracker::instance()->border().in_recognition_bounds(bounds.pos() + bounds.size() * 0.5))
                return true;

            if (posture->cached()) {
                auto L = posture->midline_length.value();
                median_midline.addNumber(L);
                if (calculate_std)
                    midline_lengths.insert(L);

                if (previous_midline && previous_midline->frame == frame - 1_f) {
                    auto pangle = previous_midline->midline_angle.value();
                    auto cangle = posture->midline_angle.value();

                    auto first = Vec2(sin(pangle), cos(pangle));
                    auto second = Vec2(sin(cangle), cos(cangle));
                    auto diff = (first - second).length();
                    median_angle_diff.addNumber(diff);
                }

                previous_midline = posture;
            }

            if (posture->outline) {
                median_outline.addNumber(posture->outline.size());
                if (calculate_std)
                    outline_stds.insert(posture->outline.size());
            }

            return true;
        }, step_size);
    }

    if(not median_midline.empty())
        constraints->median_midline_length_px = median_midline.getValue();
    if(not median_outline.empty())
        constraints->median_number_outline_pts = median_outline.getValue();
    
    if(!midline_lengths.empty())
        constraints->midline_length_px_std = standard_deviation(midline_lengths);
    if(!outline_stds.empty())
        constraints->outline_pts_std = standard_deviation(outline_stds);
    
    constraints->median_angle_diff = not median_angle_diff.empty() ? median_angle_diff.getValue() : 0;
    
    if(!constraints->empty()) {
        auto guard = LOGGED_LOCK(filter_mutex());
        if(calculate_std)
            _filter_cache_std[fish->identity().ID()][tracklet] = constraints;
        else
            _filter_cache_no_std[fish->identity().ID()][tracklet] = constraints;
    }
    
    return constraints;
}
}
}
