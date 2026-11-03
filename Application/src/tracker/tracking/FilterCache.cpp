#include "FilterCache.h"
#include <tracking/Individual.h>
#include <tracking/Stuffs.h>
#include <tracking/TrackletInformation.h>
#include <gui/Transform.h>
#include <misc/Image.h>
#include <core/idx_t.h>
#include <misc/frame_t.h>
#include <core/default_config.h>
#include <processing/PVBlob.h>
#include <misc/ranges.h>
#include <misc/Timer.h>
#include <tracking/Stuffs.h>
#include <tracking/Tracker.h>
#include <processing/Background.h>
#include <misc/Median.h>


using namespace default_config;

namespace track {
namespace image {

std::optional<Vec2> normalize_image(
      Image& padded,
      const cv::Mat& mask,
      const cv::Mat& image,
      const gui::Transform &midline_transform,
      float midline_length,
      const Size2 &output_size,
      bool use_legacy)
{
    if(midline_length < 0) {
        static Timer timer;
        if(timer.elapsed() > 1) { // dont spam messages
            FormatWarning("[calculate_normalized_diff_image] invalid midline_length");
            timer.reset();
        }
        return std::nullopt;
    }
    
    if(!output_size.empty()) {
        padded.create(output_size.height, output_size.width, image.channels());
        //padded.set_to(0);
        //padded = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC(image.channels()));
    } else {
        padded.create(image.rows, image.cols, image.channels());
        //image.copyTo(padded);
    }
    //assert(padded.isContinuous());
    
    const auto size = padded.dimensions();
    const auto scale = FAST_SETTING(individual_image_scale);
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
    
    //! TODO: questionable?
    //tf::imshow("before masking", image);
    //image.copyTo(image, mask);
    //tf::imshow("mask", mask);
    //tf::imshow("after masking", image);
    //tf::imshow("before", image);
    
    //TODO: if larger?
    auto buffer = padded.get();

    assert(buffer.type() == image.type());
    assert(buffer.data != image.data); // warpAffine is not in-place

    const uchar* expected_data = buffer.data;

    /// same size as padded was before
    cv::warpAffine(image,
                   buffer,
                   tr.toCV(),
                   (cv::Size)size,
                   Background::meta_encoding() == meta_encoding_t::r3g3b2
                     ? cv::INTER_NEAREST
                     : cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT);
    assert(expected_data == buffer.data);

    //resize_image(padded, READ_SETTING(individual_image_scale, float));
    
    //tf::imshow("after", padded);
    /*int left = 0, right = 0, top = 0, bottom = 0;
    
    if(not output_size.empty()) {
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
        
        assert(padded.cols + left + right == output_size.width);
        assert(padded.rows + top + bottom == output_size.height);

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
    }*/
    
    if(!output_size.empty() && (padded.cols != output_size.width || padded.rows != output_size.height))
        throw U_EXCEPTION("Padded size differs from expected size (",padded.cols,"x",padded.rows," != ",output_size.width,"x",output_size.height,")");
    
    return tr.getInverse().transformPoint(0, 0);
}

template<ImageFromLinesMode Mode>
static std::optional<Vec2>
calculate_normalized_image(cv::Mat& mask,
                           cv::Mat& image,
                           Image& output,
                           const gui::Transform &midline_transform,
                           const pv::BlobWeakPtr& blob,
                           float midline_length,
                           const Size2 &output_size,
                           bool use_legacy,
                           const Background* background)
{
    if(blob->encoding() != meta_encoding_t::binary
       && not blob->pixels())
    {
        throw std::invalid_argument("[calculate_normalized_diff_image] The blob has to contain pixels.");
    }

    if constexpr(Mode == ImageFromLinesMode::Cached)
        imageFromLinesCached(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels().get(), 0, background, 0);
    else
        imageFromLines(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels().get(), 0, background, 0);

    return normalize_image(output, mask, image, midline_transform,
                           midline_length, output_size, use_legacy);
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
    auto ptr = Image::Make();
    auto pt = calculate_normalized_image<ImageFromLinesMode::Exact>(
        mask, image, *ptr, midline_transform, blob, midline_length,
        output_size, use_legacy, background);
    if(not pt)
        throw std::invalid_argument("[calculate_normalized_diff_image] Failed to normalize_image.");
    return {std::move(ptr), *pt};
}

std::optional<Vec2>
calculate_normalized_image_cached(cv::Mat& mask,
                                  cv::Mat& image,
                                  Image& output,
                                  const gui::Transform &midline_transform,
                                  const pv::BlobWeakPtr& blob,
                                  float midline_length,
                                  const Size2 &output_size,
                                  bool use_legacy,
                                  const Background* background)
{
    return calculate_normalized_image<ImageFromLinesMode::Cached>(
        mask, image, output, midline_transform, blob, midline_length,
        output_size, use_legacy, background);
}

template<ImageFromLinesMode Mode>
static std::optional<Vec2>
calculate_normalized_diff_image(cv::Mat& mask,
                                cv::Mat& image,
                                Image& output,
                                const gui::Transform &midline_transform,
                                const pv::BlobWeakPtr& blob,
                                float midline_length,
                                const Size2 &output_size,
                                bool use_legacy,
                                const Background* background)
{
    if(not blob->is_binary() && not blob->pixels())
        throw std::invalid_argument("[calculate_normalized_diff_image] The blob has to contain pixels.");

    if(   background
       && Background::track_background_subtraction())
    {
        if constexpr(Mode == ImageFromLinesMode::Cached)
            imageFromLinesCached(blob->input_info(), blob->hor_lines(), &mask, NULL, &image, blob->pixels() ? blob->pixels().get() : nullptr, 0, background, 0);
        else
            imageFromLines(blob->input_info(), blob->hor_lines(), &mask, NULL, &image, blob->pixels() ? blob->pixels().get() : nullptr, 0, background, 0);
    } else {
        if constexpr(Mode == ImageFromLinesMode::Cached)
            imageFromLinesCached(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels() ? blob->pixels().get() : nullptr, 0, nullptr, 0);
        else
            imageFromLines(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels() ? blob->pixels().get() : nullptr, 0, nullptr, 0);
    }

    return normalize_image(output, mask, image, midline_transform,
                           midline_length, output_size, use_legacy);
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
    auto ptr = Image::Make();
    auto pt = calculate_normalized_diff_image<ImageFromLinesMode::Exact>(
        mask, image, *ptr, midline_transform, blob, midline_length,
        output_size, use_legacy, background);
    if(not pt)
        return {nullptr, Vec2{}};
    return {std::move(ptr), *pt};
}

std::optional<Vec2>
calculate_normalized_diff_image_cached(cv::Mat& mask,
                                       cv::Mat& image,
                                       Image& output,
                                       const gui::Transform &midline_transform,
                                       const pv::BlobWeakPtr& blob,
                                       float midline_length,
                                       const Size2 &output_size,
                                       bool use_legacy,
                                       const Background* background)
{
    return calculate_normalized_diff_image<ImageFromLinesMode::Cached>(
        mask, image, output, midline_transform, blob, midline_length,
        output_size, use_legacy, background);
}

template<ImageFromLinesMode Mode>
static std::optional<Vec2>
calculate_diff_image(cv::Mat& mask,
                     cv::Mat& image,
                     Image& output,
                     pv::BlobWeakPtr blob,
                     const Size2& output_size,
                     const Background* background)
{
    if(not blob->is_binary() && not blob->pixels())
        throw std::invalid_argument("[calculate_diff_image] The blob has to contain pixels.");
    
    if(background
       && Background::track_background_subtraction())
    {
        if constexpr(Mode == ImageFromLinesMode::Cached)
            imageFromLinesCached(blob->input_info(), blob->hor_lines(), &mask, NULL, &image, blob->pixels() ? blob->pixels().get() : nullptr, 0, background, 0);
        else
            imageFromLines(blob->input_info(), blob->hor_lines(), &mask, NULL, &image, blob->pixels() ? blob->pixels().get() : nullptr, 0, background, 0);
    } else {
        if constexpr(Mode == ImageFromLinesMode::Cached)
            imageFromLinesCached(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels() ? blob->pixels().get() : nullptr, 0, nullptr, 0);
        else
            imageFromLines(blob->input_info(), blob->hor_lines(), &mask, &image, NULL, blob->pixels() ? blob->pixels().get() : nullptr, 0, nullptr, 0);
    }

    const double scale = FAST_SETTING(individual_image_scale);
    if(scale <= 0)
        throw InvalidArgumentException("individual_image_scale must be greater than zero.");

    const int scaled_width = scale == 1 ? image.cols : cvRound(image.cols * scale);
    const int scaled_height = scale == 1 ? image.rows : cvRound(image.rows * scale);
    if(scaled_width <= 0 || scaled_height <= 0)
        throw InvalidArgumentException("Scaled image dimensions must be greater than zero.");

    const int output_width = output_size.empty() ? scaled_width : static_cast<int>(output_size.width);
    const int output_height = output_size.empty() ? scaled_height : static_cast<int>(output_size.height);

    int left = 0, right = 0, top = 0, bottom = 0;
    int source_x = 0, source_y = 0;
    int destination_x = 0, destination_y = 0;
    int padded_width = scaled_width, padded_height = scaled_height;
    Bounds bounds(blob->bounds().pos(), blob->bounds().size() + blob->bounds().pos());

    if(not output_size.empty()) {
        if(padded_width < output_width) {
            left = output_width - padded_width;
            right = left / 2;
            left -= right;
            destination_x = left;
            padded_width += left + right;

            bounds.x -= left;
            bounds.width += right;
        }

        if(padded_height < output_height) {
            top = output_height - padded_height;
            bottom = top / 2;
            top -= bottom;
            destination_y = top;
            padded_height += top + bottom;

            bounds.y -= top;
            bounds.height += bottom;
        }

        bounds << Size2(bounds.size() - bounds.pos());

        assert(padded_width >= output_width && padded_height >= output_height);
        if(padded_width > output_width || padded_height > output_height) {
            left = padded_width - output_width;
            right = left / 2;
            left -= right;
            source_x = left;

            top = padded_height - output_height;
            bottom = top / 2;
            top -= bottom;
            source_y = top;

            bounds.x += left;
            bounds.y += top;
            bounds.width = output_width;
            bounds.height = output_height;
        }
    }

    const bool output_matches = not output.empty()
                                && output.cols == sign_cast<uint>(output_width)
                                && output.rows == sign_cast<uint>(output_height)
                                && output.channels() == sign_cast<uint>(image.channels());
    const auto expected_data = output.data();
    output.create(output_height, output_width, image.channels());
    assert(not output_matches || output.data() == expected_data);

    auto padded = output.get();
    assert(padded.type() == image.type());
    assert(padded.data != image.data);

    const int copy_width = min(scaled_width, output_width);
    const int copy_height = min(scaled_height, output_height);
    if(destination_x != 0 || destination_y != 0
       || copy_width != output_width || copy_height != output_height)
    {
        padded.setTo(cv::Scalar::all(0));
    }

    if(scale == 1) {
        auto source = image(cv::Rect(source_x, source_y, copy_width, copy_height));
        auto destination = padded(cv::Rect(destination_x, destination_y, copy_width, copy_height));
        source.copyTo(destination);

    } else {
        const int channels = image.channels();
        const double inverse_scale = 1.0 / scale;
        for(int y = 0; y < copy_height; ++y) {
            const int input_y = min(cvFloor((source_y + y) * inverse_scale), image.rows - 1);
            const auto input = image.ptr<uchar>(input_y);
            auto destination = padded.ptr<uchar>(destination_y + y);

            for(int x = 0; x < copy_width; ++x) {
                const int input_x = min(cvFloor((source_x + x) * inverse_scale), image.cols - 1);
                std::memcpy(destination + (destination_x + x) * channels,
                            input + input_x * channels,
                            channels);
            }
        }
    }

    return bounds.pos();
}

std::tuple<Image::Ptr, Vec2>
calculate_diff_image(pv::BlobWeakPtr blob,
                     const Size2& output_size,
                     const Background* background)
{
    cv::Mat mask, image;
    auto output = Image::Make();
    auto position = calculate_diff_image<ImageFromLinesMode::Exact>(
        mask, image, *output, blob, output_size, background);
    if(not position)
        return {nullptr, Vec2{}};
    return {std::move(output), *position};
}

std::optional<Vec2>
calculate_diff_image_cached(cv::Mat& mask,
                            cv::Mat& image,
                            Image& output,
                            pv::BlobWeakPtr blob,
                            const Size2& output_size,
                            const Background* background)
{
    return calculate_diff_image<ImageFromLinesMode::Cached>(
        mask, image, output, blob, output_size, background);
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

template<ImageFromLinesMode Mode>
static std::optional<Vec2> diff_image(
     cv::Mat& mask,
     cv::Mat& image,
     Image& output,
     const individual_image_normalization_t::Class &normalize,
     pv::BlobWeakPtr blob,
     const gui::Transform& midline_transform,
     float median_midline_length_px,
     const Size2& output_shape,
     const Background* background)
{
    if(normalize == individual_image_normalization_t::posture)
        return image::calculate_normalized_diff_image<Mode>(mask, image, output,
               midline_transform,
               blob,
               median_midline_length_px,
               output_shape,
               false,
               background);
    else if(normalize == individual_image_normalization_t::legacy)
        return image::calculate_normalized_diff_image<Mode>(mask, image, output,
               midline_transform,
               blob,
               median_midline_length_px,
               output_shape,
               true,
               background);
    else if (normalize == individual_image_normalization_t::moments)
    {
        blob->calculate_moments();
        
        gui::Transform tr;
        float angle = narrow_cast<float>(DEGREE(-blob->orientation() + float(M_PI) * 0.25f));
        
        tr.rotate(angle);
        tr.translate( -blob->bounds().size() * 0.5);
        //tr.translate(-offset());
        
        return image::calculate_normalized_diff_image<Mode>(mask, image, output,
                   tr,
                   blob,
                   0,
                   output_shape,
                   false,
                   background);
    }
    else {
        return image::calculate_diff_image<Mode>(
            mask, image, output, blob, output_shape, background);
    }
}

std::tuple<Image::Ptr, Vec2> diff_image(
     const individual_image_normalization_t::Class &normalize,
     pv::BlobWeakPtr blob,
     const gui::Transform& midline_transform,
     float median_midline_length_px,
     const Size2& output_shape,
     const Background* background)
{
    cv::Mat mask, image;
    auto output = Image::Make();
    auto position = diff_image<ImageFromLinesMode::Exact>(
        mask, image, *output, normalize, blob, midline_transform,
        median_midline_length_px, output_shape, background);
    if(not position)
        return {nullptr, Vec2{}};
    return {std::move(output), *position};
}

std::optional<Vec2> diff_image_cached(
     cv::Mat& mask,
     cv::Mat& image,
     Image& output,
     const individual_image_normalization_t::Class &normalize,
     pv::BlobWeakPtr blob,
     const gui::Transform& midline_transform,
     float median_midline_length_px,
     const Size2& output_shape,
     const Background* background)
{
    return diff_image<ImageFromLinesMode::Cached>(
        mask, image, output, normalize, blob, midline_transform,
        median_midline_length_px, output_shape, background);
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
                     const track::Border* border,
                     const bool calculate_std)
{
    auto tracklet = fish->get_tracklet(frame);
    if(tracklet.contains(frame)) {
        return local_midline_length(fish, tracklet.range, border, calculate_std);
    }
    
    return nullptr;
}

std::shared_ptr<FilterCache> local_midline_length(const Individual *fish,
                                                  const Range<Frame_t>& tracklet,
                                                  const Border* border,
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
            if (border
                && not border->in_recognition_bounds(bounds.pos() + bounds.size() * 0.5))
            {
                return true;
            }

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
