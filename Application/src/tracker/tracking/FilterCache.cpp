#include "FilterCache.h"
#include <gui/Transform.h>
#include <misc/Image.h>
#include <misc/idx_t.h>
#include <misc/frame_t.h>
#include <tracker/misc/default_config.h>
#include <misc/PVBlob.h>
#include <misc/ranges.h>
#include <misc/Timer.h>
#include <misc/vec2.h>
#include <tracking/Tracker.h>

using namespace cmn;
using namespace default_config;

namespace track {
namespace image {
std::tuple<Image::UPtr, Vec2> normalize_image(
      const cv::Mat& mask,
      const cv::Mat& image,
      const gui::Transform &midline_transform,
      const pv::BlobPtr& blob,
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
        padded = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC1);
    else
        image.copyTo(padded);
    assert(padded.isContinuous());
    
    auto size = Size2(padded.size());
    auto scale = SETTING(recognition_image_scale).value<float>();
    //Vec2 pos = size * 0.5 + Vec2(midline_length * 0.4);
    
    gui::Transform tr;
    if(use_legacy) {
        tr.translate(size * 0.5);
        tr.scale(Vec2(scale));
        tr.translate(Vec2(-midline_length * 0.5, 0));
        
    } else {
        tr.translate(size * 0.5);
        tr.scale(Vec2(scale));
        tr.translate(Vec2(midline_length * 0.4));
    }
    tr.combine(midline_transform);
    
    auto t = tr.toCV();
    
    image.copyTo(image, mask);
    //tf::imshow("before", image);
    
    cv::warpAffine(image, padded, t, (cv::Size)size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    //resize_image(padded, SETTING(recognition_image_scale).value<float>());
    
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

std::tuple<Image::UPtr, Vec2>
calculate_normalized_image(const gui::Transform &midline_transform,
                           const pv::BlobPtr& blob,
                           float midline_length,
                           const Size2 &output_size,
                           bool use_legacy,
                           const Image* background)
{
    cv::Mat mask, image;
    if(!blob->pixels())
        throw std::invalid_argument("[calculate_normalized_diff_image] The blob has to contain pixels.");
    imageFromLines(blob->hor_lines(), &mask, &image, NULL, blob->pixels().get(), 0, background, 0);
    
    return normalize_image(mask, image, midline_transform, blob, midline_length, output_size, use_legacy);
}

std::tuple<Image::UPtr, Vec2>
calculate_normalized_diff_image(const gui::Transform &midline_transform,
                                const pv::BlobPtr& blob,
                                float midline_length,
                                const Size2 &output_size,
                                bool use_legacy,
                                const Image* background)
{
    cv::Mat mask, image;
    if(!blob->pixels())
        throw std::invalid_argument("[calculate_normalized_diff_image] The blob has to contain pixels.");
    imageFromLines(blob->hor_lines(), &mask, NULL, &image, blob->pixels().get(), 0, background, 0);
    
    return normalize_image(mask, image, midline_transform, blob, midline_length, output_size, use_legacy);
}

std::tuple<Image::UPtr, Vec2>
calculate_diff_image(pv::BlobPtr blob,
                     const Size2& output_size,
                     const Image* background)
{
    cv::Mat mask, image;
    cv::Mat padded;
    
    if(!blob->pixels())
        throw std::invalid_argument("[calculate_diff_image] The blob has to contain pixels.");
    imageFromLines(blob->hor_lines(), &mask, NULL, &image, blob->pixels().get(), 0, background, 0);
    image.copyTo(padded, mask);
    
    auto scale = SETTING(recognition_image_scale).value<float>();
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

inline static std::mutex _filter_mutex;
inline static std::map<Idx_t, std::map<Range<Frame_t>, std::shared_ptr<FilterCache>>> _filter_cache_std, _filter_cache_no_std;

inline float standard_deviation(const std::set<float> & v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    
    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    
    return (float)std::sqrt(sq_sum / v.size());
}

std::tuple<Image::UPtr, Vec2> diff_image(
     const recognition_normalization_t::Class &normalize,
     const pv::BlobPtr& blob,
     const gui::Transform& midline_transform,
     float median_midline_length_px,
     const Size2& output_shape,
     const Image* background)
{
    if(normalize == recognition_normalization_t::posture)
        return calculate_normalized_diff_image(midline_transform, blob, median_midline_length_px, output_shape, false, background);
    else if(normalize == recognition_normalization_t::legacy)
        return calculate_normalized_diff_image(midline_transform, blob, median_midline_length_px, output_shape, true, background);
    else if (normalize == recognition_normalization_t::moments)
    {
        blob->calculate_moments();
        
        gui::Transform tr;
        float angle = narrow_cast<float>(-blob->orientation() + M_PI * 0.25);
        
        tr.rotate(DEGREE(angle));
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
    std::lock_guard<std::mutex> guard(_filter_mutex);
    _filter_cache_std.clear();
    _filter_cache_no_std.clear();
}

bool cached_filter(Idx_t fdx, const Range<Frame_t>& segment, FilterCache & constraints, const bool with_std) {
    std::lock_guard<std::mutex> guard(_filter_mutex);
    const auto &cache = with_std ? _filter_cache_std : _filter_cache_no_std;
    auto fit = cache.find(fdx);
    if(fit != cache.end()) {
        auto sit = fit->second.find(segment);
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
    auto segment = fish->get_segment(frame);
    if(segment.contains(frame)) {
        return local_midline_length(fish, segment.range, calculate_std);
    }
    
    return nullptr;
}

std::shared_ptr<FilterCache> local_midline_length(const Individual *fish,
                                                  const Range<Frame_t>& segment,
                                                  const bool calculate_std)
{
    std::shared_ptr<FilterCache> constraints = std::make_shared<FilterCache>();
    if(cached_filter(fish->identity().ID(), segment, *constraints, calculate_std))
        return constraints;
    
    Median<float> median_midline, median_outline, median_angle_diff;
    std::set<float> midline_lengths, outline_stds;
    
    const PostureStuff* previous_midline = nullptr;
    
    fish->iterate_frames(segment, [&](Frame_t frame, const auto&, auto basic, auto posture) -> bool
    {
        if(!basic || !posture || basic->blob.split())
            return true;
        
        auto bounds = basic->blob.calculate_bounds();
        if(!Tracker::instance()->border().in_recognition_bounds(bounds.pos() + bounds.size() * 0.5))
            return true;
        
        if(posture->cached()) {
            median_midline.addNumber(posture->midline_length);
            if(calculate_std)
                midline_lengths.insert(posture->midline_length);
            
            if(previous_midline && previous_midline->frame == frame - 1_f) {
                auto first = Vec2(sin(previous_midline->midline_angle), cos(previous_midline->midline_angle));
                auto second = Vec2(sin(posture->midline_angle), cos(posture->midline_angle));
                auto diff = (first - second).length();
                median_angle_diff.addNumber(diff);
            }
            
            previous_midline = posture;
        }
        
        if(posture->outline) {
            median_outline.addNumber(posture->outline->size());
            if(calculate_std)
                outline_stds.insert(posture->outline->size());
        }
        
        return true;
    });
    
    if(median_midline.added())
        constraints->median_midline_length_px = median_midline.getValue();
    if(median_outline.added())
        constraints->median_number_outline_pts = median_outline.getValue();
    
    if(!midline_lengths.empty())
        constraints->midline_length_px_std = standard_deviation(midline_lengths);
    if(!outline_stds.empty())
        constraints->outline_pts_std = standard_deviation(outline_stds);
    
    constraints->median_angle_diff = median_angle_diff.added() ? median_angle_diff.getValue() : 0;
    
    if(!constraints->empty()) {
        std::lock_guard<std::mutex> guard(_filter_mutex);
        if(calculate_std)
            _filter_cache_std[fish->identity().ID()][segment] = constraints;
        else
            _filter_cache_no_std[fish->identity().ID()][segment] = constraints;
    }
    
    return constraints;
}
}
}
