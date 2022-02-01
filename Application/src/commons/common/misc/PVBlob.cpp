#include <misc/PVBlob.h>
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
namespace pv {
    using namespace cmn;

    bid CompressedBlob::blob_id() const {
        if(!own_id.valid())
            own_id = pv::bid::from_blob(*this);
        
        return own_id;
    }

bool Blob::operator!=(const pv::Blob& other) const {
    return blob_id() != other.blob_id();
}

bool Blob::operator==(const pv::Blob& other) const {
    return blob_id() == other.blob_id();
}

    cmn::Bounds CompressedBlob::calculate_bounds() const {
        Float2_t max_x = 0, height = 0, min_x = lines.empty() ? 0 : infinity<Float2_t>();
        Float2_t x0, x1;
        for(auto &line : lines) {
            x0 = line.x0();
            x1 = line.x1();
            if(x1 > max_x) max_x = x1;
            if(x0 < min_x) min_x = x0;
            if(line.eol()) ++height;
        }
        
/*#ifndef NDEBUG
        auto bds = cmn::Bounds(min_x, start_y, max_x - min_x + 1, height + 1);
        auto ptr = unpack();
        if(ptr->bounds() != bds) {
            auto A = Meta::toStr(ptr->bounds());
            auto B = Meta::toStr(bds);
            Except("%S != %S", &A, &B);
        }
        return bds;
#else*/
        return cmn::Bounds(min_x, start_y, max_x - min_x + 1, height + 1);
//#endif
    }

    pv::BlobPtr CompressedBlob::unpack() const {
        auto flines = ShortHorizontalLine::uncompress(start_y, lines);
        auto ptr = std::make_shared<pv::Blob>(flines, nullptr);
        ptr->set_parent_id((status_byte & 0x2) != 0 ? parent_id : -1);
        
        bool tried_to_split = (status_byte & 0x4) != 0;
        ptr->set_tried_to_split(tried_to_split);
        
        if((status_byte & 0x1) != 0 && (status_byte & 0x2) == 0) {
            ptr->set_split(true);
        } else
            ptr->set_split(false);
        
        return ptr;
    }

    void Blob::set_split(bool split) {
        _split = split;
    }
    
    std::vector<ShortHorizontalLine>
        ShortHorizontalLine::compress(const std::vector<HorizontalLine>& lines)
    {
        std::vector<pv::ShortHorizontalLine> ret((NoInitializeAllocator<pv::ShortHorizontalLine>()));
        ret.resize(lines.size());
        
        auto start = lines.data(), end = lines.data() + lines.size();
        auto rptr = ret.data();
        auto prev_y = ret.empty() ? 0 : lines.front().y;
        
        for(auto lptr = start; lptr != end; lptr++, rptr++) {
            *rptr = pv::ShortHorizontalLine(lptr->x0, lptr->x1);
            
            if(prev_y != lptr->y)
                (rptr-1)->eol(true);
            prev_y = lptr->y;
        }
        
        return ret;
    }
    
    Blob::line_ptr_t ShortHorizontalLine::uncompress(
        uint16_t start_y,
        const std::vector<ShortHorizontalLine>& compressed)
    {
        auto uncompressed = std::make_shared<std::vector<HorizontalLine>>((NoInitializeAllocator<HorizontalLine>()));
        uncompressed->resize(compressed.size());
        
        auto y = start_y;
        auto uptr = uncompressed->data();
        auto cptr = compressed.data(), end = compressed.data()+compressed.size();
        
        for(; cptr != end; cptr++, uptr++) {
            uptr->y = y;
            uptr->x0 = cptr->x0();
            uptr->x1 = cptr->x1();
            
            if(cptr->eol())
                y++;
        }
        
        return uncompressed;
    }
    
    std::atomic<float> cm_per_pixel = 1;
    bool correct_illegal_lines = false;
    std::mutex cm_per_pixel_mutex;
    std::atomic_bool callback_registered = false;
    
    Blob::Blob() : Blob(std::make_shared<std::vector<HorizontalLine>>(), nullptr) {
        
    }
    
    Blob::Blob(const Blob& other)
        : Blob(other.lines(), other.pixels())
    //Blob(other.hor_lines(), other.pixels() ? std::make_shared<decltype(_pixels)::element_type>(*other.pixels()) : nullptr)
    {
        _tried_to_split = other._tried_to_split;
    }
    
    Blob::Blob(line_ptr_t lines, decltype(_pixels) pixels)
        : cmn::Blob(lines), _pixels(pixels)
    {
        init();
    }
    
    Blob::Blob(std::shared_ptr<const std::vector<HorizontalLine>> lines, decltype(_pixels) pixels)
        : Blob(std::make_shared<std::vector<HorizontalLine>>(*lines), pixels)
    { }
    
    Blob::Blob(const cmn::Blob* blob, decltype(_pixels) pixels)
        : cmn::Blob(*blob), _pixels(pixels)
    {
        init();
    }
    
    /*Blob::Blob(const std::vector<HorizontalLine>& lines,
               decltype(_pixels) pixels)
        : cmn::Blob(lines), _pixels(pixels)
    {
        init();
    }*/

struct Callback {
    const char *ptr;
    
    Callback() {
        ptr = "PVBlob::Callback";
        
        bool expected = false;
        if(callback_registered.compare_exchange_strong(expected, true))
        {
            sprite::Map::callback_func fn = [this](sprite::Map::Signal signal, sprite::Map&map, auto&name, auto&)
            {
                if(signal == sprite::Map::Signal::EXIT) {
                    map.unregister_callback(ptr);
                    ptr = nullptr;
                    return;
                }
                
                if(name != "cm_per_pixel" && name != "correct_illegal_lines")
                    return;
                cm_per_pixel = SETTING(cm_per_pixel).value<float>();
                correct_illegal_lines = SETTING(correct_illegal_lines).value<bool>();
            };
            GlobalSettings::map().register_callback(ptr, fn);
            cm_per_pixel = SETTING(cm_per_pixel).value<float>();
            correct_illegal_lines = SETTING(correct_illegal_lines).value<bool>();
        }
    }
    
    ~Callback() {
        if(ptr)
            GlobalSettings::map().unregister_callback(ptr);
    }
};

static Callback callback;
    
    void Blob::init() {
        _tried_to_split = false;
        
        _split = false;
        _recount = _recount_threshold = -1;
        _parent_id = bid::invalid;
        
//#ifndef NDEBUG
        static std::atomic_int counter(0);
        static std::atomic_bool displayed_warning_once(false);
        if(correct_illegal_lines || (!displayed_warning_once && counter < 1000)) {
            ++counter;
            
            HorizontalLine prev = hor_lines().empty() ? HorizontalLine() : hor_lines().front();
            
            bool incorrect = false;
            for (auto &line : hor_lines()) {
                if(!(prev == line) && !(prev < line)) {
                    if(!displayed_warning_once) {
                        Warning("HorizontalLines are not properly ordered, or overlapping in x [%d-%d] < [%d-%d] (%d/%d). Please set 'correct_illegal_lines' = true in your settings if you havent already.", prev.x0, prev.x1, line.x0, line.x1, prev.y, line.y);
                        displayed_warning_once = true;
                    }
                    incorrect = true;
                    break;
                }
                prev = line;
            }
            
            if(incorrect) {
                if(_pixels) {
                    std::vector<uchar> pixels(_pixels->begin(), _pixels->end());
                    std::vector<HorizontalLine> lines(_hor_lines->begin(), _hor_lines->end());
                    HorizontalLine::repair_lines_array(lines, pixels);
                    _hor_lines = std::make_shared<decltype(_hor_lines)::element_type>(lines);
                    _pixels = std::make_shared<decltype(_pixels)::element_type>(pixels);
                } else {
                    std::vector<HorizontalLine> lines(_hor_lines->begin(), _hor_lines->end());
                    HorizontalLine::repair_lines_array(lines);
                    _hor_lines = std::make_shared<decltype(_hor_lines)::element_type>(lines);
                }
            }
        }
//#endif
        
        calculate_properties();
        _blob_id = bid::from_blob(*this);
    }
    
    void Blob::set_split(bool split, pv::BlobPtr parent) {
        _split = split;
        if(parent)
            _parent_id = parent->parent_id().valid() ? parent->parent_id() : parent->blob_id();
        else
            _parent_id = bid::invalid;
        
        if(!_parent_id.valid() && split)
            Warning("Parent has to be set correctly in order to split blobs (%d).", blob_id());
    }
    
    void Blob::set_parent_id(const bid& parent_id) {
        _split = parent_id.valid();
        _parent_id = parent_id;
    }
    
    void Blob::force_set_recount(int32_t threshold, float value) {
        if(threshold && _recount_threshold == threshold) {
            //Warning("Not forcing recount of %d because it has already been calculated.", blob_id());
            return;
        }
        
        _recount = (value != -1 ? value : num_pixels());
        _recount_threshold = threshold;
    }
    
    float Blob::recount(int32_t threshold, const Background& background) {
        //const float cm_per_pixel = SETTING(cm_per_pixel).value<float>();
        if(threshold == 0) {
            _recount = num_pixels();
            _recount_threshold = 0;
            return _recount * SQR(cm_per_pixel);
        }
        
        if(threshold == -1 && _recount_threshold == -1)
            U_EXCEPTION("Did not calculate recount yet.");
        
        if(_recount_threshold != threshold) {
            //if(_recount_threshold != -1)
            //    Debug("Recalculating threshold...");
            
            if(_pixels == nullptr)
                U_EXCEPTION("Cannot threshold without pixel values.");
            
            _recount = 0;
#ifndef NDEBUG
            size_t local_recount = 0;
            auto local_ptr = _pixels->data();
#endif
            auto ptr = _pixels->data();
            for (auto &line : hor_lines()) {
                _recount += background.count_above_threshold(line.x0, line.x1, line.y, ptr, threshold);
                ptr += ptr_safe_t(line.x1) - ptr_safe_t(line.x0) + 1;
#ifndef NDEBUG
                for (auto x=line.x0; x<=line.x1; ++x, ++local_ptr) {
                    if(background.is_different(x, line.y, *local_ptr, threshold)) {
                        local_recount++;
                    }
                }
#endif
            }
            
            assert(_recount == local_recount);
            _recount_threshold = threshold;
        }
        
        return _recount * SQR(cm_per_pixel);
    }
    
    float Blob::recount(int32_t threshold) const {
        //if(threshold == 0)
        //    return num_pixels() * SQR(cm_per_pixel);
        if(threshold != -1 && _recount_threshold != threshold)
            U_EXCEPTION("Have to threshold() first.");
        if(threshold == -1 && _recount_threshold == -1)
            U_EXCEPTION("Did not calculate recount yet.");
        
        return _recount * SQR(cm_per_pixel);
    }
    
    BlobPtr Blob::threshold(int32_t value, const Background& background) {
        if(_pixels == nullptr)
            U_EXCEPTION("Cannot threshold without pixel values.");
        
        auto lines = std::make_shared<std::vector<HorizontalLine>>();
        lines->reserve(hor_lines().size());
        
        auto ptr = _pixels->data();
        HorizontalLine tmp;
        auto tmp_pixels = std::make_shared<std::vector<uchar>>();
        tmp_pixels->reserve(_pixels->size());
        
        for (auto &line : hor_lines()) {
            tmp.x0 = line.x0;
            tmp.y = line.y;
            
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr) {
                assert(ptr < _pixels->data() + _pixels->size());
                if(background.is_different(x, line.y, *ptr, value)) {
                    tmp.x1 = x;
                    tmp_pixels->push_back(*ptr);
                    _recount ++;
                    
                } else {
                    if(x > tmp.x0) {
                        lines->push_back(tmp);
                    }
                    tmp.x0 = x + 1;
                }
            }
            
            if(tmp.x1 == line.x1) {
                lines->push_back(tmp);
            }
        }
        
        return std::make_shared<Blob>(lines, tmp_pixels);
    }
    
    std::tuple<Vec2, Image::UPtr> Blob::image(const cmn::Background* background, const Bounds& restricted) const {
        Bounds b(bounds().pos()-Vec2(1), bounds().size()+Vec2(2));
        if(background)
            b.restrict_to(background->bounds());
        else if(restricted.width > 0)
            b.restrict_to(restricted);
        else
            b.restrict_to(Bounds(0, 0, infinity<Float2_t>(), infinity<Float2_t>()));
        
        auto image = Image::Make(b.height, b.width);
        
        if(!background)
            std::fill(image->data(), image->data() + image->size(), uchar(0));
        else
            background->image().get()(b).copyTo(image->get());
        
        auto _x = (coord_t)b.x;
        auto _y = (coord_t)b.y;
        
        auto ptr = _pixels->data();
        for (auto &line : hor_lines()) {
            auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols + ptr_safe_t(line.x0) - ptr_safe_t(_x));
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr, ++image_ptr) {
                assert(ptr < _pixels->data() + _pixels->size());
                assert(image_ptr < image->data() + image->size());
                *image_ptr = *ptr;
            }
        }
        return {b.pos(), std::move(image)};
    }
    
    std::tuple<Vec2, Image::UPtr> Blob::alpha_image(const cmn::Background& background, int32_t threshold) const {
        Bounds b(bounds().pos()-Vec2(1), bounds().size()+Vec2(2));
        b.restrict_to(background.bounds());
        
        auto image = Image::Make(b.height, b.width, 4);
        std::fill(image->data(), image->data() + image->size(), uchar(0));
        
        auto _x = (coord_t)b.x;
        auto _y = (coord_t)b.y;
        
        int32_t value;
        float maximum = 0;
        auto ptr = _pixels->data();
        for (auto &line : hor_lines()) {
            //auto image_ptr = image->data() + ((line.y - _y) * image->cols * image->dims + (line.x0 - _x) * image->dims);
            auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols + (ptr_safe_t(line.x0) - ptr_safe_t(_x))) * image->dims;
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr, image_ptr += image->dims) {
                assert(ptr < _pixels->data() + _pixels->size());
                value = background.diff(x, line.y, *ptr);
                if(background.is_value_different(x, line.y, value, threshold)) {
                    if(maximum < value)
                        maximum = value;
                    *image_ptr = *(image_ptr+1) = *(image_ptr+2) = *ptr;
                    *(image_ptr+3) = value;
                }
            }
        }
        
        if(maximum > 0) {
            for (auto &line : hor_lines()) {
                auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols + (ptr_safe_t(line.x0) - ptr_safe_t(_x))) * image->dims;
                for (auto x=line.x0; x<=line.x1; ++x, ++ptr, image_ptr += image->dims) {
                    *(image_ptr + 3) = min(255, float(*(image_ptr + 3)) / (maximum * 0.6) * 255);
                }
            }
        }
        
        return {b.pos(), std::move(image)};
    }

    std::tuple<Vec2, Image::UPtr> Blob::equalized_luminance_alpha_image(const cmn::Background& background, int32_t threshold, float minimum, float maximum) const {
        Bounds b(bounds().pos()-Vec2(1), bounds().size()+Vec2(2));
        b.restrict_to(background.bounds());
        
        auto image = Image::Make(b.height, b.width, 2);
        std::fill(image->data(), image->data() + image->size(), uchar(0));
        
        auto _x = (coord_t)b.x;
        auto _y = (coord_t)b.y;
        
        minimum *= 0.5;
        
        /*if constexpr(false) {
            static Timing timing("equalize_histogram", 0.01);
            TakeTiming take(timing);
            
            for(auto ptr = _pixels->data(); ptr != _pixels->data() + _pixels->size(); ++ptr) {
                if(*ptr < minimum) minimum = *ptr;
                if(*ptr > maximum) maximum = *ptr;
                
                if(minimum == 0 && maximum == 255)
                    break;
            }
        }*/
        
        float factor = 1;
        if(maximum > 0 && maximum != minimum)
            factor = 1.f / ((maximum - minimum) * 0.5) * 255;
        else
            minimum = 0;
        
        int32_t value;
        auto ptr = _pixels->data();
        for (auto &line : hor_lines()) {
            auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols * 2 + (ptr_safe_t(line.x0) - ptr_safe_t(_x)) * 2);
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr, image_ptr+=2) {
                assert(ptr < _pixels->data() + _pixels->size());
                value = background.diff(x, line.y, *ptr);
                if(!threshold || background.is_value_different(x, line.y, value, threshold)) {
                    *image_ptr = saturate((float(*ptr) - minimum) * factor);
                    //*image_ptr = *ptr;
                    *(image_ptr+1) = saturate(int32_t(255 - SQR(1 - value / 255.0) * 255.0));
                }
            }
        }
        return {b.pos(), std::move(image)};
    }

    std::tuple<Vec2, Image::UPtr> Blob::luminance_alpha_image(const cmn::Background& background, int32_t threshold) const {
        Bounds b(bounds().pos()-Vec2(1), bounds().size()+Vec2(2));
        b.restrict_to(background.bounds());
        
        auto image = Image::Make(b.height, b.width, 2);
        std::fill(image->data(), image->data() + image->size(), uchar(0));
        
        auto _x = (coord_t)b.x;
        auto _y = (coord_t)b.y;
        
        int32_t value;
        auto ptr = _pixels->data();
        for (auto &line : hor_lines()) {
            auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols * 2 + (ptr_safe_t(line.x0) - ptr_safe_t(_x)) * 2);
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr, image_ptr+=2) {
                assert(ptr < _pixels->data() + _pixels->size());
                value = background.diff(x, line.y, *ptr);
                if(!threshold || background.is_value_different(x, line.y, value, threshold)) {
                    *image_ptr = *ptr;
                    *(image_ptr+1) = saturate(int32_t(255 - SQR(1 - value / 255.0) * 255.0) * 2);
                }
            }
        }
        return {b.pos(), std::move(image)};
    }
    
    std::tuple<Vec2, Image::UPtr> Blob::difference_image(const cmn::Background& background, int32_t threshold) const {
        Bounds b(bounds().pos()-Vec2(1), bounds().size()+Vec2(2));
        b.restrict_to(background.bounds());
        
        auto image = Image::Make(b.height, b.width);
        std::fill(image->data(), image->data() + image->size(), uchar(0));
        
        auto _x = (coord_t)b.x;
        auto _y = (coord_t)b.y;
        
        int32_t value;
        auto ptr = _pixels->data();
        for (auto &line : hor_lines()) {
            auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols + ptr_safe_t(line.x0) - ptr_safe_t(_x));
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr, ++image_ptr) {
                assert(ptr < _pixels->data() + _pixels->size());
                value = background.diff(x, line.y, *ptr);
                if(!threshold || background.is_value_different(x, line.y, value, threshold))
                    *image_ptr = value;
            }
        }
        return {b.pos(), std::move(image)};
    }
    
    void Blob::transfer_backgrounds(const cmn::Background &from, const cmn::Background &to, const Vec2& dest_offset) {
        auto ptr = (uchar*)_pixels->data();
        for (auto &line : hor_lines()) {
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr) {
                assert(ptr < _pixels->data() + _pixels->size());
                *ptr = saturate(-int32_t(from.color(x, line.y)) + int32_t(*ptr) + to.color(ptr_safe_t(x) + dest_offset.x, ptr_safe_t(line.y) + dest_offset.y), 0, 255);
                //*ptr = saturate((int32_t)from.diff(x, line.y, *ptr) - to.color(x, line.y), 0, 255);
            }
        }
    }
    
    decltype(Blob::_pixels) Blob::calculate_pixels(Image::Ptr image, const decltype(_hor_lines) &lines) {
        auto pixels = std::make_shared<std::vector<uchar>>();
        for(auto &line : *lines) {
            auto start = image->data() + ptr_safe_t(line.y) * image->cols + ptr_safe_t(line.x0);
            auto end = start + ptr_safe_t(line.x1) - ptr_safe_t(line.x0) + 1;
            assert(line.x1 < image->cols && line.y < image->rows);
            
            pixels->insert(pixels->end(), start, end);
        }
        return pixels;
    }
    
    std::tuple<Vec2, Image::UPtr> Blob::thresholded_image(const cmn::Background& background, int32_t threshold) const {
        Bounds b(bounds().pos()-Vec2(1), bounds().size()+Vec2(2));
        b.restrict_to(background.bounds());
        
        auto image = Image::Make(b.height, b.width);
        std::fill(image->data(), image->data() + image->size(), uchar(0));
        
        auto _x = (coord_t)b.x;
        auto _y = (coord_t)b.y;
        
        auto ptr = _pixels->data();
        for (auto &line : hor_lines()) {
            auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols + ptr_safe_t(line.x0) - ptr_safe_t(_x));
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr, ++image_ptr) {
                assert(ptr < _pixels->data() + _pixels->size());
                if(background.is_value_different(x, line.y, background.diff(x, line.y, *ptr), threshold))
                    *image_ptr = *ptr;
            }
        }
        return {b.pos(), std::move(image)};
    }
    
    std::tuple<Vec2, Image::UPtr> Blob::binary_image(const cmn::Background& background, int32_t threshold) const {
        Bounds b(bounds().pos()-Vec2(1), bounds().size()+Vec2(2));
        b.restrict_to(background.bounds());
        
        auto image = Image::Make(b.height, b.width);
        std::fill(image->data(), image->data() + image->size(), uchar(0));
        
        auto _x = (coord_t)b.x;
        auto _y = (coord_t)b.y;
        
        if(_pixels == nullptr)
            U_EXCEPTION("Cannot generate binary image without pixel values.");
        
        int32_t value;
        auto ptr = _pixels->data();
        for (auto &line : hor_lines()) {
            auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols + ptr_safe_t(line.x0) - ptr_safe_t(_x));
            for (auto x=line.x0; x<=line.x1; ++x, ++ptr, ++image_ptr) {
                assert(ptr < _pixels->data() + _pixels->size());
                value = background.diff(x, line.y, *ptr);
                if(background.is_value_different(x, line.y, value, threshold))
                    *image_ptr = 255;
            }
        }
        return {b.pos(), std::move(image)};
    }
    
    std::tuple<Vec2, Image::UPtr> Blob::binary_image() const {
        Bounds b(bounds().pos()-Vec2(1), bounds().size()+Vec2(2));
        if(b.x < 0) {b.x = 0;--b.width;}
        if(b.y < 0) {b.y = 0;--b.height;}
        
        auto image = Image::Make(b.height, b.width);
        std::fill(image->data(), image->data() + image->size(), uchar(0));
        
        auto _x = (coord_t)b.x;
        auto _y = (coord_t)b.y;
        
        for (auto &line : hor_lines()) {
            auto image_ptr = image->data() + ((ptr_safe_t(line.y) - ptr_safe_t(_y)) * image->cols + ptr_safe_t(line.x0) - ptr_safe_t(_x));
            for (auto x=line.x0; x<=line.x1; ++x, ++image_ptr) {
                *image_ptr = 255;
            }
        }
        return {b.pos(), std::move(image)};
    }
    
    void Blob::set_pixels(decltype(_pixels) pixels) {
        _pixels = pixels;
        assert(!_pixels || _pixels->size() == num_pixels());
    }
    
/*void Blob::set_pixels(const cmn::grid::PixelGrid &grid, const cmn::Vec2& offset) {
    U_EXCEPTION("Deprecation.");
        auto pixels = std::make_shared<std::vector<uchar>>();
        for (auto &line : hor_lines()) {
            auto current = pixels->size();
            pixels->resize(pixels->size() + line.x1 - line.x0 + 1);
            grid.copy_row(line.x0 - offset.x, line.x1 - offset.x, line.y - offset.y, pixels->data() + current);
            //for (ushort x=line.x0; x<=line.x1; ++x) {
                //assert(ptr < _pixels->data() + _pixels->size());
              //  pixels->push_back(grid.query(x, line.y));
            //}
        }
        _pixels = pixels;
    }*/

    std::string Blob::name() const {
        auto center = bounds().pos() + bounds().size() * 0.5;
        auto id = blob_id();
        //auto x = id >> 16;
        //auto y = id & 0x0000FFFF;
        return Meta::toStr(id)+" "+Meta::toStr(center);
    }
    
    void Blob::add_offset(const cmn::Vec2 &off) {
        if(off == Vec2(0))
            return;
        
        cmn::Blob::add_offset(off);
        
        //auto center = bounds().pos() + bounds().size() * 0.5;
        _blob_id = bid::from_blob(*this);
    }
    
    void Blob::scale_coordinates(const cmn::Vec2 &scale) {
        auto center = bounds().pos() + bounds().size() * 0.5;
        Vec2 offset = center.mul(scale) - center;
        
        add_offset(offset);
    }

    

    std::string bid::toStr() const {
        return Meta::toStr<uint32_t>(_id);
    }

    bid bid::fromStr(const std::string& str) {
        return bid(Meta::fromStr<uint32_t>(str));
    }

    size_t Blob::memory_size() const {
        size_t overall = sizeof(Blob);
        if(_pixels)
            overall += _pixels->size() * sizeof(uchar);
        overall += _hor_lines->size() * sizeof(decltype(_hor_lines)::element_type::value_type);
        return overall;
    }
    
    std::string Blob::toStr() const {
        return "pv::Blob<" + Meta::toStr(blob_id()) + " " + Meta::toStr(bounds().pos() + bounds().size() * 0.5) + " " + Meta::toStr(_pixels ? _pixels->size() * SQR(cm_per_pixel) : -1) + ">";
    }
}
