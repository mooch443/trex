#pragma once

#include <misc/defines.h>
#include <misc/vec2.h>
#include <misc/Blob.h>
#include <processing/Background.h>
#include <misc/Image.h>
#include <misc/Grid.h>
#include <misc/ProximityGrid.h>
#include <misc/vec2.h>

namespace Output {
    class ResultsFormat;
}

namespace pv {
    class Blob;
    using BlobPtr = std::shared_ptr<pv::Blob>;
    struct CompressedBlob;

struct bid {
    static constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
    uint32_t _id = invalid;
    bid() = default;
    constexpr bid(uint32_t v) : _id(v) {}
    
    explicit constexpr operator uint32_t() const {
        return _id;
    }
    explicit constexpr operator int64_t() const { return static_cast<int64_t>(_id); }
    explicit constexpr operator uint64_t() const { return static_cast<uint64_t>(_id); }
    constexpr bool operator==(const bid& other) const {
        return other._id == _id;
    }
    constexpr bool operator!=(const bid& other) const {
        return other._id != _id;
    }
    //constexpr bid(uint32_t b) : _id(b) {}
    constexpr bool valid() const { return _id != invalid; }
    
    constexpr bool operator<(const bid& other) const {
        return _id < other._id;
    }
    constexpr bool operator>(const bid& other) const {
        return _id > other._id;
    }
    
    constexpr bool operator<=(const bid& other) const {
        return _id <= other._id;
    }
    constexpr bool operator>=(const bid& other) const {
        return _id >= other._id;
    }
    
    std::string toStr() const;
    static std::string class_name() { return "pv::bid"; }
    static bid fromStr(const std::string& str);

    static constexpr uint32_t from_data(ushort x0, ushort x1, ushort y0, uint8_t N) {
        assert((uint32_t)x0 < (uint32_t)4096u);
        assert((uint32_t)x1 < (uint32_t)4096u);
        assert((uint32_t)y0 < (uint32_t)4096u);
        
        return (uint32_t(x0 + (x1 - x0) / 2) << 20)
                | ((uint32_t(y0) & 0x00000FFF) << 8)
                |  (uint32_t(N)  & 0x000000FF);
    }
    
    constexpr cmn::Vec2 calc_position() const {
        auto x = (_id >> 20) & 0x00000FFF;
        auto y = (_id >> 8) & 0x00000FFF;
        //auto N = id & 0x000000FF;
        return cmn::Vec2(x, y);
    }
    //static uint32_t id_from_position(const cmn::Vec2&);
    static inline bid from_blob(const pv::Blob& blob);
    static inline bid from_blob(const pv::CompressedBlob& blob);
};

}

namespace std
{
    template <>
    struct hash<pv::bid>
    {
        size_t operator()(const pv::bid& k) const
        {
            return std::hash<uint32_t>{}((uint32_t)k);
        }
    };
}

namespace pv {
    
    class Blob : public cmn::Blob {
    public:
        using line_ptr_t = std::shared_ptr<std::vector<cmn::HorizontalLine>>;
        using pixel_ptr_t = std::shared_ptr<const std::vector<uchar>>;
        
    protected:
        GETTER_PTR(pixel_ptr_t, pixels)
        GETTER_I(bool, split, false)
        GETTER_I(bid, parent_id, pv::bid::invalid)
        GETTER_I(bid, blob_id, pv::bid::invalid)
        GETTER_SETTER(bool, tried_to_split)
        
        float _recount;
        int32_t _recount_threshold;
        uchar _color_percentile_5, _color_percentile_95;
        
    public:
        Blob();
        Blob(line_ptr_t lines, pixel_ptr_t pixels);
        Blob(std::shared_ptr<const std::vector<cmn::HorizontalLine>> lines, decltype(_pixels) pixels);
        Blob(const cmn::Blob* blob, pixel_ptr_t pixels);
        Blob(const pv::Blob& other);
        
        BlobPtr threshold(int32_t value, const cmn::Background& background);
        std::tuple<cmn::Vec2, std::unique_ptr<cmn::Image>> image(const cmn::Background* background = NULL, const cmn::Bounds& restricted = cmn::Bounds(-1,-1,-1,-1)) const;
        std::tuple<cmn::Vec2, std::unique_ptr<cmn::Image>> alpha_image(const cmn::Background& background, int32_t threshold) const;
        std::tuple<cmn::Vec2, std::unique_ptr<cmn::Image>> difference_image(const cmn::Background& background, int32_t threshold) const;
        std::tuple<cmn::Vec2, std::unique_ptr<cmn::Image>> thresholded_image(const cmn::Background& background, int32_t threshold) const;
        std::tuple<cmn::Vec2, std::unique_ptr<cmn::Image>> luminance_alpha_image(const cmn::Background& background, int32_t threshold) const;
        std::tuple<cmn::Vec2, std::unique_ptr<cmn::Image>> equalized_luminance_alpha_image(const cmn::Background& background, int32_t threshold, float minimum, float maximum) const;
        std::tuple<cmn::Vec2, std::unique_ptr<cmn::Image>> binary_image(const cmn::Background& background, int32_t threshold) const;
        std::tuple<cmn::Vec2, std::unique_ptr<cmn::Image>> binary_image() const;
        
        void set_pixels(decltype(_pixels) pixels);
        //void set_pixels(const cmn::grid::PixelGrid &grid, const cmn::Vec2& offset = cmn::Vec2(0));
        
        static decltype(_pixels) calculate_pixels(cmn::Image::Ptr image, const decltype(_hor_lines)& lines);
        
        float recount(int32_t threshold) const;
        float recount(int32_t threshold, const cmn::Background&);
        void force_set_recount(int32_t threshold, float value = -1);
        void transfer_backgrounds(const cmn::Background& from, const cmn::Background& to, const cmn::Vec2& dest_offset = cmn::Vec2());
        
        void set_split(bool split, pv::BlobPtr parent);
        
        std::string name() const;
        virtual void add_offset(const cmn::Vec2& off) override;
        void scale_coordinates(const cmn::Vec2& scale);
        size_t memory_size() const;
        
        template<typename... Args>
        static pv::BlobPtr make(Args... args) {
            return std::make_shared<pv::Blob>(std::forward<Args>(args)...);
        }
        
        bool operator!=(const pv::Blob& other) const;
        bool operator==(const pv::Blob& other) const;
        std::string toStr() const override;
        
    protected:
        friend class Output::ResultsFormat;
        friend struct CompressedBlob;
        
        void set_split(bool);
        void set_parent_id(const bid& parent_id);
        void init();
    };

    struct ShortHorizontalLine {
    private:
        //! starting and end position on x
        //  the last bit of _x1 is a flag telling the program
        //  whether this line is the last line on the current y coordinate.
        //  the following lines are on current_y + 1.
        uint16_t _x0, _x1;
        
    public:
        //! compresses an array of HorizontalLines to an array of ShortHorizontalLines
        static std::vector<ShortHorizontalLine> compress(const std::vector<cmn::HorizontalLine>& lines);
        //! uncompresses an array of ShortHorizontalLines back to HorizontalLines
        static Blob::line_ptr_t uncompress(uint16_t start_y, const std::vector<ShortHorizontalLine>& compressed);
        
    public:
        constexpr ShortHorizontalLine() : _x0(0), _x1(0) {}
        
        constexpr ShortHorizontalLine(uint16_t x0, uint16_t x1, bool eol = false)
            : _x0(x0), _x1((x1 & 0x7FFF) | uint16_t(eol << 15))
        {
            assert(x1 < 32768); // MAGIC NUMBERZ (uint16_t - 1 bit)
        }
        
        constexpr uint16_t x0() const { return _x0; }
        constexpr uint16_t x1() const { return _x1 & 0x7FFF; }
        
        //! returns true if this is the last element on the current y coordinate
        //  if true, the following lines are on current_y + 1.
        //  @note stored in the last bit of _x1
        constexpr bool eol() const { return (_x1 & 0x8000) != 0; }
        constexpr void eol(bool v) { _x1 = (_x1 & 0x7FFF) | uint16_t(v << 15); }
    };

struct CompressedBlob {
    //! this represents parent_id (2^1), split (2^0) and tried_to_split (2^2)
    uint8_t status_byte = 0;
    pv::bid parent_id = pv::bid::invalid;
    mutable pv::bid own_id = pv::bid::invalid;

    //! y of first position (everything is relative to this)
    uint16_t start_y{0};
    std::vector<ShortHorizontalLine> lines;

    CompressedBlob() = default;
    CompressedBlob(const pv::BlobPtr& val) :
        parent_id(val->parent_id()),
        own_id(val->blob_id())
    {
        status_byte = (uint8_t(val->split())             * 0x1)
                    | (uint8_t(val->parent_id().valid()) * 0x2)
                    | (uint8_t(val->tried_to_split())    * 0x4);
        lines = ShortHorizontalLine::compress(val->hor_lines());
        start_y = val->lines()->empty() ? 0 : val->lines()->front().y;
    }
        
    bool split() const { return status_byte & 0x1; }
    cmn::Bounds calculate_bounds() const;
        
    pv::BlobPtr unpack() const;
    uint64_t num_pixels() const {
        // adding +1 to result for each line (in order to include x1 as part of the total count)
        uint64_t result = lines.size();
            
        // adding all line lengths
        for(auto &line : lines)
            result += line.x1() - line.x0();
            
        return result;
    }
        
    bid blob_id() const;
};

static_assert(int32_t(-1) == (uint32_t)bid::invalid, "Must be equal to ensure backwards compatibility.");

inline bid bid::from_blob(const pv::Blob &blob) {
    if(!blob.lines() || blob.lines()->empty())
        return bid::invalid;
    
    return from_data(blob.lines()->front().x0,
                     blob.lines()->front().x1,
                     blob.lines()->front().y,
                     blob.lines()->size());
}

inline bid bid::from_blob(const pv::CompressedBlob &blob) {
    if(blob.lines.empty())
        return bid::invalid;
    
    return from_data(blob.lines.front().x0(),
                     blob.lines.front().x1(),
                     blob.start_y,
                     blob.lines.size());
}

}
