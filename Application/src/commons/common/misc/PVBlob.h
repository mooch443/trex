#pragma once

#include <types.h>
#include <misc/vec2.h>
#include <misc/Blob.h>
#include <processing/Background.h>
#include <misc/Image.h>
#include <misc/Grid.h>
#include <misc/ProximityGrid.h>

namespace Output {
    class ResultsFormat;
}

namespace pv {
    class Blob;
    using BlobPtr = std::shared_ptr<pv::Blob>;
    struct CompressedBlob;
    
    class Blob : public cmn::Blob {
    protected:
        GETTER_PTR(std::shared_ptr<const std::vector<uchar>>, pixels)
        GETTER(bool, split)
        GETTER(long_t, parent_id)
        GETTER(uint32_t, blob_id)
        GETTER_SETTER(bool, tried_to_split)
        
        float _recount;
        int32_t _recount_threshold;
        uchar _color_percentile_5, _color_percentile_95;
        
    public:
        Blob();
        Blob(std::shared_ptr<std::vector<cmn::HorizontalLine>> lines, decltype(_pixels) pixels);
        Blob(std::shared_ptr<const std::vector<cmn::HorizontalLine>> lines, decltype(_pixels) pixels);
        //Blob(const std::vector<HorizontalLine>& lines, decltype(_pixels) pixels);
        Blob(const cmn::Blob* blob, decltype(_pixels) pixels);
        Blob(const pv::Blob& other);
        //std::vector<std::shared_ptr<Blob>> threshold(int value, const Background& background) const;
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
        static cmn::Vec2 position_from_id(uint32_t id);
        static uint32_t id_from_position(const cmn::Vec2&);
        operator cmn::MetaObject() const override;
        
    protected:
        friend class Output::ResultsFormat;
        friend struct CompressedBlob;
        
        void set_split(bool);
        void set_parent_id(long_t parent_id);
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
        static std::shared_ptr<std::vector<cmn::HorizontalLine>> uncompress(uint16_t start_y, const std::vector<ShortHorizontalLine>& compressed);
        
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
        constexpr static uint32_t invalid = std::numeric_limits<uint32_t>::max();
        
        //! this represents parent_id (2^1), split (2^0) and tried_to_split (2^2)
        uint8_t status_byte;
        int32_t parent_id;
        mutable uint32_t own_id = invalid;
        
        //! y of first position (everything is relative to this)
        uint16_t start_y;
        std::vector<ShortHorizontalLine> lines;
        
        CompressedBlob() : status_byte(0), parent_id(-1) {}
        CompressedBlob(const pv::BlobPtr& val) {
            parent_id = val->parent_id();
            own_id = val->blob_id();
            status_byte = (uint8_t(val->split())           * 0x1)
                        | (uint8_t(val->parent_id() != -1) * 0x2)
                        | (uint8_t(val->tried_to_split())  * 0x4);
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
        uint32_t blob_id() const;
    };
}
