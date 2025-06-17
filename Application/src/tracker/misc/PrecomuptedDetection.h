#pragma once
#include <commons.pc.h>
#include <misc/DetectionTypes.h>
#include <misc/TileImage.h>
#include <misc/Image.h>

namespace track {

struct TREX_EXPORT PrecomputedDetection {
    using frame_data_t = std::unordered_map<Frame_t, std::vector<Bounds>>;
    
    PrecomputedDetection(file::PathArray&& filename, Image::Ptr&&, meta_encoding_t::Class);
    static void set_background(Image::Ptr&&, meta_encoding_t::Class);
    
    static detect::ObjectDetectionType_t type() { return detect::ObjectDetectionType::precomputed; }
    static std::future<SegmentationData> apply(TileImage&& tiled);
    static void deinit();
    static double fps();
    
    static PipelineManager<TileImage, true>& manager();
    
private:
    static void apply(std::vector<TileImage>&& tiled);
    friend struct Detection;
    
    struct Data;
    friend class PrecomputedDetectionCache;
    
    static Data& data();
};

template<typename MatchType,
         utils::StringLike Name_t,
         typename Target_t>
bool match_name(
        const std::initializer_list<MatchType>& matches,
        Name_t&& name,
        Target_t& target)
{
    using Matches = cmn::remove_cvref_t<decltype(matches)>;
    
    bool does_match{false};
    for(auto& match : matches) {
        if constexpr(std::same_as<typename Matches::value_type, char>)
        {
            if(utils::contains(utils::lowercase(name), match))
            {
                does_match = true;
                break;
            }
            
        } else {
            if(utils::contains(utils::lowercase(name), match))
            {
                does_match = true;
                break;
            }
        }
    }
    
    if(does_match) {
        if(not target.has_value()) {
            target = std::string_view(name);
            return true;
        } else {
            FormatWarning("Found candidate ", name, " to be used as the ",std::vector(matches),"-column, but already have: ", target);
        }
    }
    
    return false;
};

// Helper to cache precomputed detection data in binary form and memory-map for fast reads
class TREX_EXPORT PrecomputedDetectionCache {
public:
    using frame_data_t = PrecomputedDetection::frame_data_t;

    explicit PrecomputedDetectionCache(const file::Path& csv_path);
    PrecomputedDetectionCache(PrecomputedDetectionCache&&) noexcept = default;
    PrecomputedDetectionCache& operator=(PrecomputedDetectionCache&&) noexcept = default;
    
    std::optional<frame_data_t::mapped_type> get_frame_data(Frame_t);

private:
    static void buildCache(const file::Path& csv_path, const file::Path& cache_path);

#pragma pack(push, 1)
    struct Header {
        union {
            struct { char tag[3]; uint8_t ver; } parts;
            char  str[4];
        } magic;
        uint64_t file_hash;
        uint32_t index_count;
    };

    struct IndexEntry {
        Frame_t frame;
        uint32_t count;
        uint64_t offset;
    };
    
    struct FileIndexEntry {
        uint32_t frame;
        uint32_t count;
        uint64_t offset;
    };
#pragma pack(pop)
    
    struct Offsets {
        uint64_t offset;
        uint32_t count;
    };

    file::Path                    _cache_path;
    cmn::DataFormat               _df;
    const char*                   _map_ptr{nullptr};
    size_t                        _map_size{0};
    std::vector<IndexEntry>       _index_entries;
    std::map<Frame_t, Offsets>    _index_map;
    frame_data_t                  _result;

    // helper: quick hash based on path & size
    static uint64_t computeFileHash(const file::Path& p);
};


}
