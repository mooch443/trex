#ifndef _SPLIT_BLOB_H
#define _SPLIT_BLOB_H

#include <commons.pc.h>
#include <misc/GlobalSettings.h>
#include <misc/PVBlob.h>

namespace track { class SplitBlob; }

namespace cmn { namespace CPULabeling { struct ListCache_t; }}

namespace track {

namespace split {

ENUM_CLASS(Action,
    KEEP,
    KEEP_ABORT,
    REMOVE,
    ABORT,
    TOO_FEW,
           SKIP,
    NO_CHANCE
)

using Action_t = Action::Class;

}

}

//! This class tries to find multiple blobs within a big blob.
//  One blob represents one individual.
class track::SplitBlob {
    struct ResultProp {
        float fitness;
        float ratio;
        int threshold{-1};
        std::vector<pv::BlobPtr> blobs;
        //std::vector<std::vector<uchar>> pixels;
        
        std::string toStr() const {
            return "t:"+cmn::Meta::toStr(threshold)+" obj:"+cmn::Meta::toStr(blobs.size())+" r:"+cmn::Meta::toStr(ratio);
        }
        static std::string class_name() {
            return "SplitBlob::ResultProp";
        }
    };
    
private:
    cv::Mat _original, _original_grey;
    std::atomic<size_t> max_objects;
    std::mutex mutex;
    
    uint8_t min_pixel, max_pixel;
    
    // parameters
    pv::BlobWeakPtr _blob;
    std::vector<uchar> _diff_px;
    cmn::CPULabeling::ListCache_t* _cache{nullptr};
    
public:
    SplitBlob(cmn::CPULabeling::ListCache_t* cache, const cmn::Background& average, pv::BlobWeakPtr blob);
    
    /**
     * @param presumed_nr number of individuals to find
     * @param blob The big blob containing potentially more than 1 blob
     * @param pixels greyscale pixel values for all the lines in blob
     * @param average background image of the whole arena
     * @return either an empty list if the blob cannot be split, or
     * a number of Blobs that seem to be two individuals. Also returns
     * every Blob paired with its grey value array
     */
    std::vector<pv::BlobPtr> split(size_t presumed_nr, const std::vector<cmn::Vec2>& centers);
    
private:
    size_t apply_threshold(cmn::CPULabeling::ListCache_t* cache, int threshold, std::vector<pv::BlobPtr> &output);
    split::Action_t evaluate_result_multiple(size_t presumed_nr, float first_size, std::vector<pv::BlobPtr>&);
};

#endif
