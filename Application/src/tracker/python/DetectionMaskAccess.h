#pragma once

#include <commons.pc.h>
#include <core/GPURecognitionTypes.h>

namespace track::detect {

class DetectionMaskAccess {
public:
    static std::vector<MaskData>& masks(Result& result) {
        return result._masks;
    }

    static std::optional<MaskData>& semantic_mask(Result& result) {
        return result._semantic_mask;
    }

    static MaskData make_mask(std::vector<uint8_t>&& bytes, int rows, int cols) {
        return MaskData(std::move(bytes), rows, cols);
    }
};

} // namespace track::detect
