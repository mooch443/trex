#include "DetectionTypes.h"
#include <misc/metastring.h>
#include <misc/SoftException.h>


using namespace cmn;

namespace track::detect {
    std::string Keypoint::toStr() const {
        return "Keypoint<" + Meta::toStr(bones) + ">";
    }

    const Bone& Keypoint::bone(size_t index) const {
        if (index >= bones.size()) {
            throw SoftException("Index ", index, " out of bounds for array of size ", bones.size(), ".");
        }
        return bones[index];
    }

    blob::Pose Keypoint::toPose() const {
        std::vector<blob::Pose::Point> coords;
        for (auto& b : bones)
            coords.push_back(blob::Pose::Point(b.x, b.y));
        return blob::Pose{
            .points = std::move(coords)
        };
    }
}