#include <commons.pc.h>
#include <types.h>

namespace track::detect {
    class Bone {
    public:
        float x;
        float y;
        //float conf;
        std::string toStr() const {
            return "Bone<" + Meta::toStr(x) + "," + Meta::toStr(y) + ">";
        }
    };

    class Keypoint {
    public:
        std::vector<Bone> bones;
        std::string toStr() const;
        const Bone& bone(size_t index) const;
        cmn::blob::Pose toPose() const;
    };
}
