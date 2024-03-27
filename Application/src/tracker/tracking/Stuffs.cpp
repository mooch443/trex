#include "Stuffs.h"

namespace track {

PostureStuff::~PostureStuff() {
    if(head) delete head;
    if(centroid_posture) delete centroid_posture;
}

}
