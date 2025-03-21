#include "LabelWrapper.h"

namespace cmn::gui {

LabelWrapper::LabelWrapper(
       LabelCache_t& cache,
       derived_ptr<Label>&& label)
    : _label(std::move(label)), _cache(&cache)
{
    set_children({Layout::Ptr(_label)});
}


LabelWrapper::~LabelWrapper() {
    _cache->returnObject(std::move(_label));
}

Label* LabelWrapper::label() const {
    return _label.get();
}

}
