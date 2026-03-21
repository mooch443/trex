#include <tracking/RecTask.h>

#include <file/DataLocation.h>
#include <misc/GlobalSettings.h>
#include <python/BackendRegistration.h>
#include <python/GPURecognition.h>
#include <python/PythonWrapper.h>

namespace track {

void register_yolo_backend();
void register_sam3_backend();

namespace {

constexpr auto tagwork = "pretrained_tagwork";

void init_tracking_recognition() {
    namespace py = Python;
    py::init().get();

    py::schedule(py::pack([]() {
        using py = track::PythonIntegration;

        py::import_module(tagwork);
        auto path = READ_SETTING(tags_model_path, file::Path);
        if(path.empty() || !path.exists()) {
            throw SoftException("The model at ", path, " can not be found. Please set `tags_model_path` to point to an h5 file with a pretrained network. See `https://trex.run/docs/parameters_trex.html#tags_model_path` for more information.");
        }
        py::set_variable("model_path", path.str(), tagwork);
        py::set_variable("width", 32, tagwork);
        py::set_variable("height", 32, tagwork);
        py::run(tagwork, "init");
        Print("Initialized tagging successfully.");
    })).get();
}

void predict_tracking_ids(std::vector<cmn::Image::Ptr>&& images,
                          cmn::package::F<void(std::vector<int64_t>)>&& receive)
{
    namespace py = Python;
    py::schedule([images = std::move(images), receive = std::move(receive)]() mutable {
        using py = track::PythonIntegration;

        py::set_variable("tag_images", images, tagwork);
        py::set_function("receive", std::move(receive), tagwork);
        py::run(tagwork, "predict");
        py::unset_function("receive", tagwork);
    }).get();
}

void register_tracking_services() {
    install_rec_task_backend(RecTaskBackend{
        .init = init_tracking_recognition,
        .deinit = []() {},
        .predict = predict_tracking_ids
    });
}

} // namespace

void register_python_backends() {
    static std::once_flag once;
    std::call_once(once, []() {
        register_yolo_backend();
        register_sam3_backend();
        register_tracking_services();
    });
}

} // namespace track
