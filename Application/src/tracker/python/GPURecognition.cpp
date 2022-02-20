#define PYBIND11_CPP17
#ifdef PYBIND11_CPP14
#undef PYBIND11_CPP14
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#pragma clang diagnostic pop

#include <misc/Image.h>
#include <misc/SpriteMap.h>
#include <misc/vec2.h>

#include <misc/default_settings.h>
#include <misc/default_config.h>
#include <misc/GlobalSettings.h>

namespace py = pybind11;

template<typename T>
bool CHECK_NONE(T obj) {
    return !obj.ptr() || obj.is_none();
}

namespace pybind11 {
    namespace detail {
        template<> struct type_caster<cmn::Image::Ptr>
        {
        public:

            PYBIND11_TYPE_CASTER(cmn::Image::Ptr, _("Image::Ptr"));

            // Conversion part 1 (Python -> C++)
            bool load(py::handle , bool )
            {
                /*if ( !convert and !py::array_t<T>::check_(src) )
                  return false;

                auto buf = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
                if ( !buf )
                  return false;

                auto dims = buf.ndim();
                if ( dims != 3  )
                  return false;

                std::vector<size_t> shape(3);

                for ( int i = 0 ; i < 3 ; ++i )
                  shape[i] = buf.shape()[i];

                value = Matrix3D<T>(shape, buf.data(), buf.data()+buf.size())*/

                return false;
            }

            //Conversion part 2 (C++ -> Python)
            static py::handle cast(const cmn::Image::Ptr& src, py::return_value_policy, py::handle)
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                /*return py::buffer_info(
                   src->data(),
                   sizeof(uint8_t),
                   py::format_descriptor<uint8_t>::format(),
                   3,
                   { src->rows, src->cols, src->dims },
                   {
                       sizeof(uint8_t) * src->dims * src->cols,
                       sizeof(uint8_t) * src->dims,
                       sizeof(uint8_t)
                   }
                );*/
                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
            
            static py::handle cast(const cmn::Image::UPtr& src, py::return_value_policy , py::handle )
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
        };
    
    template<> struct type_caster<cmn::Image::UPtr>
    {
    public:

        PYBIND11_TYPE_CASTER(cmn::Image::UPtr, _("Image::UPtr"));

        // Conversion part 1 (Python -> C++)
        bool load(py::handle, bool)
        {
            return false;
        }

        //Conversion part 2 (C++ -> Python)
        static py::handle cast(const cmn::Image::UPtr& src, py::return_value_policy, py::handle)
        {

            std::vector<size_t> shape{ src->rows, src->cols, src->dims };
            std::vector<size_t> strides{
                sizeof(uint8_t) * src->dims * src->cols,
                sizeof(uint8_t) * src->dims,
                sizeof(uint8_t)
            };

            py::array a(std::move(shape), std::move(strides), src->data());
            return a.release();
        }
    };
    }
} // namespace pybind11::detail

#ifdef MESSAGE_TYPE
    #undef MESSAGE_TYPE
#endif
#define MESSAGE_TYPE(NAME, TYPE, FORCE_CALLBACK, COLOR, PREFIX) \
void NAME(const char *cmd, ...) { \
    va_list args; \
    va_start(args, cmd); \
    \
    std::string str;\
    DEBUG::ParseFormatString(str, cmd, args); \
    DEBUG::StatusMsg msg(DEBUG::DEBUG_TYPE::TYPE, str.c_str(), -1, NULL); \
    msg.color = DEBUG::CONSOLE_COLORS::COLOR; \
    msg.prefix = PREFIX; \
    msg.force_callback = false; \
    DEBUG::ParseStatusMessage(&msg); \
    \
    va_end(args); \
}

MESSAGE_TYPE(PythonLog, TYPE_INFO, false, CYAN, "python");
MESSAGE_TYPE(PythonWarn, TYPE_WARNING, false, YELLOW, "python");

std::shared_ptr<cmn::GlobalSettings> _settings = nullptr;
std::function<void(const std::string&, const cv::Mat&)> _mat_display = [](auto&, auto&) {

};

PYBIND11_EMBEDDED_MODULE(TRex, m) {
    namespace py = pybind11;
    /*py::class_<cmn::Image, cmn::Image::Ptr>(m, "TRex", py::buffer_protocol())
    .def_buffer([](cmn::Image &m) -> py::buffer_info {
        return py::buffer_info(
           m.data(),
           sizeof(uint8_t),
           py::format_descriptor<uint8_t>::format(),
           3,
           { m.rows, m.cols, m.dims },
           {
               sizeof(uint8_t) * m.dims * m.cols,
               sizeof(uint8_t) * m.dims,
               sizeof(uint8_t)
           }
        );
    });*/

    m.def("log", [](std::string text) {
        PythonLog("%S", &text);
        });
    m.def("warn", [](std::string text) {
        PythonWarn("%S", &text);
        });

    /*m.def("show_work_image", [](std::string name, pybind11::buffer b) {
#if CMN_WITH_IMGUI_INSTALLED
        namespace py = pybind11;
        py::buffer_info info = b.request();

        if (info.format != py::format_descriptor<uint8_t>::format())
            throw std::runtime_error("Incompatible format: expected a uint8_t array!");

        if (info.ndim != 3 && info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        auto map = cv::Mat((int)info.shape[0], (int)info.shape[1], (int)CV_8UC(info.ndim == 2 ? 1 : info.shape[2]), static_cast<uint8_t*>(info.ptr));

        //auto desktop = Size2(sf::VideoMode::getDesktopMode().width * 0.65,
         //                    sf::VideoMode::getDesktopMode().height * 0.65);
        if (GUI::instance()) {
            cmn::Image::Ptr image = std::make_shared<cmn::Image>(map.rows, map.cols, 4);

            if (map.channels() == 3)
                cv::cvtColor(map, image->get(), cv::COLOR_BGR2BGRA);
            else if (map.channels() == 4)
                map.copyTo(image->get());
            //else if(map.channels() == 4)
            //    cv::cvtColor(map, image->get(), cv::COLOR_RGBA2BGRA);

            GUI::work().set_image(name, image);
        }
#endif
    }, pybind11::arg().none(), pybind11::arg().noconvert());*/

    m.def("video_size", []() -> pybind11::dict {
        using namespace pybind11::literals;
        using namespace cmn;
        pybind11::dict d;
        auto w = _settings->map().get<cmn::Size2>("video_size").value().width,
            h = _settings->map().get<cmn::Size2>("video_size").value().height;

        d["width"] = w;
        d["height"] = h;
        return d;
    });
    
    m.def("setting", [](const std::string& name) -> std::string {
        using namespace pybind11::literals;
        using namespace cmn;
        return _settings->map().operator[](name).get().valueString();
    });
    
    m.def("setting", [](const std::string& name, const std::string& value) {
        using namespace pybind11::literals;
        using namespace cmn;
        try {
            constexpr auto accessLevel = default_config::AccessLevelType::PUBLIC;
            if(!_settings->has_access(name, accessLevel))
                Error("User cannot write setting '%S' (AccessLevel::%s).", &name, _settings->access_level(name).name());
            else {
                if(_settings->has(name)) {
                    _settings->map().operator[](name).get().set_value_from_string(value);
                } else
                    Error("Setting '%S' unknown.", &name);
            }
        } catch(...) {
            Except("Failed to set setting '%S' to '%S'.", &name, &value);
        }
    });

    m.def("imshow", [](std::string name, pybind11::buffer b) {
#if CMN_WITH_IMGUI_INSTALLED
        namespace py = pybind11;
        using namespace cmn;
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some sanity checks ... */
        if (info.format != py::format_descriptor<uint8_t>::format())
            throw std::runtime_error("Incompatible format: expected a uint8_t array!");

        if (info.ndim != 3 && info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        if (!_settings->map().get<bool>("nowindow").value()) {
            auto map = cv::Mat((int)info.shape[0], (int)info.shape[1], (int)CV_8UC(info.ndim == 2 ? 1 : info.shape[2]), static_cast<uint8_t*>(info.ptr));
            _mat_display(name, map);
            //tf::imshow(name, map);
        }
#endif
    }, pybind11::arg().none(), pybind11::arg().noconvert());

    py::bind_vector<std::vector<cmn::Image::Ptr>>(m, "ImageVector", "Vector of images");
    py::bind_vector<std::vector<float>>(m, "FloatVector", "Float vector");
    py::bind_vector<std::vector<std::string>>(m, "StringVector", "String vector");
    py::bind_vector<std::vector<long_t>>(m, "LongVector", "Long vector");
    py::bind_vector<std::vector<uchar>>(m, "UcharVector", "Uchar vector");
}

#include "GPURecognition.h"
#include <pybind11/stl.h>
#include <gui/WorkProgress.h>
#include <misc/SoftException.h>
#include <misc/metastring.h>
#include <misc/Timer.h>

namespace track {
    namespace py = pybind11;

    struct PackagedTask {
        std::packaged_task<bool(void)> _task;
        bool _can_run_before_init = false;
    };

    std::shared_ptr<py::scoped_interpreter> guard = nullptr;
    pybind11::module numpy, TRex, _main;
    pybind11::dict* _locals = nullptr;
    std::mutex module_mutex;

    std::map<std::string, std::string> contents;
    std::map<std::string, pybind11::module> _modules;

    std::atomic<bool> _terminate = false;
    std::map<Idx_t, std::deque<std::tuple<long_t, Image::Ptr>>> _classes;
    std::map<Idx_t, std::set<long_t>> _received;
    std::map<Idx_t, std::set<long_t>> _sent_to_training;
    std::map<Idx_t, std::vector<Image::Ptr>> _test_data;

    std::vector<PackagedTask> tasks;

    std::thread* _network_update_thread = nullptr;
    std::condition_variable _update_condition;
    std::mutex _data_mutex, _initialize_mutex;
    std::condition_variable _initialize_condition;
    std::thread::id _saved_id;

    std::unique_ptr<std::promise<bool>> _initialize_promise;
    std::shared_future<bool> _initialize_future;

    void PythonIntegration::set_settings(std::shared_ptr<GlobalSettings> obj) {
        GlobalSettings::set_instance(obj);
        _settings = obj;
    }

    template<typename T>
    void set_function_internal(const char* name_, T f, const std::string& m);

    void PythonIntegration::set_display_function(std::function<void(const std::string&, const cv::Mat&)> fn) {
        _mat_display = fn;
    }

    std::atomic_bool& PythonIntegration::python_initialized() {
        static std::atomic_bool _python_initialized = false;
        return _python_initialized;
    }
    std::atomic_bool& PythonIntegration::python_initializing() {
        static std::atomic_bool _python_initializing = false;
        return _python_initializing;
    }
    std::atomic_bool& PythonIntegration::python_gpu_initialized() {
        static std::atomic_bool _python_gpu_initialized = false;
        return _python_gpu_initialized;
    }
    std::atomic_int& PythonIntegration::python_major_version() {
        static std::atomic_int _python_major_version = 0;
        return _python_major_version;
    }
    std::atomic_int& PythonIntegration::python_minor_version() {
        static std::atomic_int _python_minor_version = 0;
        return _python_minor_version;
    }
    std::atomic_int& PythonIntegration::python_uses_gpu() {
        static std::atomic_int _python_uses_gpu = false;
        return _python_uses_gpu;
    }
    
    std::string& PythonIntegration::python_init_error() {
        static std::string _python_init_error = "";
        return _python_init_error;
    }
    std::string& PythonIntegration::python_gpu_name() {
        static std::string _python_gpu_name;
        return _python_gpu_name;
    }
    
    PythonIntegration*& PythonIntegration::instance(bool check) {
        static PythonIntegration *_instance = nullptr;
        if(!_instance && !check)
            _instance = new PythonIntegration;
        return _instance;
    }
    
    PythonIntegration::PythonIntegration()
    {
        initialize();
    }

    PythonIntegration::~PythonIntegration() {
        shutdown();
    }
    
    void PythonIntegration::quit() {
        if(_network_update_thread) {
            delete instance(false);
        }
        if(instance(true))
            instance(true) = nullptr;
    }
    
    void PythonIntegration::shutdown() {
        _terminate = true;
        python_initialized() = false;
        
        if(_network_update_thread) {
            _update_condition.notify_all();
            
            _network_update_thread->join();
            delete _network_update_thread;
            _network_update_thread = nullptr;
        }
    }

void PythonIntegration::reinit() {
    async_python_function([]() -> bool {
        using namespace py::literals;
        python_gpu_initialized() = false;
        python_initializing() = true;
        
        auto fail = [](const auto& e, int line){
            Debug("Python runtime error (%s:%d): '%s'", __FILE_NO_PATH__, line, e.what());
            python_gpu_initialized() = false;
            python_initializing() = false;
        };
        
        try {
            //if(_settings->map().get<bool>("recognition_enable").value())
            {
                async_python_function([fail](){
                    try {
                        auto cmd = utils::read_file("trex_init.py");
                        py::exec(cmd);
                        python_gpu_initialized() = true;
                        python_initializing() = false;
                        
                    } catch(const UtilsException& ex) {
                        Warning("Error while executing 'trex_init.py'. Content: %s", ex.what());
                        fail(ex, __LINE__);
                        return false;
                        
                    } catch(py::error_already_set& e) {
                        fail(e, __LINE__);
                        e.restore();
                        return false;
                    }
                    
                    return true;
                    
                }, Flag::FORCE_ASYNC, true);
                
            }
            
            return true;
            
        } catch(py::error_already_set &e) {
            fail(e, __LINE__);
            e.restore();
            return false;
        }
        
    }, Flag::DEFAULT, true);
}
    
    void PythonIntegration::initialize() {
        using namespace py::literals;
        
        _initialize_promise = std::make_unique<std::promise<bool>>();
        _initialize_future = _initialize_promise->get_future().share();
        
        _network_update_thread = new std::thread([]() -> void {
            cmn::set_thread_name("PythonIntegration::update");
            std::unique_lock<std::mutex> lock(_data_mutex);
            _saved_id = std::this_thread::get_id();

            python_initialized() = false;
            python_initializing() = true;
            _terminate = false;
            
            auto fail = [](const auto& e, int line){
                python_init_error() = e.what();
                python_initializing() = false;
                Debug("Python runtime error (%s:%d): '%s'", __FILE_NO_PATH__, line, e.what());
                
                python_initialized() = false;
                python_initializing() = false;
                
                //guard = nullptr;
                _initialize_promise->set_value(false);
            };

            try {
#if defined(WIN32)
                if (!getenv("TREX_DONT_SET_PATHS")) {
                    std::string sep = "/";
                    auto home = Py_GetPythonHome();
                    auto home2 = SETTING(python_path).value<file::Path>().str();
                    if(file::Path(home2).exists() && file::Path(home2).is_regular())
                        home2 = file::Path(home2).remove_filename().str();
                    Debug("Setting home to '%S'", &home2);

                    if (!home2.empty()) {
                        home2 = utils::find_replace(home2, "/", sep);

                        int nChars = MultiByteToWideChar(CP_ACP, 0, home2.c_str(), -1, NULL, 0);
                        wchar_t* pwcsName = new wchar_t[nChars];
                        MultiByteToWideChar(CP_ACP, 0, home2.c_str(), -1, (LPWSTR)pwcsName, nChars);
                        Py_SetPythonHome(pwcsName);
                        SetEnvironmentVariable("PYTHONHOME", home2.c_str());

                        // delete it
                        delete[] pwcsName;
                    }
                }
#endif

                py::initialize_interpreter();
                
                _main = py::module::import("__main__");
                _main.def("set_version", [](std::string x, bool has_gpu, std::string physical_name) {
#ifndef NDEBUG
                    Debug("set_version called with '%S' and '%S' - %s", &x, &physical_name, has_gpu?"gpu":"no gpu");
#endif
                    auto array = utils::split(x, ' ');
                    if(array.size() > 0) {
                        array = utils::split(array.front(), '.');
                        if(array.size() >= 1)
                            python_major_version() = Meta::fromStr<int>(array[0]);
                        if(array.size() > 1)
                            python_minor_version() = Meta::fromStr<int>(array[1]);
                    }
                    
                    python_uses_gpu() = has_gpu;
                    python_gpu_name() = physical_name;
                });
                
                TRex = _main.import("TRex");
                _locals = new pybind11::dict("model"_a="None");
                
                PythonIntegration::execute("import sys\nset_version(sys.version, False, '')");
                
                python_initialized() = true;
                python_initializing() = false;
                _initialize_promise->set_value(true);
                
            } catch(const UtilsException& ex) {
                Warning("Error while executing 'trex_init.py'. Content: %s", ex.what());
                fail(ex, __LINE__);
                return;
                
            } catch(py::error_already_set& e) {
                fail(e, __LINE__);
                e.restore();
                return;
            }
            catch (...) {
                python_init_error() = "Cannot initialize interpreter.";
                python_initializing() = false;
                python_initialized() = false;
                Except("Cannot initialize the python interpreter.");
                _initialize_promise->set_value(false);
                return;
            }
            
            bool printed = false;
            
            while (!_terminate) {
                while(!tasks.empty() && !_terminate) {
                    auto it = tasks.begin();
                    
                    if(!python_gpu_initialized() && !tasks.front()._can_run_before_init)
                    {
                        for (; it != tasks.end(); ++it) {
                            if(it->_can_run_before_init) {
                                break;
                            }
                        }
                        
                        if(it == tasks.end()) {
                            if(!printed) {
                                Warning("Cannot run python tasks while python is not initialized.");
                                printed = true;
                            }
                            
                            lock.unlock();
                            reinit();
                            lock.lock();
                            continue;
                        }
                    }
                    
                    if(it == tasks.end())
                        continue;
                    
                    printed = false;
                    
                    auto task = std::move(*it);
                    tasks.erase(it);
                    
                    lock.unlock();
                    try {
                        task._task();
                    } catch(py::error_already_set& e) {
                        Except("Python runtime exception: %s", e.what());
                        e.restore();
                    } catch( ... ) {
                        Debug("Caught one exception.");
                    }
                    lock.lock();
                }
                
                if(_terminate)
                    break;
                
                _update_condition.wait_for(lock, std::chrono::milliseconds(250));
            }
            
            try {
                track::numpy.release();
                track::TRex.release();
                track::_main.release();

                {
                    std::lock_guard<std::mutex> guard(module_mutex);
                    _modules.clear();
                }

                if (_locals) {
                    delete _locals;
                    _locals = nullptr;
                }

            } catch(py::error_already_set &e) {
                Debug("Python runtime error during clean-up: '%s'", e.what());
                e.restore();
            }
            
            try {
                //py::finalize_interpreter();
                Py_Finalize();
                
            } catch(py::error_already_set &e) {
                Debug("Python runtime error during clean-up: '%s'", e.what());
                e.restore();
            }
        });
    }
    
    std::tuple<std::vector<float>, std::vector<float>> PythonIntegration::probabilities(const std::vector<Image::Ptr>& images)
    {
        check_correct_thread_id();
        
        namespace py = pybind11;
        using namespace py::literals;
        
        static std::mutex mutex;
        static std::vector<float> values, indexes;
        
        std::lock_guard<std::mutex> guard(mutex);
        values.clear();
        indexes.clear();
        //values.resize(images.size() * FAST_SETTINGS(manual_identities).size());
        
        try {
            //check_module("learn_static");
            py::module module;
            
            {
                std::lock_guard<std::mutex> guard(module_mutex);
                if(!_modules.count("learn_static"))
                    SOFT_EXCEPTION("Cannot find 'learn_static'.");
                module = _modules.find("learn_static")->second;
            }
            
            module.attr("images") = images;
            module.def("receive", [&](py::array_t<float> x, py::array_t<float> idx) {
                std::vector<float> temporary;
                auto array = x.unchecked<2>();
                auto idxes = idx.unchecked<1>();
                
                Debug("Copying %d data", array.size());
                auto ptr = array.data(0,0);
                auto end = ptr + array.size();
                temporary.insert(temporary.end(), ptr, end);
                values = temporary;
                
                ptr = idxes.data(0);
                end = ptr + idxes.size();
                temporary.clear();
                temporary.insert(temporary.end(), ptr, end);
                indexes = temporary;
                x.release();
                idx.release();
                
            }, py::arg("x").noconvert(), py::arg("idx").noconvert());
        
            
            module.attr("predict")();
            
            module.attr("receive") = py::none();
            //std::string str = utils::read_file("probs.py");
            //py::exec(str, py::globals(), *_locals);
            
            //(*_locals)["images"] = nullptr;
            
        } catch (py::error_already_set &e) {
            Debug("Runtime error: '%s'", e.what());
            e.restore();
        }
        
        return {indexes, values};
    }
    
    std::future<bool> PythonIntegration::async_python_function(const std::function<bool ()> &fn, Flag flag, bool can_run_without_init)
    {
        PackagedTask task{std::packaged_task<bool()>(fn), can_run_without_init};
        auto future = task._task.get_future();
        if(flag != Flag::FORCE_ASYNC
           && std::this_thread::get_id() == _saved_id)
        {
            try {
                task._task();
            } catch (py::error_already_set &e) {
                Except("Python runtime error: '%s'", e.what());
                e.restore();
                SOFT_EXCEPTION(e.what());
            } catch(...) {
                Except("Random exception");
            }
        } else {
            std::unique_lock lock(_data_mutex);
            tasks.push_back(std::move(task));
            _update_condition.notify_one();
        }
        return future;
    }

std::shared_future<bool> PythonIntegration::ensure_started() {
    if(!_initialize_promise)
        _initialize_promise = std::make_unique<std::promise<bool>>();
    
    if(!_initialize_future.valid()) {
        _initialize_future = _initialize_promise->get_future().share();
    }
    
    if(!python_initialized() && !python_initializing() && !python_init_error().empty())
    {
        //async_python_function([]()->bool{return true;});
        
    } else if(!python_initialized()) {
        Warning("Python not yet initialized. Waiting...");
        PythonIntegration::instance();
    }
    
    return _initialize_future;
}

bool PythonIntegration::check_module(const std::string& name) {
    check_correct_thread_id();
    
    std::lock_guard<std::mutex> guard(module_mutex);
    bool result = false;
    
    auto c = utils::read_file(name+".py");
    if (c != contents[name] || CHECK_NONE(_modules[name])) {
        auto& mod = _modules[name];

        try {
            if (CHECK_NONE(mod)) {
                mod = _main.import(name.c_str());
            }
            mod.reload();
            Debug("Reloaded '%S.py'.", &name);
            result = true;
        }
        catch (pybind11::error_already_set & e) {
            Except("Python runtime exception while reloading %S: '%s'", &name, e.what());
            e.restore();
            mod.release();
        }

        contents[name] = c;
    }
    
    return result;
}

void PythonIntegration::run(const std::string& module_name, const std::string& function) {
    check_correct_thread_id();
    
    std::unique_lock<std::mutex> guard(module_mutex);

    try {
        py::handle module;
        
        if(!CHECK_NONE(_modules.at(module_name))) {
            if(function.empty())
                module = _modules.at(module_name);
            else
                module = _modules.at(module_name).attr(function.c_str());
        }
        
        if(!CHECK_NONE(module)) {
            guard.unlock();
            module();
        } else
            Except("Pointer of %S::%S is null.", &module_name, &function);
    }
    catch (pybind11::error_already_set & e) {
        e.restore();

        if (PyErr_Occurred()) {
            PyErr_PrintEx(0);
            PyErr_Clear(); // this will reset the error indicator so you can run Python code again
        }
        
        _modules.at(module_name).release();
        //_modules.at(module_name) = pybind11::none();
        SOFT_EXCEPTION("Python runtime exception while running %S.%S: '%s'", &module_name, &function, e.what());
    }
}

std::string PythonIntegration::run_retrieve_str(const std::string& module_name, const std::string& function)
{
    check_correct_thread_id();
    
    std::unique_lock<std::mutex> guard(module_mutex);
    try {
        if(!CHECK_NONE(_modules.at(module_name))) {
            auto result = _modules.at(module_name).attr(function.c_str());
            guard.unlock();
            return result().cast<std::string>();
        }
    }
    catch (pybind11::error_already_set & e) {
        e.restore();

        _modules.at(module_name).release();
        SOFT_EXCEPTION("Python runtime exception while running %S.%S: '%s'", &module_name, &function, e.what());
    }
    
    return "";
}

template<typename T>
T get_variable_internal(const std::string& name, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
    
    try {
        if(m.empty()) {
            if(_locals->contains(name.c_str())) {
                if(!CHECK_NONE(_locals->attr(name.c_str())))
                    return _locals->attr(name.c_str()).cast<T>();
            }
        } else {
            if(_modules.count(m)) {
                auto &mod = _modules[m];
                if(!CHECK_NONE(mod))
                    if(!CHECK_NONE(mod.attr(name.c_str())))
                        return mod.attr(name.c_str()).cast<T>();
            }
        }
    } catch(py::error_already_set & e) {
        Except("Python runtime error: '%s'", e.what());
        e.restore();
    }
    
    SOFT_EXCEPTION("Cannot find variable '%S' in '%S'.", &name, &m);
}

template<> TREX_EXPORT std::string PythonIntegration::get_variable(const std::string& name, const std::string& m) {
    return get_variable_internal<std::string>(name, m);
}
template<> TREX_EXPORT float PythonIntegration::get_variable(const std::string& name, const std::string& m) {
    return get_variable_internal<float>(name, m);
}

template<typename T>
void set_function_internal(const char* name_, T f, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
    
    if(m.empty()) {
        _main.def(name_, f);
    } else {
        if(_modules.count(m)) {
            auto &mod = _modules[m];
            if(!CHECK_NONE(mod)) {
                mod.def(name_, f);
                return;
            }
        }
        
        SOFT_EXCEPTION("Cannot define function '%s' in '%S' because the module does not exist.", name_, &m);
    }
}

void PythonIntegration::set_function(const char* name_, std::function<bool()> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<float()> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<void(float)> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<void(std::string)> f, const std::string &m) {
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<float>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uchar>, std::vector<std::string>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::unset_function(const char *name_, const std::string &m) {
    check_correct_thread_id();
    
    if(m.empty()) {
        if(!CHECK_NONE(_main.attr(name_))) {
            _main.attr(name_) = nullptr;
        } else
            Warning("Cannot find '%s' in _main.", name_);
    } else {
        if(_modules.count(m)) {
            auto &mod = _modules[m];
            if(!CHECK_NONE(mod)) {
                mod.attr(name_) = nullptr;
            }
        }
    }
}

bool PythonIntegration::is_none(const std::string& name, const std::string &m) {
    check_correct_thread_id();
    
    try {
        if(m.empty()) {
            if(CHECK_NONE(_main.attr(name.c_str()))) {
                return false;
            }
            
        } else {
            if(_modules.count(m)) {
                auto &mod = _modules[m];
                if(!CHECK_NONE(mod))
                    if(!CHECK_NONE(mod.attr(name.c_str())))
                        return false;
            }
        }
    } catch(py::error_already_set& e) {
        e.restore();
    }
    
    return true;
}

/*/else SOFT_EXCEPTION("Cannot find key '%S'.", &m); \*/
#define IMPL_VARIABLE_SHAPE(T) void PythonIntegration::set_variable(const std::string& name, const std::vector< T >& input, const std::string& m, const std::vector<size_t>& shapes, const std::vector<size_t>& strides) {\
    check_correct_thread_id(); \
    if(m.empty()) \
        (*_locals)[name.c_str()] = pybind11::array_t<T> ( !shapes.empty() ? shapes : std::vector<size_t>{ input.size() }, !strides.empty() ? strides : std::vector<size_t>{ sizeof(T) }, input.data() ); \
    else if(_modules.count(m)) { \
        auto &mod = _modules[m]; \
        if(mod.ptr() != nullptr) \
            mod.attr(name.c_str()) = pybind11::array_t<T> ( !shapes.empty() ? shapes : std::vector<size_t>{ input.size() }, !strides.empty() ? strides : std::vector<size_t>{ sizeof(T) }, input.data() ); \
    } \
}
#define IMPL_VARIABLE(T) void PythonIntegration::set_variable(const std::string& name, T input, const std::string& m) {\
    check_correct_thread_id(); \
    if(m.empty()) \
        (*_locals)[name.c_str()] = input; \
    else if(_modules.count(m)) { \
        auto &mod = _modules[m]; \
        if(mod.ptr() != nullptr) \
            mod.attr(name.c_str()) = input; \
    } \
}
IMPL_VARIABLE(const std::vector<Image::Ptr>&)
IMPL_VARIABLE_SHAPE(long_t)
IMPL_VARIABLE_SHAPE(float)
IMPL_VARIABLE(float)
IMPL_VARIABLE(long_t)
IMPL_VARIABLE(const std::string&)
IMPL_VARIABLE(bool)
IMPL_VARIABLE(uint64_t)

void PythonIntegration::set_variable(const std::string & name, const std::vector<std::string> & v, const std::string& m) {
    check_correct_thread_id();
    
    if(m.empty())
        (*_locals)[name.c_str()] = v;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = v;
    }
}

void PythonIntegration::check_correct_thread_id() {
    if(std::this_thread::get_id() != _saved_id) {
        auto name = get_thread_name();
        U_EXCEPTION("Executing python code in wrong thread ('%S').", &name);
    }
}

void PythonIntegration::execute(const std::string& cmd)  {
    check_correct_thread_id();
    
    try {
        pybind11::exec(cmd, pybind11::globals(), *_locals);
    }
    catch (pybind11::error_already_set & e) {
        e.restore();
        if (e.what()) {
            SOFT_EXCEPTION(e.what());
        }
        else {
            SOFT_EXCEPTION("Unknown error message from Python.");
        }
    }
}
void PythonIntegration::import_module(const std::string& name) {
    check_correct_thread_id();
    
    std::lock_guard<std::mutex> guard(module_mutex);
    contents[name] = utils::read_file(name + ".py");

    try {
        _modules[name] = _main.import(name.c_str());
    }
    catch (pybind11::error_already_set & e) {
        e.restore();

        _modules[name].release();
        SOFT_EXCEPTION(e.what());
    }
}

}
