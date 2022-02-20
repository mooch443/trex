#include "CheckUpdates.h"
#include <misc/GlobalSettings.h>
#include <misc/metastring.h>
#include <python/GPURecognition.h>
#include <tracker/gui/gui.h>
#include <gui/WorkProgress.h>
#include <tracker/misc/default_config.h>

#if WIN32
#include <shellapi.h>
#endif

namespace track {
namespace CheckUpdates {

static void update_loop();
static void check_thread();

std::string _newest_version = "";
std::string _last_error = "";

std::unique_ptr<std::thread> _thread;
std::mutex _mutex;
std::condition_variable _variable;
std::atomic_bool _good_time = false;
std::atomic_bool _terminate = false;
std::atomic_bool _last_check_success = true;

const std::string& newest_version() {
    return _newest_version;
}

const std::string& last_error() {
    return _last_error;
}

std::string current_version() {
    auto v = SETTING(version).value<std::string>();
    if(v.empty() || v[0] != 'v')
        return "0";
    return utils::split(utils::split(v, 'v').back(), '-').front();
}

std::string last_asked_version() {
    auto v = SETTING(app_last_update_version).value<std::string>();
    return v;
}

void cleanup() {
    std::unique_lock guard(_mutex);
    if(_thread) {
        guard.unlock();
        
        _terminate = true;
        _variable.notify_all();
        
        print("Cleaning up...");
        _thread->join();
        guard.lock();
        _thread = nullptr;
    }
}

void init() {
    std::string contents;
    try {
        if (!file::Path("update_check").exists()) {
            if (!SETTING(quiet))
                Debug("Initial start, no update_check file exists.");
            return;
        }
        contents = utils::read_file("update_check");
        auto array = utils::split(contents, '\n');
        if (array.size() != 3) {
            file::Path("update_check").delete_file();
            U_EXCEPTION("Array does not have the right amount of elements ('%S', %lu). Deleting file.", &contents, array.size());
        }
        SETTING(app_last_update_check) = Meta::fromStr<uint64_t>(array.front());
        SETTING(app_check_for_updates) = Meta::fromStr<default_config::app_update_check_t::Class>(array.at(1));
        SETTING(app_last_update_version) = array.back();
        
    } catch(const UtilsException& ex) {
        Except("Utils Exception: '%s'", ex.what());
    } catch(const std::exception& ex) {
        Except("Exception: '%s'", ex.what());
    } catch(...) {
        Warning("Illegal content, or parsing failed for app_last_update_check: '%S'", &contents);
    }
}

void check_thread() {
    std::lock_guard guard(_mutex);
    if(!_thread) {
        if(user_has_been_asked() && automatically_check()) {
            _thread = std::make_unique<std::thread>([](){
                set_thread_name("CheckUpdate::thread");
                
                std::unique_lock guard(_mutex);
                while(!_terminate) {
                    // wait for something to happen
                    // e.g. the main program giving us a hint that this might be a good time to check
                    _variable.wait_for(guard, std::chrono::seconds(10));
                    
                    if(_good_time) {
                        guard.unlock();
                        update_loop();
                        guard.lock();
                        
                        _good_time = false;
                    }
                }
            });
        }
    }
}

void this_is_a_good_time() {
    check_thread();
    
    _good_time = true;
    _variable.notify_one();
}

bool user_has_been_asked() {
    using namespace default_config;
    return SETTING(app_check_for_updates).value<app_update_check_t::Class>() != app_update_check_t::none;
}

bool automatically_check() {
    using namespace default_config;
    return SETTING(app_check_for_updates).value<app_update_check_t::Class>() == app_update_check_t::automatically;
}

void display_update_dialog() {
    if(!GUI::instance() && !GUI_SETTINGS(nowindow)) {
        Debug("Newer version (%S) available for download. Visit https://trex.run/docs/update.html for instructions on how to update.", &CheckUpdates::newest_version());
        return;
    }
    
    GUI::instance()->gui().dialog([](gui::Dialog::Result r) {
        if(r == gui::Dialog::OKAY) {
            auto website = "https://trex.run/docs/update.html";
#if __linux__
            auto pid = fork();
            if (pid == 0) {
                execl("/usr/bin/xdg-open", "xdg-open", website, (char *)0);
                exit(0);
            }
#elif __APPLE__
            auto pid = fork();
            if (pid == 0) {
                execl("/usr/bin/open", "open", website, (char *)0);
                exit(0);
            }
#elif defined(WIN32)
            ShellExecute(
                NULL,
                "open",
                website,
                NULL,
                NULL,
                SW_SHOWNORMAL
            );
#endif
        }
        
    }, "The newest available version is <nr>"+CheckUpdates::newest_version()+"</nr>. You have version <str>"+CheckUpdates::current_version()+"</str>. Do you wish to see update instructions?", "Update available", "Yes", "No");
}

void write_version_file() {
    // write changed date to file 'update_check' in the resource folder
    std::string str = SETTING(app_last_update_check).get().valueString()+"\n"+SETTING(app_check_for_updates).get().valueString()+"\n"+SETTING(app_last_update_version).get().valueString();
    auto f = fopen("update_check", "wb");
    if (f) {
        fwrite(str.c_str(), sizeof(char), str.length(), f);
        fclose(f);
    }
    else
        Except("Cannot open update_check file for writing to save the settings (maybe no permissions in app folder?).");
}

void update_loop() {
    using namespace default_config;
    
    if(SETTING(app_check_for_updates).value<app_update_check_t::Class>() == app_update_check_t::automatically)
    {
        using namespace std::chrono;
        using namespace std::chrono_literals;
        
        auto timestamp = SETTING(app_last_update_check).value<uint64_t>();
        
        auto tp = microseconds(timestamp);
        auto now = system_clock::now();
        auto dt = (now - tp).time_since_epoch();
        
        auto str = Meta::toStr(DurationUS{(uint64_t)dt.count()});
        
        static const auto short_update_time = 24s * 7;
        static const auto long_update_time = 24h * 7;
        
        if(   ( _last_check_success && dt >= long_update_time)
           || (!_last_check_success && dt >= short_update_time))
        {
            if(_last_check_success)
                print("[CHECK_UPDATES] It has been a week. Let us check for updates...");
            else
                Debug("[CHECK_UPDATES] Trying again after %S...", &str);
            
            SETTING(app_last_update_check) = (uint64_t)duration_cast<microseconds>( now.time_since_epoch() ).count();
            
            try {
                auto status = perform(false).get();
                if(status != VersionStatus::NONE) {
                    SETTING(app_last_update_version) = std::string(newest_version());
                    write_version_file();
                    _last_check_success = true;
                    
                    if(status == VersionStatus::NEWEST) {
                        Debug("[CHECK_UPDATES] Already have the newest version (%S).", &newest_version());
                    } else if(status == VersionStatus::ALREADY_ASKED) {
                        Debug("[CHECK_UPDATES] There is a new version available (%S), but you have already acknowledged this. If you want to see instructions again, please go to the top-right menu -> check updates.");
                    } else {
                        display_update_dialog();
                    }
                    
                } else {
                    U_EXCEPTION("Status suggested the check failed.");
                }
                
            } catch(...) {
                Error("There was an error checking for the newest version:\n\n%s\n\nPlease check your internet connection and try again. This also happens if you are checking for versions too often, or if GitHub changed their API (in which case you should probably update).", last_error().c_str());
            }
        }
    }
}

std::future<VersionStatus> perform(bool manually_triggered) {
    auto promise = std::make_shared<std::promise<VersionStatus>>();
    auto future = promise->get_future();
    
    using py = PythonIntegration;
    if(manually_triggered && !GUI_SETTINGS(nowindow) && GUI::instance()) {
        GUI::work().add_queue("Initializing python...", [](){
            py::ensure_started().wait();
        });
        
    } else {
        if(GUI::instance() && GUI::work().is_this_in_queue()) {
            GUI::work().set_item("Initializing python...");
            py::ensure_started().get();
            GUI::work().set_item("");
        } else {
            py::ensure_started();
        }
    }
    
    auto fn = [ptr = std::move(promise)](std::string v) {
        if(v.empty()) {
            ptr->set_value(VersionStatus::NONE);
            return;
        }
        
        auto my_sub_versions = current_version();
        _newest_version = v;
        
        if(v == my_sub_versions) {
            ptr->set_value(VersionStatus::NEWEST);
            
        } else {
            if(v == last_asked_version())
                ptr->set_value(VersionStatus::ALREADY_ASKED); // user has already been asked for this version
            else
                ptr->set_value(VersionStatus::OLD);
        }
    };
    
    py::async_python_function([fn]() -> bool {
        py::set_function("retrieve_version", fn);
        
        try {
            py::execute("import requests");
            py::execute("retrieve_version(sorted([o['name'].split(':')[0].split('v')[1] for o in requests.get('https://api.github.com/repos/mooch443/trex/releases', headers={'accept':'application/vnd.github.v3.full+json'}).json() if 'v' in o['name']])[-1])");
        } catch(const SoftException& ex) {
            std::string line = ex.what();
            auto array = utils::split(line, '\n');
            for(auto &l : array)
                l = escape_html(l);
            
            if(array.size() > 3) {
                array.erase(array.begin() + 1, array.begin() + (array.size() - 2));
                array.insert(array.begin() + 1, std::string("<i>see terminal for full stack...</i>"));
            }
            
            line.clear();
            for(auto &l : array) {
                if(l.empty())
                    continue;
                
                if(!line.empty())
                    line += "\n";
                line += l;
            }
            
            _last_error = line;
            
            Error("Failed to retrieve github status to determine what the current version is. Assuming current version is the most up-to-date one.");
            fn("");
        }
        
        py::unset_function("retrieve_version");
        return true;
    }, py::Flag::DEFAULT, true);
    
    return future;
}

}
}
