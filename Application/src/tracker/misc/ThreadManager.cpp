#include "ThreadManager.h"
#include <misc/format.h>

using namespace cmn;

ThreadManager& ThreadManager::getInstance() {
    static ThreadManager *instance = new ThreadManager;
    return *instance;
}

ThreadManager::~ThreadManager() {
    terminate();
}

void ThreadManager::addThread(int group, const std::string& name, ManagedThread&& managedThread) {
    std::lock_guard<std::mutex> lock(mtx);
    ManagedThreadWrapper wrapper{nullptr, std::move(managedThread)};
    groups.at(group).threads.push_back(std::move(wrapper));
    cmn::thread_print("Added thread", name, "to group", group);
}

void ThreadManager::registerGroup(int group, const std::string& name, source_location loc) {
    std::lock_guard<std::mutex> lock(mtx);
    if(name.empty())
        throw std::invalid_argument("Name cannot be empty.");
    groups[group].name = name;
    thread_print("Registered group ", group, " with name ", name, " from ", loc.file_name(),":", loc.line());
}

void ThreadManager::addOnEndCallback(int group, OnEndMethod onEndMethod) {
    std::lock_guard<std::mutex> lock(mtx);
    groups.at(group).onEndCallbacks.push_back(onEndMethod);
    thread_print("Added on end callback to group ", group, " (", groups.at(group).name,")");
}

void ThreadManager::ThreadGroup::terminate() {
    //! TODO: maybe exchange_strong?
    if(not started)
        return;
    
    started = false;
    thread_print("Terminating group ", name);
    
    for(auto& wrapper : threads) {
        if(wrapper.t) {
            auto f = wrapper.m.terminate();
            if(f.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
                // if the thread is still not terminated after 1 second, print out a message
                thread_print("Thread in group ", name, " is taking a long time to terminate");
            }

            f.get();
            wrapper.t->join();
            wrapper.t = nullptr;
        }
    }
    for(auto& callback : onEndCallbacks) {
        callback.lambda();
    }
}

void ThreadManager::ThreadGroup::notify() {
    for(auto &t : threads)
        t.m.notify();
}

void ThreadManager::notify(int group) {
    std::lock_guard<std::mutex> lock(mtx);
    if(groups.find(group) != groups.end()) {
        groups.at(group).notify();
    } else {
        thread_print("Group ", group, " does not exist");
    }
}

void ThreadManager::terminateGroup(int group) {
    std::lock_guard<std::mutex> lock(mtx);
    if(groups.find(group) != groups.end()) {
        groups.at(group).terminate();
    } else {
        thread_print("Group ", group, " does not exist");
    }
}

void ThreadManager::startGroup(int group) {
    std::lock_guard<std::mutex> lock(mtx);
    for(auto& wrapper : groups.at(group).threads) {
        if(not wrapper.t) {
            wrapper.t = std::make_unique<std::thread>([&wrapper] {
                wrapper.m.loop();
            });
            thread_print("Started thread in group ", group);
        } else {
            thread_print("Thread already started in group ", group);
        }
    }
    groups.at(group).started = true;
}

bool ThreadManager::groupStarted(int group) const {
    std::lock_guard<std::mutex> lock(mtx);
    if(groups.contains(group)) {
        return groups.at(group).started;
    }
    return false;
}

void ThreadManager::terminate() {
    std::lock_guard<std::mutex> lock(mtx);
    // Terminating threads and calling callbacks in reverse order (reverse dependencies)
    for(auto it = groups.rbegin(); it != groups.rend(); ++it) {
        it->second.terminate();
    }
}

void ThreadManager::printThreadTree() {
    std::lock_guard<std::mutex> lock(mtx);
    for(auto& group : groups) {
        thread_print("Group ", group.first, " - ", group.second.name);
        for(auto& wrapper : group.second.threads) {
            thread_print(" - Thread ", get_thread_name(wrapper.t->native_handle()));
        }
    }
}
