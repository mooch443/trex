#include "ThreadManager.h"
#include <misc/format.h>
#include <file/Path.h>

using namespace cmn;

void PersistentCondition::notify() noexcept {
    std::unique_lock g(mtx);
    notified = true;
    cond_var.notify_one();
}

void PersistentCondition::wait(std::unique_lock<std::mutex> &lock, std::function<bool()>&& lambda) {
    bool expected = true;
    if(not notified.compare_exchange_strong(expected, false)) {
        lock.unlock();
        
        {
            std::unique_lock g(mtx);
            if(notified) {
                notified = false;
                g.unlock();
                lock.lock();
                return;
            }
            
            if(lambda)
                cond_var.wait(g, lambda);
            else
                cond_var.wait(g);
        }
        
        lock.lock();
    }
}

std::string ThreadGroupId::toStr() const {
    return "ThreadGroup<"+Meta::toStr(index)+">";
}

ThreadManager& ThreadManager::getInstance() {
    static ThreadManager *instance = new ThreadManager;
    return *instance;
}

ThreadManager::~ThreadManager() {
    terminate();
}

void ThreadManager::addThread(ThreadGroupId group, const std::string& name, ManagedThread&& managedThread) {
    std::unique_lock lock(mtx);
    groups.at(group)->threads.emplace_back(ManagedThreadWrapper{
        .t = nullptr,
        .m = std::move(managedThread),
        .name = name
    });
    cmn::thread_print("Added thread ", group, "::", name.c_str());
    //printThreadTree(lock);
}

ThreadGroupId ThreadManager::registerGroup(const std::string& name, source_location loc) {
    std::unique_lock lock(mtx);
    if(name.empty())
        throw InvalidArgumentException("Name cannot be empty.");
    
    auto group = ThreadGroupId{running_id++};
    auto& g = groups[group];
    if (g && g->started) {
        FormatExcept("Thread ", g->name, " was already started.");
        g->terminate();
    }
    
    g = std::make_shared<ThreadGroup>();
    g->id = group;
    g->name = name;
    g->threads.clear();
    g->onEndCallbacks.clear();
    g->started = false;
    thread_print("Registered group ", group, "::", name.c_str(), " from ", file::Path(loc.file_name()).filename(),":", loc.line());
    
    //printThreadTree(lock);
    return g->id;
}

void ThreadManager::addOnEndCallback(ThreadGroupId group, OnEndMethod onEndMethod) {
    std::lock_guard<std::mutex> lock(mtx);
    groups.at(group)->onEndCallbacks.push_back(onEndMethod);
    thread_print("Added on end callback to group ", group, " (", groups.at(group)->name,")");
}

void ManagedThread::loop(const ThreadGroup &group, const ManagedThreadWrapper& thread) {
    set_thread_name(utils::ShortenText(group.name+"::"+thread.name, 25));
    thread_print("Starting loop ", group.name);
    terminationProof = {};
    terminationSignal = false;
    try {
        std::unique_lock guard(mutex);
        while (!terminationSignal.load()) {
            //thread_print("TM Loop ", group.name);
            guard.unlock();
            try {
                lambda(group.id);
            } catch(...) {
                FormatExcept("TM Exception in thread group ", group.name);
            }
            guard.lock();
            if(not terminationSignal)
                variable.wait(guard);
        }
    } catch(...) {
        FormatExcept("Terminating thread (",thread.name,") from group(",group.name,") due to an exception.");
        terminationProof.set_exception(std::current_exception());
        throw;
    }
    terminationProof.set_value();
}

void ThreadGroup::terminate() {
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

void ThreadGroup::notify() {
    for(auto &t : threads) {
        t.m.notify();
    }
}

void ThreadManager::notify(ThreadGroupId group) {
    std::shared_ptr<ThreadGroup> g;
    {
        std::unique_lock lock(mtx);
        if(groups.find(group) != groups.end()) {
            g = groups.at(group);
        } else {
            thread_print("Group ", group, " does not exist");
        }
    }
    
    if(g) {
        //thread_print("TM Notify ", g->name);
        g->notify();
    }
}

void ThreadManager::terminateGroup(ThreadGroupId group) {
    std::shared_ptr<ThreadGroup> g;
    
    {
        std::unique_lock lock(mtx);
        if(groups.find(group) != groups.end()) {
            g = groups.at(group);
        } else {
            thread_print("Group ", group, " does not exist");
        }
    }
    
    if(g) {
        g->terminate();
        //printThreadTree();
    }
}

void ThreadManager::startGroup(ThreadGroupId group) {
    std::unique_lock lock(mtx);
    if(not groups.contains(group)) {
        throw InvalidArgumentException("Group ", group, " cannot has not been registered.");
    }
    
    auto& g = groups.at(group);
    
    //! only start the group if it has not been started yet
    bool expected = false;
    if(g->started.compare_exchange_strong(expected, true)) {
        for(auto& wrapper : g->threads) {
            if(not wrapper.t) {
                wrapper.t = std::make_unique<std::thread>([&wrapper, &g] {
                    wrapper.m.loop(*g, wrapper);
                });
                thread_print("Started thread in group ", group);
            } else {
                thread_print("Thread already started in group ", group);
            }
        }
        
        //printThreadTree(lock);
    }
}

/*bool ThreadManager::groupStarted(int group) const {
    std::lock_guard<std::mutex> lock(mtx);
    if(groups.contains(group)) {
        return groups.at(group).started;
    }
    return false;
}*/

void ThreadManager::terminate() {
    std::unique_lock lock(mtx);
    // Terminating threads and calling callbacks in reverse order (reverse dependencies)
    for(auto it = groups.rbegin(); it != groups.rend(); ++it) {
        it->second->terminate();
    }
    printThreadTree(lock);
}

void ThreadManager::printThreadTree() {
    std::unique_lock lock(mtx);
    printThreadTree(lock);
}

void ThreadManager::printThreadTree(std::unique_lock<std::mutex> &) {
    thread_print("[ThreadManager::print]");
    for(auto& group : groups) {
        thread_print("  Group ", group.first, "::", group.second->name.c_str());
        for(auto& wrapper : group.second->threads) {
            if(group.second->started && wrapper.t)
                thread_print("   ", fmt::clr<FormatColor::GREEN>("+"), " Thread ", wrapper.name.c_str());
            else
                thread_print("   ", fmt::clr<FormatColor::RED>("-"), " Thread ", wrapper.name.c_str());
        }
    }
}
