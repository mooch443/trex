#pragma once
#include <commons.pc.h>
#include <list>
#include <misc/format.h>

template<typename T>
concept ManagedThreadConcept = requires(T a) {
    { a.loop() } -> std::same_as<void>;
    { a.terminate() } -> std::same_as<void>;
};

struct OnEndMethod {
    std::function<void()> lambda;
};

class PersistentCondition {
private:
    std::condition_variable cond_var;
    std::atomic<bool> notified{false};
    std::mutex mtx;

public:
    // Notify one waiting thread
    void notify() noexcept;

    // Wait for a notification
    void wait(std::unique_lock<std::mutex>& lock);
};

/**
 * @struct ManagedThreadWrapper
 *
 * @brief Structure for holding an instance of a managed thread and the std::thread it runs in.
 */
struct ManagedThreadWrapper;

struct ThreadGroupId {
    size_t index{0};
    
    explicit constexpr ThreadGroupId(size_t index) noexcept : index(index) {}
    constexpr ThreadGroupId() noexcept = default;
    
    constexpr bool valid() const noexcept { return index > 0; }
    constexpr auto operator<=>(const ThreadGroupId& other) const noexcept {
        return index <=> other.index;
    }
    std::string toStr() const;
    static std::string class_name() { return "ThreadGroupId"; }
};

/**
 * @struct ThreadGroup
 *
 * @brief Structure for grouping ManagedThreadWrappers and their associated on-end-callbacks.
 */
struct ThreadGroup {
    ThreadGroupId id;
    std::list<ManagedThreadWrapper> threads;
    std::list<OnEndMethod> onEndCallbacks;
    std::string name;
    std::atomic<bool> started{false};
    
    void terminate();
    void notify();
};

class ManagedThread {
    std::function<void(const ThreadGroupId&)> lambda;
    PersistentCondition variable;
    std::mutex mutex;
    std::atomic<bool> terminationSignal{false};
    std::promise<void> terminationProof;
public:
    ManagedThread(std::function<void(const ThreadGroupId&)> fn) : lambda(fn) {}
    ManagedThread(ManagedThread&& other)
        : lambda(std::move(other.lambda)),
          terminationSignal(other.terminationSignal.load()),
          terminationProof(std::move(other.terminationProof))
    {
        other.lambda = nullptr;
    }
    ~ManagedThread() {
        if(lambda)
            cmn::thread_print("Ending thread.");
    }
    
    void loop(const ThreadGroup& group, const ManagedThreadWrapper& thread);

    std::future<void> terminate() {
        std::future<void> future;
        
        {
            std::unique_lock guard(mutex);
            future = terminationProof.get_future();
            terminationSignal.store(true);
            variable.notify();
        }
        
        return future;
    }
    
    void notify() {
        variable.notify();
    }
};

struct ManagedThreadWrapper {
    std::unique_ptr<std::thread> t;
    ManagedThread m;
    std::string name;
};

/**
 * @class ThreadManager
 * 
 * @brief This class manages the creation, grouping and orderly termination of threads.
 * 
 * This class provides methods for registering groups of threads, adding threads or on-end-callbacks to groups,
 * starting all threads in a group, and orderly terminating all threads. The termination of threads happens
 * in the reverse order of the groups' registration.
 * 
 * This class follows the Singleton design pattern, ensuring only one instance exists. On deletion of the instance,
 * all threads are terminated orderly.
 */
class ThreadManager {
private:
    std::atomic<size_t> running_id{1u};
    std::map<ThreadGroupId, std::shared_ptr<ThreadGroup>> groups;
    mutable std::mutex mtx;

    /**
     * @brief Private constructor for Singleton implementation.
     */
    ThreadManager() {}

public:
    // Delete copy constructor and assignment operator
    ThreadManager(ThreadManager const&) = delete;
    void operator=(ThreadManager const&)  = delete;

    /**
     * @brief Returns the singleton instance of ThreadManager.
     * 
     * @return ThreadManager& Singleton instance of ThreadManager.
     */
    static ThreadManager& getInstance();

    /**
     * @brief Destructor for ThreadManager, terminates all threads upon destruction.
     */
    ~ThreadManager();

    /**
     * @brief Registers a new thread group with a given group id and name.
     * 
     * @param group The group id to be assigned to the new thread group.
     * @param name  The name to be assigned to the new thread group.
     */
    ThreadGroupId registerGroup(const std::string& name, cmn::source_location loc = cmn::source_location::current());

    /**
     * @brief Signals termination to all threads within a specific group and joins them.
     * 
     * @param group The group id of the group whose threads are to be terminated.
     */
    void terminateGroup(ThreadGroupId group);

    /**
     * @brief Adds a thread to a group. Thread is defined by the ManagedThread concept.
     * 
     * @param group          The group id to which the thread is to be added.
     * @param name           The name to be assigned to the thread.
     * @param managedThread  The managed thread to be added.
     */
    void addThread(ThreadGroupId group, const std::string& name, ManagedThread&& managedThread);

    /**
     * @brief Adds an on-end-callback to a group.
     * 
     * @param group         The group id to which the on-end-callback is to be added.
     * @param onEndMethod   The on-end-callback to be added.
     */
    void addOnEndCallback(ThreadGroupId group, OnEndMethod onEndMethod);

    /**
     * @brief Starts all threads in a group.
     * 
     * @param group The group id of the group whose threads are to be started.
     */
    void startGroup(ThreadGroupId group);
    //bool groupStarted(int group) const;
    void notify(ThreadGroupId group);

    /**
     * @brief Signals termination to all threads and joins them in reverse order of group registration.
     */
    void terminate();

    /**
     * @brief Prints the tree of registered threads and their states.
     */
    void printThreadTree();
    
private:
    void printThreadTree(std::unique_lock<std::mutex>&);
};

#define REGISTER_THREAD_GROUP(NAME) ThreadManager::getInstance().registerGroup(NAME, cmn::source_location::current())
