#include <commons.pc.h>
#include <gtest/gtest.h>
#include <misc/ThreadPool.h>

using namespace cmn;

class MockThreadPool {
public:
    std::vector<std::unique_ptr<std::thread>> _threads;
    explicit MockThreadPool(uint32_t num_threads)
        : num_threads_(num_threads), enqueued_tasks_(0) 
    {
        for (uint32_t i = 0; i < num_threads_; ++i) {
            _threads.push_back(std::make_unique<std::thread>([this] {
                while (true) {
                    decltype(tasks_)::value_type task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return !tasks_.empty(); });
                        task = std::move(tasks_.front());
                        if (holds_alternative<std::monostate>(task)) {
                            return;
                        }
                        tasks_.pop_front();
                    }
                    std::get<std::packaged_task<void()>>(task)();
                }
            }));
        }
    }

    ~MockThreadPool() {
        for (uint32_t i = 0; i < num_threads_; ++i) {
            enqueue(nullptr);
        }
        for (auto& thread : _threads) {
            thread->join();
        }
    }

    uint32_t num_threads() const {
        return num_threads_;
    }

    std::future<void> enqueue(std::nullptr_t) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace_back(std::monostate{});
        }
        cv_.notify_one();
        std::promise<void> p;
        auto f = p.get_future();
        p.set_value();
        return f;
    }
    
    template<typename F, typename... Args>
    std::future<void> enqueue(F&& f, Args&&... args) {
        /*if (enqueued_tasks_ >= num_threads_) {
            throw std::runtime_error("Attempted to enqueue more tasks than allowed by the thread pool");
        }*/
        std::lock_guard<std::mutex> lock(mutex_);
        auto fn = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        std::packaged_task<void()> task(fn);
        auto fut = task.get_future();
        tasks_.emplace_back(std::move(task));
        ++enqueued_tasks_;
        cv_.notify_one();
        return fut;
    }

    size_t task_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return enqueued_tasks_;
    }

private:
    uint32_t num_threads_;
    size_t enqueued_tasks_;
    std::deque<std::variant<std::monostate, std::packaged_task<void()>>> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

TEST(DistributeIndexesTest, DoesNotExceedMaxThreads) {
    const int num_threads = 8;

    for(uint32_t item_num : {0, 1, 12, 9, 10, 100, 52, 53}) {
        std::vector<int> items(item_num);
        std::iota(items.begin(), items.end(), 0);
        
        // Test various numbers of requested threads, including more than available
        for (uint32_t requested_threads : {4, 1, 2, 3, 5, 8, 10, 20}) {
            MockThreadPool pool(num_threads); // Reset the pool for each iteration
            std::atomic<int> processed_items{0};
            std::mutex mutex;
            std::set<size_t> thread_indexes_used;
            
            auto fn = [&](int64_t i, std::vector<int>::iterator start, std::vector<int>::iterator end, size_t thread_index) {
                ASSERT_LT(thread_index, requested_threads);
                {
                    std::unique_lock guard(mutex);
                    thread_indexes_used.insert(thread_index);
                }
                
                for (auto it = start; it != end; ++it) {
                    ++processed_items;
                }
            };
            
            try {
                distribute_indexes(fn, pool, items.begin(), items.end(), requested_threads);
                
                // Verify the number of processed items is correct
                ASSERT_EQ(processed_items.load(), items.size());
                
                // Ensure that no more than the available number of threads are used
                //ASSERT_LE(pool.task_count(), num_threads);
                
                ASSERT_EQ(thread_indexes_used.size(), min(items.size(), requested_threads)) << "items: " << items.size();
                
                auto expected = std::set<size_t>();
                for (size_t i = 0; i < requested_threads && i < items.size(); ++i) {
                    expected.insert(i);
                }
                ASSERT_EQ(thread_indexes_used, expected);
                
            } catch (const std::runtime_error& e) {
                if (requested_threads > num_threads) {
                    // If requested threads are more than available, it's expected to throw an exception
                    ASSERT_STREQ(e.what(), "Attempted to enqueue more tasks than allowed by the thread pool");
                } else {
                    // If it's within the thread limit, rethrow the exception to fail the test
                    throw;
                }
            }
        }
    }
}

TEST(DistributeIndexesTest, ProcessesCorrectNumberOfItems) {
    const int num_threads = 4;
    MockThreadPool pool(num_threads);

    std::vector<int> items(100);
    std::iota(items.begin(), items.end(), 0);  // Fill with sequential values

    std::atomic<int> processed_items{0};

    auto fn = [&processed_items](int64_t i, std::vector<int>::iterator start, std::vector<int>::iterator end, size_t thread_index) {
        for (auto it = start; it != end; ++it) {
            ++processed_items;
        }
    };

    distribute_indexes(fn, pool, items.begin(), items.end(), num_threads);

    // Check that the correct number of items was processed
    ASSERT_EQ(processed_items.load(), items.size());
}

TEST(DistributeIndexesTest, HandlesSingleThread) {
    const int num_threads = 1;
    MockThreadPool pool(num_threads);

    std::vector<int> items(100);
    std::iota(items.begin(), items.end(), 0);

    std::atomic<int> processed_items{0};

    auto fn = [&processed_items](int64_t i, std::vector<int>::iterator start, std::vector<int>::iterator end, size_t thread_index) {
        for (auto it = start; it != end; ++it) {
            ++processed_items;
        }
    };

    distribute_indexes(fn, pool, items.begin(), items.end(), num_threads);

    ASSERT_EQ(processed_items.load(), items.size());
}

TEST(DistributeIndexesTest, HandlesUnevenDistribution) {
    const int num_threads = 3;
    MockThreadPool pool(num_threads);

    std::vector<int> items(103);  // Not evenly divisible by the number of threads
    std::iota(items.begin(), items.end(), 0);

    std::atomic<int> processed_items{0};

    auto fn = [&processed_items](int64_t i, std::vector<int>::iterator start, std::vector<int>::iterator end, size_t thread_index) {
        for (auto it = start; it != end; ++it) {
            ++processed_items;
        }
    };

    distribute_indexes(fn, pool, items.begin(), items.end(), num_threads);

    ASSERT_EQ(processed_items.load(), items.size());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
