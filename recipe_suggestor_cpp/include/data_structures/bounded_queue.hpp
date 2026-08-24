#ifndef BOUNDED_QUEUE_HPP
#define BOUNDED_QUEUE_HPP

#include <iostream>
#include <queue>
#include <sys/types.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <vector>
#include <string>
#include "constants.hpp"

template <typename T>
class BoundedQueue {

private:
    std::queue<std::optional<T>> q;
    std::mutex mx;

    // Variables that are alerted when elements are pushed or popped to monitor queue status
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    const u_int16_t max_size;
    bool closed = false;

public:
    // I prevent the compiler from using it for implicit conversions and copy-initialization.
    explicit BoundedQueue(u_int16_t size) : max_size(size) {}

    void push(std::optional<T> item) {
        std::unique_lock<std::mutex> lock(mx);
        // if queue is full, wait -- unless we are shutting down
        cv_not_full.wait(lock, [this] { return q.size() < max_size || closed; });
        if (closed) return; // dropped on purpose: nobody is going to consume it
        q.push(std::move(item));
        cv_not_empty.notify_one(); // notify that a new item is inserted (stop waiting if pop was waiting)
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mx);
        cv_not_empty.wait(lock, [this] { return !q.empty() || closed; });
        if (q.empty()) return std::nullopt; // closed and drained: same sentinel as end-of-stream
        std::optional<T> item = std::move(q.front());
        q.pop();
        cv_not_full.notify_one();
        return item;
    }

    // Unblocks every waiting producer and consumer. After this, pop() returns
    // std::nullopt once the queue is drained and push() discards its argument.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mx);
            closed = true;
        }
        cv_not_empty.notify_all();
        cv_not_full.notify_all();
    }

    // Hands back whatever is still queued so the owner can free it. The queue
    // holds raw pointers in this project, so it cannot clean up after itself.
    std::vector<T> drain() {
        std::lock_guard<std::mutex> lock(mx);
        std::vector<T> leftovers;
        while (!q.empty()) {
            if (q.front()) leftovers.push_back(*q.front());
            q.pop();
        }
        cv_not_full.notify_all();
        return leftovers;
    }

};

#endif // BOUNDED_QUEUE_HPP
