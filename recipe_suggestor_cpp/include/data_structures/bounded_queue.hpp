#ifndef BOUNDED_QUEUE_HPP
#define BOUNDED_QUEUE_HPP

#include <iostream>
#include <queue>
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
    const std::size_t max_size = constants::queue_max_size;

public:
    // I prevent the compiler from using it for implicit conversions and copy-initialization.
    explicit BoundedQueue() {}

    void push(std::optional<T> item) {
        std::unique_lock<std::mutex> lock(mx);
        cv_not_full.wait(lock, [this] { return q.size() < max_size; }); // if queue is full, wait
        queue.push(std::move(item));
        cv_not_empty.notify_one(); // notify that a new item is inserted (stop waiting if pop was waiting)
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mx);
        cv_not_empty.wait(lock, [this] { return !q.empty(); });
        std::optional<T> item = std::move(queue.front());
        queue.pop();
        cv_not_full.notify_one();
        return item;
    }

};

#endif // BOUNDED_QUEUE_HPP