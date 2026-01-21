#ifndef CIRCULAR_LIST_HPP
#define CIRCULAR_LIST_HPP
#pragma once
#include <cstddef>
#include <initializer_list>
#include <string>
#include <atomic>
#include <iterator>
#include <thread>

namespace cust {
template<typename T>
class Node {
public:
    T value;
    Node* next;
    Node* prev;
    Node(const T& v) : value(v), next(nullptr), prev(nullptr) {}
};

template<typename T>
class CircularList {
private:
    Node<T>* head;
    std::atomic<std::size_t> size_;
    mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;

    void lock() const {
        while (lock_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    void unlock() const {
        lock_.clear(std::memory_order_release);
    }

    struct LockGuard { // RAII
        const CircularList* list;
        LockGuard(const CircularList* l) : list(l) { l->lock(); }
        ~LockGuard() { list->unlock(); }
    };

    void clear_unlocked() {
        if (!this->head) return;
        Node<T>* curr = this->head->next;
        while (curr != this->head) {
            Node<T>* next = curr->next;
            delete curr;
            curr = next;
        }
        delete this->head;
        this->head = nullptr;
        size_ = 0;
    }

    void add_unlocked(const T& value) {
        Node<T>* node = new Node<T>(value);
        if (this->head == nullptr) {
            this->head = node;
            this->head->next = this->head;
            this->head->prev = this->head;
        } else {
            Node<T>* tail = this->head->prev;
            tail->next = node;
            node->prev = tail;
            node->next = this->head;
            this->head->prev = node;
        }
        ++this->size_;
    }
    
    Node<T>* find_unlocked(const T& value) {
        if (!this->head) return nullptr;
        Node<T>* curr = this->head;
        do {
            if (curr->value == value) return curr;
            curr = curr->next;
        } while (curr != this->head);
        return nullptr;
    }

public:
    CircularList() : head(nullptr), size_(0) {}
    
    CircularList(const CircularList& other) : head(nullptr), size_(0) {
    LockGuard lock(other);
    if (!other.head) return;
    
    Node<T>* curr = other.head;
    do {
        add_unlocked(curr->value);
        curr = curr->next;
    } while (curr != other.head);
}

    ~CircularList() { clear(); }
    
    void add(const T& value) {
        LockGuard lock(this);
        Node<T>* node = new Node<T>(value);
        if (this->head == nullptr) {
            this->head = node;
            this->head->next = this->head;
            this->head->prev = this->head;
        } else {
            Node<T>* tail = this->head->prev;
            tail->next = node;
            node->prev = tail;
            node->next = this->head;
            this->head->prev = node;
        }
        ++this->size_;
    }
    
    Node<T>* find(const T& value) {
        LockGuard lock(this);
        if (!this->head) return nullptr;
        Node<T>* curr = this->head;
        do {
            if (curr->value == value) return curr;
            curr = curr->next;
        } while (curr != this->head);
        return nullptr;
    }
    
    const Node<T>* find(const T& value) const {
        LockGuard lock(this);
        if (!this->head) return nullptr;
        Node<T>* curr = this->head;
        do {
            if (curr->value == value) return curr;
            curr = curr->next;
        } while (curr != this->head);
        return nullptr;
    }
    
    template<typename... Args>
    void add_all(Args&&... values) {
        LockGuard lock(this);
        (add_unlocked(std::forward<Args>(values)), ...);
    }
    
    void clear() {
        LockGuard lock(this);
        if (!this->head) return;
        Node<T>* curr = this->head->next;
        while (curr != this->head) {
            Node<T>* next = curr->next;
            delete curr;
            curr = next;
        }
        delete this->head;
        this->head = nullptr;
        size_ = 0;
    }
    
    T* next(const T& current) {
        LockGuard lock(this);
        Node<T>* node = find_unlocked(current);
        return node ? &(node->next->value) : nullptr;
    }
    
    T* previous(const T& current) {
        LockGuard lock(this);
        Node<T>* node = find_unlocked(current);
        return node ? &(node->prev->value) : nullptr;
    }
    
    std::string to_string() {
        LockGuard lock(this);
        std::string return_string;
        if (!head) {
            return "[]";
        }
        Node<T>* temp = head;
        return_string.append("[");
        do {
            return_string.append(std::to_string(temp->value));
            temp = temp->next;
            if (temp != head) return_string.append(", ");
        } while (temp != head);
        return_string.append("]");
        return return_string;
    }
    
    void rotate(int n) {
        LockGuard lock(this);
        if (!this->head || size_ == 0 || n % this->size_ == 0) return;
        n = n % size_;
        if (n < 0) n += size_;
        Node<T>* temp = this->head;
        for (int i = 0; i < size_ - n; i++) temp = temp->next;
        head = temp;
    }
    
    void anti_rotate(int n) {
        LockGuard lock(this);
        if (!this->head || size_ == 0 || n % this->size_ == 0) return;
        n = n % size_;
        if (n < 0) n += size_;
        Node<T>* temp = this->head;
        for (int i = 0; i < n; i++) temp = temp->prev;
        this->head = temp;
    }
    
    void remove(const T& value) {
        LockGuard lock(this);
        Node<T>* node = find_unlocked(value);
        if (node == nullptr) return;
        if (this->size_ == 1) {
            delete node;
            this->head = nullptr;
            this->size_ = 0;
            return;
        }
        if (node == this->head) {
            this->head = node->next;
        }
        node->prev->next = node->next;
        node->next->prev = node->prev;
        delete node;
        this->size_--;
    }
    
    void remove_all() {
        clear();
    }
    
    std::size_t size() const {
        return size_.load();
    }

    // Forward iterator to support range-based for loops: for (auto &el : circularList)
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        iterator(Node<T>* cur = nullptr, Node<T>* head = nullptr) : cur(cur), head(head) {}

        reference operator*() const { return cur->value; }
        pointer operator->() const { return &(cur->value); }

        iterator& operator++() {
            if (!cur) return *this;
            // If next is the head, we've completed the loop and set to end (nullptr)
            if (cur->next == head) cur = nullptr;
            else cur = cur->next;
            return *this;
        }

        iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }

        bool operator==(const iterator& other) const { return cur == other.cur; }
        bool operator!=(const iterator& other) const { return cur != other.cur; }

    private:
        Node<T>* cur;
        Node<T>* head; // head pointer to detect loop end
    };

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator(const Node<T>* cur = nullptr, const Node<T>* head = nullptr) : cur(cur), head(head) {}

        reference operator*() const { return cur->value; }
        pointer operator->() const { return &(cur->value); }

        const_iterator& operator++() {
            if (!cur) return *this;
            if (cur->next == head) cur = nullptr;
            else cur = cur->next;
            return *this;
        }

        const_iterator operator++(int) { const_iterator tmp = *this; ++(*this); return tmp; }

        bool operator==(const const_iterator& other) const { return cur == other.cur; }
        bool operator!=(const const_iterator& other) const { return cur != other.cur; }

    private:
        const Node<T>* cur;
        const Node<T>* head;
    };

    iterator begin() {
        LockGuard lock(this);
        if (!head) return iterator(nullptr, nullptr);
        return iterator(head, head);
    }

    iterator end() {
        return iterator(nullptr, nullptr);
    }

    const_iterator begin() const {
        LockGuard lock(this);
        if (!head) return const_iterator(nullptr, nullptr);
        return const_iterator(head, head);
    }

    const_iterator end() const {
        return const_iterator(nullptr, nullptr);
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // Copy assignment operator
    CircularList& operator=(const CircularList& other) {
        if (this == &other) return *this;
        
        // Lock both carefully avoiding deadlock by locking by address order
        const CircularList* first = this;
        const CircularList* second = &other;
        if (first > second) std::swap(first, second);
        
        first->lock();
        second->lock();
        
        this->clear_unlocked();
        
        if (other.head) {
            Node<T>* curr = other.head;
            do {
                this->add_unlocked(curr->value);
                curr = curr->next;
            } while (curr != other.head);
        }
        
        second->unlock();
        first->unlock();
        
        return *this;
    }

    CircularList(CircularList&& other) noexcept : head(nullptr), size_(0) {
        LockGuard lock(&other);
        head = other.head;
        size_ = other.size_.load();
        other.head = nullptr;
        other.size_ = 0;
    }

    // Move assignment operator
    CircularList& operator=(CircularList&& other) noexcept {
        if (this == &other) return *this;
        
        const CircularList* first = this;
        const CircularList* second = &other;
        if (first > second) std::swap(first, second);
        
        first->lock();
        second->lock();
        
        this->clear_unlocked();
        head = other.head;
        size_ = other.size_.load();
        other.head = nullptr;
        other.size_ = 0;
        
        second->unlock();
        first->unlock();
        
        return *this;
    }
};

template<typename T>
bool operator!=(const Node<T>& lhs, const Node<T>& rhs) { 
    return !(lhs.value == rhs.value); 
}

template<typename T>
bool operator==(const Node<T>& lhs, const Node<T>& rhs) { 
    return lhs.value == rhs.value; 
}
}
#endif // CIRCULAR_LIST_HPP