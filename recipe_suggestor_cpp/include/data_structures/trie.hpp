#ifndef TRIE_HPP
#define TRIE_HPP

#include "constants.hpp"
#include "types.hpp"
#include <sys/types.h>
#include <unordered_map>
#include <vector>
#include <optional>
#include <memory>

class Node {
public:
    std::unordered_map<types::ConsumableID, std::unique_ptr<Node>> children;
    std::optional<types::ItemID> item;
    
    // Default destructor is sufficient as unique_ptr handles cleanup
    ~Node() = default;
};

class SuggestionTrie {
private:
    std::unique_ptr<Node> root;

public:
    SuggestionTrie();
    ~SuggestionTrie();

    void insert(const std::vector<types::ConsumableID>& recipe, const types::ItemID craftItem);
    std::optional<types::ItemID> search(const std::vector<types::ItemID>& recipe);
    void partial_insert(const std::vector<types::ConsumableID>& boc_content);
    void move_along(const types::ConsumableID consumable);
};

#endif // TRIE_HPP