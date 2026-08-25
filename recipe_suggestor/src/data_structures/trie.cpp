#include "data_structures/trie.hpp"

// Node destructor is default in header now

SuggestionTrie::SuggestionTrie() { root = std::make_unique<Node>(); }

// Destructor is default, unique_ptr handles cleanup
SuggestionTrie::~SuggestionTrie() = default;

void SuggestionTrie::insert(const std::vector<types::ItemID>& recipe, const types::ItemID craftItem) {
    Node* current = root.get();
    for (const auto& item : recipe) {
        if (current->children.find(item) == current->children.end())
            current->children[item] = std::make_unique<Node>();
        current = current->children[item].get();
    }
    current->item = craftItem;
}

std::optional<types::ItemID> SuggestionTrie::search(const std::vector<types::ItemID>& recipe) {
    Node* current = root.get();
    for (const auto& item : recipe) {
        if (current->children.find(item) == current->children.end())
            return std::nullopt;
        current = current->children[item].get();
    }
    return current->item;
}

void SuggestionTrie::partial_insert(const std::vector<types::ConsumableID>& boc_content) {
    Node* current = root.get();
    for (const auto& item : boc_content) {
        if (current->children.find(item) == current->children.end())
            current->children[item] = std::make_unique<Node>();
        current = current->children[item].get();
    }
}

void SuggestionTrie::move_along(const types::ConsumableID consumable) {
    auto it = root->children.find(consumable);
    if (it == root->children.end()) {
        root = std::make_unique<Node>(); // Reset trie
        return;
    }

    std::unique_ptr<Node> chosen_child = std::move(it->second);    
    root = std::move(chosen_child);
}