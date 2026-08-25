#include "data_structures/trie.hpp"
#include "types.hpp"
#include <iostream>
#include <vector>
#include <cassert>

void test_insert_search() {
    SuggestionTrie trie;
    std::vector<types::ItemID> recipe = {1, 2, 3};
    types::ItemID craftItem = 100;

    trie.insert(recipe, craftItem);

    auto result = trie.search(recipe);
    assert(result.has_value());
    assert(result.value() == craftItem);
    std::cout << "test_insert_search passed" << std::endl;
}

void test_move_along() {
    SuggestionTrie trie;
    // Recipe: 1 -> 2 -> 3 puts item 100
    std::vector<types::ItemID> recipe = {1, 2, 3};
    types::ItemID craftItem = 100;
    trie.insert(recipe, craftItem);

    // Initial check: current root should not have item (unless empty recipe was inserted, which isn't the case)
    // Actually, search starts from root.

    // Move along 1
    trie.move_along(1);
    
    // Now searching for {2, 3} should work (conceptually) if we were searching from "current"
    // But search() takes a full recipe from root... wait.
    // Clarification: search() in the trie always starts from root. 
    // If move_along updates root, then we should search for the REMAINDER.
    
    std::vector<types::ItemID> remainder = {2, 3};
    auto result = trie.search(remainder);
    assert(result.has_value());
    assert(result.value() == craftItem);

    // Move along 2
    trie.move_along(2);
    std::vector<types::ItemID> last_step = {3};
    result = trie.search(last_step);
    assert(result.has_value());
    assert(result.value() == craftItem);

    // Move along 3
    trie.move_along(3);
    // Now we should be at the node with the item.
    // If we search for empty recipe, we should get the item?
    // Let's check search implementation:
    /*
    Node* current = root.get();
    for (const auto& item : recipe) { ... }
    return current->item;
    */
    // Yes, empty recipe means return current->item.
    result = trie.search({});
    assert(result.has_value());
    assert(result.value() == craftItem);

    std::cout << "test_move_along passed" << std::endl;
}

void test_move_along_reset() {
    SuggestionTrie trie;
    std::vector<types::ItemID> recipe = {1, 2};
    trie.insert(recipe, 50);

    // Move along something that doesn't exist
    trie.move_along(999);
    
    // Should have reset. Root should be empty.
    auto result = trie.search({});
    assert(!result.has_value());
    
    // Even original recipe shouldn't exist anymore if data structures are cleared?
    // "root = std::make_unique<Node>();" -> Yes, cleared.
    result = trie.search(recipe);
    assert(!result.has_value());

    std::cout << "test_move_along_reset passed" << std::endl;
}

int main() {
    test_insert_search();
    test_move_along();
    test_move_along_reset();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
