#include "pipeline/nodes/recipe_suggestor.hpp"
#include "types.hpp"
#include <iostream>

RecipeSuggestor::RecipeSuggestor() { }

void RecipeSuggestor::suggest(cust::CircularList<types::ItemID>& boc, std::map<types::ConsumableID, types::Quantity>& floor_obj) {
    // TODO: implement later
    std::cout << "RecipeSuggestor::suggest called" << std::endl;
}