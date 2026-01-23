#ifndef RECIPE_SUGGESTOR_HPP
#define RECIPE_SUGGESTOR_HPP

#include "utils.hpp"
#include "data_structures/circular_list.hpp"
#include <map>
#include <sys/types.h>
#include "types.hpp"

class RecipeSuggestor {

public:
    RecipeSuggestor();
    void suggest(cust::CircularList<types::ItemID>& boc, std::map<types::ItemID, types::Quantity>& floor_obj); // It will receive both floor objects and BoC content

private:
    // TODO: Add functions for querying the "dataset" and returning ranked recipes

};

#endif // RECIPE_SUGGESTOR_HPP