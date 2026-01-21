#ifndef RECIPE_SUGGESTOR_HPP
#define RECIPE_SUGGESTOR_HPP

#include "utils.hpp"
#include "data_structures/circular_list.hpp"
#include <map>
#include <sys/types.h>

class RecipeSuggestor {

public:
    RecipeSuggestor();
    void suggest(cust::CircularList<u_int16_t> boc, std::map<u_int16_t, u_int8_t> floor_obj); // TODO: define arguments and return type. It will receive both floor objects and BoC content

private:
    // TODO: Add functions for querying the "dataset" and returning ranked recipes

};

#endif // RECIPE_SUGGESTOR_HPP