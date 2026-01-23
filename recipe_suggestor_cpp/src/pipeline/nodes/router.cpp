#include "pipeline/nodes/router.hpp"
#include "constants.hpp"
#include <sys/types.h>

Router::Router() {}

void Router::route(ScreenCapture* source_img, ScreenCapture* floor_img, ScreenCapture* boc_img) {
    std::thread boc_thread(&Router::process_boc, this, source_img, boc_img);
    std::thread floor_thread(&Router::process_floor, this, source_img, floor_img);

    boc_thread.join(); floor_thread.join();
}

void Router::process_boc(ScreenCapture* source, ScreenCapture* dest) {
    u_int16_t crop_width = static_cast<u_int16_t>(source->width * constants::crop_width_factor);
    u_int16_t crop_height = static_cast<u_int16_t>(source->height * constants::crop_height_factor);
    int start_X = static_cast<int>(constants::crop_start_x_factor * source->width) - crop_width;
    int start_Y = source->height - crop_height;
    
    // Allocate destination
    dest->width = crop_width;
    dest->height = crop_height;
    dest->data = new unsigned char[crop_width * crop_height * 3];

    // Copy the cropped region
    for (int y = 0; y < crop_height; y++) {
        for (int x = 0; x < crop_width; x++) {
            int srcX = start_X + x;
            int srcY = start_Y + y;
            
            // Calculate source and destination indices (assuming RGB format)
            int srcIndex = (srcY * source->width + srcX) * 3;
            int dstIndex = (y * crop_width + x) * 3;
            
            // Copy RGB pixels
            dest->data[dstIndex] = source->data[srcIndex];         // R
            dest->data[dstIndex + 1] = source->data[srcIndex + 1]; // G
            dest->data[dstIndex + 2] = source->data[srcIndex + 2]; // B
        }
    }
}

void Router::process_floor(ScreenCapture* source, ScreenCapture* dest) {
    // Cover region on the left
    // TODO

    // Cover bottom right region
    // TODO
    std::cout << "PROVA\n\n";
    dest = source;
    return;
}