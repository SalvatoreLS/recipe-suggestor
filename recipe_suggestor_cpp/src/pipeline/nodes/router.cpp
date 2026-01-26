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
    u_int16_t crop_width = static_cast<u_int16_t>(source->width * constants::boc_crop_width_factor);
    u_int16_t crop_height = static_cast<u_int16_t>(source->height * constants::boc_crop_height_factor);
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
            
            // Calculate source and destination indices
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
    // 1. Initialize destination
    dest->width = source->width;
    dest->height = source->height;
    
    size_t dataSize = source->width * source->height * 3;
    dest->data = new unsigned char[dataSize];
    std::memcpy(dest->data, source->data, dataSize);

    // 2. Cover BoC region
    int x1 = static_cast<u_int16_t>((constants::crop_start_x_factor * source->width) - constants::boc_crop_width_factor);
    int y1 = static_cast<u_int16_t>(source->height - constants::boc_crop_height_factor);
    int x2 = x1 + (source->width * static_cast<u_int16_t>(constants::boc_crop_width_factor));
    int y2 = y1 + (source->height * static_cast<u_int16_t>(constants::boc_crop_height_factor));
    cover_region_inplace(dest, x1, y1, x2, y2);

    // 3. Cover left region
    x1 = static_cast<u_int16_t>(source->width * constants::left_section_x_factor);
    y1 = static_cast<u_int16_t>(source->height * constants::left_section_y_factor);
    x2 = x1 + static_cast<u_int16_t>(source->width * constants::left_crop_width_factor);
    y2 = y1 + static_cast<u_int16_t>(source->height * constants::left_crop_height_factor);
    cover_region_inplace(dest, x1, y1, x2, y2);
}


void Router::cover_region_inplace(ScreenCapture* img, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2) {
    // Ensure coordinates are within image bounds
    uint16_t startX = std::max((uint16_t)0, x1);
    uint16_t startY = std::max((uint16_t)0, y1);
    uint16_t endX = std::min((uint16_t)(img->width - 1), x2);
    uint16_t endY = std::min((uint16_t)(img->height - 1), y2);

    for (uint16_t y = startY; y <= endY; ++y) {
        for (uint16_t x = startX; x <= endX; ++x) {
            size_t offset = (y * img->width + x) * 3;
            img->data[offset] = 0;     // R
            img->data[offset + 1] = 0; // G
            img->data[offset + 2] = 0; // B
        }
    }
}

// TODO: add function for filling BoC image with black to make it 640x640 (define constants)
// TODO: define function for reshaping floor image to 640x640