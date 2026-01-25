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

    // Cover BoC region
    int x1 = static_cast<u_int16_t>((constants::crop_start_x_factor * source->width) - constants::boc_crop_width_factor);
    int y1 = static_cast<u_int16_t>(source->height - constants::boc_crop_height_factor);
    int x2 = x1 + (source->width * static_cast<u_int16_t>(constants::boc_crop_width_factor));
    int y2 = y1 + (source->height * static_cast<u_int16_t>(constants::boc_crop_height_factor));
    dest = coverRegion(source, x1, y1, x2, y2);

    // Cover left region
    x1 = static_cast<u_int16_t>(source->width * constants::left_section_x_factor);
    y1 = static_cast<u_int16_t>(source->height * constants::left_section_y_factor);
    x2 = x1 + static_cast<u_int16_t>(source->width * constants::left_crop_width_factor);
    y2 = y1 + static_cast<u_int16_t>(source->height * constants::left_crop_height_factor);
    dest = coverRegion(source, x1, y1, x2, y2);

}


ScreenCapture* Router::coverRegion(const ScreenCapture* img, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2) {
    size_t pixelCount = img->width * img->height;
    size_t dataSize = pixelCount * 4; 
    
    unsigned char* newData = new unsigned char[dataSize];
    std::memcpy(newData, img->data, dataSize);

    // Ensure coordinates are within image bounds
    uint16_t startX = std::max((uint16_t)0, x1);
    uint16_t startY = std::max((uint16_t)0, y1);
    uint16_t endX = std::min((uint16_t)(img->width - 1), x2);
    uint16_t endY = std::min((uint16_t)(img->height - 1), y2);

    for (uint16_t y = startY; y <= endY; ++y) {
        for (uint16_t x = startX; x <= endX; ++x) {
            size_t offset = (y * img->width + x) * 3;
            newData[offset] = 0;     // R
            newData[offset + 1] = 0; // G
            newData[offset + 2] = 0; // B
            // newData[offset + 3] = 0; // A
        }
    }

    return new ScreenCapture{newData, img->width, img->height};
}

// TODO: fix router_test.cpp