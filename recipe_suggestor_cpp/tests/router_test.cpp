#include <iostream>
#include "pipeline/nodes/router.hpp"
#include "utils.hpp"

int main() {
    Router router;
    
    // Mock Source Image (100x100)
    int w = 1920;
    int h = 1080;
    ScreenCapture* source = new ScreenCapture();
    source->width = w;
    source->height = h;
    source->data = new unsigned char[w * h * 3];
    // Fill with some data
    for(int i=0; i<w*h*3; ++i) source->data[i] = (unsigned char)(i % 255);
    
    ScreenCapture floor_img = {nullptr, 0, 0};
    ScreenCapture boc_img = {nullptr, 0, 0};
    
    std::cout << "Routing..." << std::endl;
    router.route(source, &floor_img, &boc_img);
    
    std::cout << "BoC Image: " << boc_img.width << "x" << boc_img.height << std::endl;
    std::cout << "Floor Image: " << floor_img.width << "x" << floor_img.height << std::endl; // Floor not implemented yet fully
    
    // Cleanup
    delete[] source->data;
    delete source;
    if (boc_img.data) delete[] boc_img.data;
    if (floor_img.data) delete[] floor_img.data;
    
    return 0;
}
