#include "pipeline/nodes/frame_capturer.hpp"

FrameCapturer::FrameCapturer() {
    Display* display = XOpenDisplay(nullptr);
    if (display) {
        Screen* screen = DefaultScreenOfDisplay(display);
        this->width = screen->width;
        this->height = screen->height;
        XCloseDisplay(display);
    }
}

ScreenCapture* FrameCapturer::capture_screen() {
    Display* display = XOpenDisplay(nullptr);
    if (!display) return nullptr;

    Window root = DefaultRootWindow(display);
    XWindowAttributes attributes;
    XGetWindowAttributes(display, root, &attributes);

    int width = attributes.width;
    int height = attributes.height;

        XImage* image = XGetImage(display, root, 
                            0, 0, 
                            width, height, 
                            AllPlanes, ZPixmap);

    if (!image) {
        XCloseDisplay(display);
        return nullptr;
    }
    
    ScreenCapture* capture = new ScreenCapture();
    capture->width = width;
    capture->height = height;
    capture->data = new unsigned char[width * height * 4];
    
    // XImage to BGRA
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned long pixel = XGetPixel(image, x, y);
            
            int index = (y * width + x) * 4;
            capture->data[index + 0] = (pixel & image->blue_mask) >> 0;   // B
            capture->data[index + 1] = (pixel & image->green_mask) >> 8;  // G
            capture->data[index + 2] = (pixel & image->red_mask) >> 16;   // R
            // capture->data[index + 3] = 255;                               // A
        }
    }
    
    XDestroyImage(image);
    XCloseDisplay(display);
    
    return capture;
}