#include "pipeline/nodes/frame_capturer.hpp"

#include <iostream>

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

    XImage* image = XGetImage(display, root, 0, 0, width, height, AllPlanes, ZPixmap);

    if (!image) {
        XCloseDisplay(display);
        return nullptr;
    }

    if (this->width && (width != this->width || height != this->height)) {
        std::cerr << "[FrameCapturer] screen size changed: " << this->width << "x" << this->height
                  << " -> " << width << "x" << height << "\n";
        this->width = width;
        this->height = height;
    }

    ScreenCapture* capture = new ScreenCapture();
    capture->width = width;
    capture->height = height;

    // ScreenCapture::data is always tightly packed BGR24 -- see utils.hpp.
    const size_t out_size = static_cast<size_t>(width) * height * 3;
    capture->data = new unsigned char[out_size];

    const bool packed_bgr = (image->red_mask   == 0x00FF0000 &&
                             image->green_mask == 0x0000FF00 &&
                             image->blue_mask  == 0x000000FF);

    if (packed_bgr && (image->bits_per_pixel == 32 || image->bits_per_pixel == 24)) {
        // Fast path: wrap the XImage rows in place (honouring bytes_per_line,
        // which is padded and is NOT width * bpp) and let OpenCV do the
        // conversion. The old per-pixel XGetPixel loop cost ~2M virtual calls
        // per frame and was the pipeline's throughput bottleneck.
        const int type = (image->bits_per_pixel == 32) ? CV_8UC4 : CV_8UC3;
        cv::Mat src(height, width, type, image->data, image->bytes_per_line);
        cv::Mat bgr;
        if (type == CV_8UC4) {
            cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        } else {
            bgr = src;
        }
        // cvtColor output is continuous; a 24bpp source may not be, so copy row-wise.
        if (bgr.isContinuous()) {
            std::memcpy(capture->data, bgr.data, out_size);
        } else {
            for (int y = 0; y < height; y++) {
                std::memcpy(capture->data + static_cast<size_t>(y) * width * 3,
                            bgr.ptr(y), static_cast<size_t>(width) * 3);
            }
        }
    } else {
        // Fallback for exotic visuals (odd masks or bit depths).
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                unsigned long pixel = XGetPixel(image, x, y);
                size_t index = (static_cast<size_t>(y) * width + x) * 3;
                capture->data[index + 0] = (pixel & image->blue_mask) >> 0;   // B
                capture->data[index + 1] = (pixel & image->green_mask) >> 8;  // G
                capture->data[index + 2] = (pixel & image->red_mask) >> 16;   // R
            }
        }
    }

    XDestroyImage(image);
    XCloseDisplay(display);

    return capture;
}
