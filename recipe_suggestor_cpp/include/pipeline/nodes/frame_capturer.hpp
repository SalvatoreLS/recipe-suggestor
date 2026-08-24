#ifndef FRAME_CAPTURER_CPP
#define FRAME_CAPTURER_CPP

#include "utils.hpp"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cstdint>
#include <cstring>
#include <sys/types.h>
#include <vector>

class FrameCapturer {

public:
    FrameCapturer();
    ScreenCapture* capture_screen(); // Executed by the pipeline


private:
    // Screen size sampled at construction; used to notice resolution changes.
    u_int16_t width = 0;
    u_int16_t height = 0;
};

#endif // FRAME_CAPTURER_CPP