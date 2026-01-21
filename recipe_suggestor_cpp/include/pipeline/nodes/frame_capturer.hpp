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
    u_int16_t width;
    u_int16_t height; // TODO: collect them when initializing the frame capturer
};

#endif // FRAME_CAPTURER_CPP