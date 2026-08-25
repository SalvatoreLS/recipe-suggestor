#pragma once
#include <sys/types.h>

namespace types {
    using ItemID = u_int16_t;
    using ConsumableID = u_int16_t;
    using Quantity = u_int8_t;
    using FrameCount = u_int32_t;

    // How a source crop is fitted to the model's square input. This MUST match
    // how the model's dataset was built, and the two datasets differ:
    //   Stretch        -- squash both axes independently, no padding. The floor
    //                     dataset is whole 1920x1080 frames exported from
    //                     Roboflow with "Resize to 640x640 (Stretch)".
    //   LetterboxBlack -- preserve aspect, centre, pad the remainder with BLACK.
    //                     The BoC dataset is a wide bag strip padded to a square
    //                     with black before it ever reached Roboflow: content
    //                     occupies rows 235..405 of every 640x640 training image.
    enum class Preprocess { Stretch, LetterboxBlack };
}