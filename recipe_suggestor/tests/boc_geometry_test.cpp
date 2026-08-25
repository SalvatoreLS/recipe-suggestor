// Regression gate for the live bag geometry.
//
// The BoC model is only ever exercised directly on 640x640 dataset crops
// (boc_test), which hides everything the Router and the preprocessing do to a
// real screen frame. This test closes that gap without needing the game: it
// rebuilds a full 1920x1080 frame from a training crop by undoing the crop's
// black padding and pasting the content band back where constants say the bag
// strip lives, then checks that the detections coming out of
// Router -> BoCDetector match the ones the detector produces on the original
// crop. If the strip rect or the preprocessing drift apart from the dataset,
// this stops matching.
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>

#include "constants.hpp"
#include "pipeline/nodes/boc_detector.hpp"
#include "pipeline/nodes/router.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

static int failures = 0;
static void check(bool cond, const std::string& what) {
    std::cout << (cond ? "  ok   " : "  FAIL ") << what << std::endl;
    if (!cond) failures++;
}

// The content band of a BoC training crop: the strip that is not black padding.
// The padding is symmetric (the wide strip was centred in the square), so the
// first non-black row gives the whole geometry. Measuring the last non-black row
// instead would track where the sprites happen to end, not where the pad does.
static cv::Rect content_band(const cv::Mat& img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    int top = 0;
    while (top < img.rows / 2 && cv::countNonZero(gray.row(top) > 12) == 0) top++;
    return cv::Rect(0, top, img.cols, img.rows - 2 * top);
}

static std::map<int, int> class_counts(const std::vector<Prediction>& preds) {
    std::map<int, int> counts;
    for (const auto& p : preds) counts[p.classId]++;
    return counts;
}

static std::string describe(const std::map<int, int>& counts, const std::vector<std::string>& names) {
    std::string s;
    for (const auto& [id, n] : counts) {
        if (!s.empty()) s += ", ";
        s += (id < (int)names.size() ? names[id] : "?") + " x" + std::to_string(n);
    }
    return s.empty() ? "(nothing)" : s;
}

int main() {
    const std::string model_path = constants::boc_model_path;
    const std::string image_path =
        "resources/test_images/C00144_png.rf.a60cb305b62d17bb303017d841b5c12b.jpg";

    if (!fs::exists(model_path)) {
        std::cout << "boc_geometry_test skipped: " << model_path << " not present" << std::endl;
        return 0;
    }
    cv::Mat crop = cv::imread(image_path);
    if (crop.empty()) {
        std::cerr << "Failed to load " << image_path << std::endl;
        return 1;
    }

    BoCDetector detector(model_path, constants::img_width, constants::img_height);

    // 1. Baseline: the path boc_test already covers.
    std::vector<Prediction> direct;
    detector.detect(crop, &direct);
    const auto& names = detector.class_names();
    const auto direct_counts = class_counts(direct);
    std::cout << "direct on the 640x640 crop: " << describe(direct_counts, names) << std::endl;
    check(!direct.empty(), "the baseline crop detects something at all");

    // 2. Rebuild a screen frame: strip the padding, scale the band to the strip
    //    rect the Router will cut, and paste it there on black.
    const int W = 1920, H = 1080;
    Router router;
    const cv::Rect strip = router.boc_rect(W, H);
    std::cout << "strip rect: " << strip << " (aspect " << (double)strip.width / strip.height
              << "), training band aspect " << (double)crop.cols / content_band(crop).height
              << std::endl;

    cv::Mat band = crop(content_band(crop)).clone();
    cv::Mat scaled;
    cv::resize(band, scaled, strip.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat screen(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    scaled.copyTo(screen(strip));

    // 3. Route it exactly as the pipeline does, then detect on the crop it hands over.
    ScreenCapture source;
    source.width = W;
    source.height = H;
    source.data = new unsigned char[static_cast<size_t>(W) * H * 3];
    std::memcpy(source.data, screen.data, static_cast<size_t>(W) * H * 3);

    ScreenCapture floor_img = {nullptr, 0, 0};
    ScreenCapture boc_img = {nullptr, 0, 0};
    router.route(&source, &floor_img, &boc_img);

    cv::Mat routed = screen_capture_to_mat(boc_img);
    std::vector<Prediction> through_router;
    detector.detect(routed, &through_router);
    const auto routed_counts = class_counts(through_router);
    std::cout << "through Router -> BoCDetector: " << describe(routed_counts, names) << std::endl;

    check(routed_counts == direct_counts,
          "the routed frame yields the same classes and counts as the raw crop");

    // 4. Boxes must land in the same places too, in strip coordinates.
    std::vector<Prediction> d = direct, r = through_router;
    auto by_x = [](const Prediction& a, const Prediction& b) { return a.bbox.x < b.bbox.x; };
    std::sort(d.begin(), d.end(), by_x);
    std::sort(r.begin(), r.end(), by_x);
    if (d.size() == r.size()) {
        const cv::Rect band_rect = content_band(crop);
        const double sx = (double)strip.width / crop.cols;
        const double sy = (double)strip.height / band_rect.height;
        double worst = 0.0;
        for (size_t i = 0; i < d.size(); ++i) {
            // Baseline boxes are in padded-crop space; map them into strip space.
            const double ex = d[i].bbox.x * sx;
            const double ey = (d[i].bbox.y - band_rect.y) * sy;
            worst = std::max({worst, std::abs(ex - r[i].bbox.x), std::abs(ey - r[i].bbox.y)});
        }
        std::cout << "worst box-corner drift: " << worst << " px" << std::endl;
        check(worst < 12.0, "box positions agree within 12 px after the geometry mapping");
    }

    fs::create_directories("outputs");
    cv::imwrite("outputs/boc_geometry_screen.png", screen);
    cv::imwrite("outputs/boc_geometry_routed.png",
                detector.visualize_detections(routed, through_router));

    std::cout << (failures ? "boc_geometry_test FAILED" : "boc_geometry_test passed") << std::endl;
    return failures ? 1 : 0;
}
