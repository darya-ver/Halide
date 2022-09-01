//#include "/Users/darya/Code/MyHalide_old/Halide/distrib/include/Halide.h"
//#include "HalideBuffer.h"
//#include "Gene"

namespace {

class Blur3x3 : public Halide::Generator<Blur3x3> {
 public:
  Input<Buffer<uint16_t, 2>> input{"input"};
  Output<Buffer<uint16_t, 2>> blur_y{"blur_y"};

  void generate() {
    Func blur_x("blur_x");
    Var x("x"), y("y"), xi("xi"), yi("yi");

    // The algorithm
    blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;

    // CPU schedule.
    // Compute blur_x as needed at each vector of the output.
    // Halide will store blur_x in a circular buffer so its
    // results can be re-used.
    blur_y.split(y, y, yi, 32).parallel(y).vectorize(x, 16);

    blur_y.dim(0).set_min(0);
    blur_y.dim(0).set_extent(1920);
    blur_y.dim(1).set_min(0);
    blur_y.dim(1).set_extent(1080);

    blur_x.store_at(blur_y, y).compute_at(blur_y, x).vectorize(x, 16);
  }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Blur3x3, blur3x3)






//#include "HalideBuffer.h"
//#include "HalideRuntime.h"
//#include <assert.h>
//#include <math.h>
//#include <stdio.h>
//
//#include "blur2x2.h"
//
//#define RUN_BENCHMARKS 0
//#if RUN_BENCHMARKS
//#include "halide_benchmark.h"
//#endif
//
//using namespace Halide::Runtime;
//
//const int W = 80, H = 80;
//
//Buffer<float, 3> buffer_factory_planar(int w, int h, int c) {
//    return Buffer<float, 3>(w, h, c);
//}
//
//Buffer<float, 3> buffer_factory_interleaved(int w, int h, int c) {
//    return Buffer<float, 3>::make_interleaved(w, h, c);
//}
//
//void test(Buffer<float, 3> (*factory)(int w, int h, int c)) {
//    Buffer<float, 3> input = factory(W, H, 3);
//    input.for_each_element([&](int x, int y, int c) {
//        // Just an arbitrary color pattern with enough variation to notice the blur
//        if (c == 0) {
//            input(x, y, c) = ((x % 7) + (y % 3)) / 255.f;
//        } else if (c == 1) {
//            input(x, y, c) = (x + y) / 255.f;
//        } else {
//            input(x, y, c) = ((x * 5) + (y * 2)) / 255.f;
//        }
//    });
//    Buffer<float, 3> output = factory(W, H, 3);
//
//    printf("Evaluating output over %d x %d\n", W, H);
//    blur2x2(input, W, H, output);
//
//#if RUN_BENCHMARKS
//    double t = Halide::Tools::benchmark(10, 100, [&]() {
//        blur2x2(input, W, H, output);
//    });
//    const float megapixels = (W * H) / (1024.f * 1024.f);
//    printf("Benchmark: %d %d -> %f mpix/s\n", W, H, megapixels / t);
//#endif
//}
//
//int main(int argc, char **argv) {
//    printf("Testing planar buffer...\n");
//    test(buffer_factory_planar);
//
//    printf("Testing interleaved buffer...\n");
//    test(buffer_factory_interleaved);
//
//    printf("Success!\n");
//    return 0;
//}
