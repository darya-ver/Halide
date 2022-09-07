//#include "GenGen.cpp"
//#include "Halide.h"
//#include <stdio.h>
//
//using namespace Halide;
//using namespace Halide::Internal;
//
//class Blur3x3 : public Halide::Generator<Blur3x3> {
//public:
//    Input<Buffer<uint16_t, 2>> input{"input"};
//    Output<Buffer<uint16_t, 2>> blur_y{"blur_y"};
//
//    void generate() {
//        Func blur_x("blur_x");
//        Var x("x"), y("y"), xi("xi"), yi("yi");
//
//        // The algorithm
//        blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;
//        blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;
//
//        // CPU schedule.
//        // Compute blur_x as needed at each vector of the output.
//        // Halide will store blur_x in a circular buffer so its
//        // results can be re-used.
//        blur_y.split(y, y, yi, 32).parallel(y).vectorize(x, 16);
//
//        blur_y.dim(0).set_min(0);
//        blur_y.dim(0).set_extent(1920);
//        blur_y.dim(1).set_min(0);
//        blur_y.dim(1).set_extent(1080);
//
//        blur_x.store_at(blur_y, y).compute_at(blur_y, x).vectorize(x, 16);
//    }
//};
//
//HALIDE_REGISTER_GENERATOR(Blur3x3, blur3x3)

#include "GenGen.cpp"
#include "Halide.h"
#include "./hannk/common_halide.h"
#include "./hannk/common_halide.cpp"

using namespace Halide;
using namespace Halide::ConciseCasts;

namespace hannk {

class DepthwiseConv : public Generator<DepthwiseConv> {
public:
    // This is used to compute ci = co * inv_depth_multiplier. There are
    // only 2 values that make sense here:
    // - inv_depth_multiplier = 1 => depth_multiplier = 1
    // - inv_depth_multiplier = 0 => broadcasting 1 channel of input
    GeneratorParam<int> inv_depth_multiplier_{"inv_depth_multiplier", 0};

    // When true, we assume the vector size is divided evenly by the number
    // of channels, and we use the input_stride_x parameter as the stride of
    // x of the input, instead of the x dimension of the buffer.
    GeneratorParam<bool> shallow_{"shallow", false};

    // Unsigned 8-bit input tensor, indexed by ci, x, y, b.
    Input<Buffer<uint8_t, 4>> input_{"input"};
    Input<uint8_t> input_zero_{"input_zero"};

    // A 3D array of 8-bit filter coefficients indexed by co, x, y.
    Input<Buffer<uint8_t, 3>> filter_{"filter"};
    Input<uint8_t> filter_zero_{"filter_zero"};

    // A 1D array of 32-bit biases indexed by co.
    Input<Buffer<int32_t, 1>> bias_{"bias"};

    // The stride specifies how the input [x, y] are sub-subsampled. For every
    // spatial location [x, y] in the output buffer, the input buffer is sampled
    // spatially at [x * stride, y * stride]. The caller should ensure that
    // [x * stride, y * stride] is a valid spatial location in the input buffer.
    // Generally, this means setting the output buffer's [width, height] to be
    // the input buffer's [width, height] / stride.
    Input<int> stride_x_{"stride_x"};
    Input<int> stride_y_{"stride_y"};
    Input<int> dilation_x_{"dilation_x"};
    Input<int> dilation_y_{"dilation_y"};

    // When c and x are fused, this is used to specify the stride of x of the input
    // within the fused c-x dimension.
    Input<int> input_stride_x_{"input_stride_x"};

    Input<int32_t> output_multiplier_{"output_multiplier"};
    Input<int32_t> output_shift_{"output_shift"};
    Input<uint8_t> output_zero_{"output_zero"};
    Input<uint8_t> output_min_{"output_min"};
    Input<uint8_t> output_max_{"output_max"};

    Output<Buffer<uint8_t, 4>> output_{"output"};

    void generate() {
        // The algorithm.

        // For the shallow case, we need to know the vector size in the algorithm.
        int vector_size = natural_vector_size<uint8_t>();
        if (get_register_count(target) < 32) {
            vector_size = natural_vector_size<int16_t>();
        }

        // Some free variables, where x and y represent the spatial dimensions.
        Var x("x"), y("y"), c("c"), b("b");

        // Apply the c multiplier.
        Func resampled_input("resampled_input");
        resampled_input(c, x, y, b) = input_(c * inv_depth_multiplier_, x, y, b);

        Func filter_bounded("filter_bounded");
        Func bias_bounded("bias_bounded");
        Expr filter_c = c;
        if (shallow_) {
            // When the filter is shallow, we need a boundary condition on the
            // filter and bias.
            Expr filter_depth = filter_.dim(0).extent();
            filter_bounded(c, x, y) = filter_(c % filter_depth, x, y);
            bias_bounded(c) = bias_(c % filter_depth);

            // For shallow depthwise, we repeat the filter at multiples of the vector size.
            filter_c = c % vector_size;
        } else {
            filter_bounded(c, x, y) = filter_(c, x, y);
            bias_bounded(c) = bias_(c);
        }

        Func filter_zeroed("filter_zeroed");
        filter_zeroed(c, x, y) = i16(filter_bounded(c, x, y)) - i16(filter_zero_);

        // Do the convolution in 32-bit.
        filter_.dim(1).set_min(0);
        filter_.dim(2).set_min(0);
        Expr filter_width = filter_.dim(1).extent();
        Expr filter_height = filter_.dim(2).extent();
        RDom r(0, filter_width, 0, filter_height);
        Expr filter_zeroed_rdxy = filter_zeroed(filter_c, r.x, r.y);

        // We want to compute the reduction:
        // convolved(c, x, y, b) = bias(c)
        // convolved(c, x, y, b) +=
        //    i32(filter_zeroed_rdxy) *
        //    (i32(input_rdxy) - i32(input_zero_))
        //
        // However, this requires subtracting the input zero at every output.
        // We can factor the reduction like so:
        //
        // convolved(c, x, y, b) = bias(c)
        // convolved(c, x, y, b) +=
        //    i32(filter_zeroed_rdxy) * i32(input_rdxyc) -
        //    i32(filter_zeroed_rdxy) * i32(input_zero_)
        //
        // The latter reduction can be computed once per output channel.
        Func sum_filter("sum_filter");
        sum_filter(c) += i32(filter_zeroed_rdxy);

        Func offset_c("offset_c");
        offset_c(c) = bias_bounded(c) - sum_filter(c) * i32(input_zero_);

        Expr rx = x * stride_x_ + r.x * dilation_x_;
        Expr ry = y * stride_y_ + r.y * dilation_y_;
        Expr input_rdxy;
        if (shallow_) {
            input_rdxy = resampled_input(c + rx * input_stride_x_, 0, ry, b);
        } else {
            input_rdxy = resampled_input(c, rx, ry, b);
        }
        Func convolved("convolved");
        convolved(c, x, y, b) = offset_c(filter_c);
        convolved(c, x, y, b) += i32(filter_zeroed_rdxy) * i32(input_rdxy);

        output_(c, x, y, b) =
            quantize_and_relu_u8(convolved(c, x, y, b), output_multiplier_, output_shift_,
                                 output_zero_, output_min_, output_max_, target);

        // Schedule.
        interpret_as_tensor(input_);
        interpret_as_tensor(filter_);
        interpret_as_tensor(bias_);
        interpret_as_tensor(output_);
        require_same_min_extent(3, input_, output_);
        if (shallow_) {
            // Shallow inputs should have fused c and x, and left x as a dummy dim.
            output_.dim(1).set_min(0).set_extent(1);
        } else {
            require_same_min_extent(0, output_, bias_);
            require_same_min_extent(0, output_, filter_);
        }

        if (inv_depth_multiplier_ == 0) {
            // When we're broadcasting input channels, require that the input has only
            // one channel.
            input_.dim(0).set_extent(1);
        } else if (shallow_) {
            // Don't require alignment for shallow. We'd like to do so, but don't
            // have a good way to express it currently, since it requires
            // padding the fusion of two dimensions, and requiring alignment
            // will cause failures on wide-vector architectures like AVX512, HVX, etc.
            // We'll just pay the alignment penalty here for now.
        } else if (inv_depth_multiplier_ == 1) {
            // Require the input to be aligned.
            const int input_alignment = vector_size;
            input_.set_host_alignment(input_alignment);
            for (int d = 1; d < input_.dimensions(); d++) {
                input_.dim(d).set_stride(align(input_.dim(d).stride(), input_alignment));
            }
        }

        // Tile the output, so we can try to re-use loads spatially when performing
        // convolution. This also helps because we can schedule the input and not
        // waste work for strides less than the tile size.
        // We split co and reorder it outermost, so we can maximize locality of the
        // filter. We even put it outside of the batch loop, so we can compute the
        // boundary condition on the filter at co and reuse it across batches.
        const int kAccumulators = 4;
        const int kTileW = shallow_ ? 1 : 2;
        const int kTileH = kAccumulators / kTileW;
        // When the output is small, the overhead from shift inwards can be large.
        // Only tile when the input is at least this many tiles to avoid this.
        const int kMinTiles = 4;
        Var xo("xo"), yo("yo"), co("co");
        Expr output_width = output_.dim(1).extent();
        Expr output_height = output_.dim(2).extent();
        Expr use_tiles =
            (output_width >= kTileW * kMinTiles || output_width % kTileW == 0) &&
            (output_height >= kTileH * kMinTiles || output_height % kTileH == 0);
        output_.compute_root()
            .specialize(use_tiles)
            .tile(x, y, xo, yo, x, y, kTileW, kTileH, TailStrategy::ShiftInwards)
            .split(c, co, c, vector_size, TailStrategy::PredicateStores)
            .reorder(x, y, c, xo, yo, b, co)
            .unroll(x)
            .unroll(y)
            .vectorize(c);

        // In the general case, use dummy 1x1 tiles.
        output_
            .tile(x, y, xo, yo, x, y, 1, 1)
            .split(c, co, c, vector_size, TailStrategy::PredicateStores)
            .reorder(x, y, c, xo, yo, b, co)
            .unroll(x)
            .unroll(y)
            .vectorize(c);

        convolved.compute_at(output_, xo)
            .store_in(MemoryType::Register)
            .bound_extent(c, vector_size)
            .unroll(x)
            .unroll(y)
            .vectorize(c);
        convolved.update()
            .reorder(x, y, r.x, r.y)
            .unroll(x)
            .unroll(y)
            .vectorize(c);
        convolved.update()
            .specialize(filter_width == 3 && filter_height == 3)
            .unroll(r.x)
            .unroll(r.y);

        LoopLevel filter_compute_at = shallow_ ? LoopLevel::root() : LoopLevel(output_, co);

        // This doesn't read from any of the inputs directly, so we can vectorize
        // rounding up.
        offset_c.compute_at(filter_compute_at)
            .store_in(MemoryType::Stack)
            .vectorize(c, vector_size, TailStrategy::RoundUp);

        filter_zeroed.compute_at(filter_compute_at)
            .store_in(MemoryType::Stack)
            .align_storage(c, vector_size)
            .vectorize(c, vector_size, TailStrategy::PredicateLoads);

        bias_bounded.compute_at(filter_compute_at)
            .store_in(MemoryType::Stack)
            .vectorize(c, vector_size, TailStrategy::PredicateLoads);
    }
};

}  // namespace hannk

HALIDE_REGISTER_GENERATOR(hannk::DepthwiseConv, depthwise_conv_broadcast_uint8)

//
// using namespace Halide;
//
// int main(int argc, char **argv) {
//    Buffer<uint32_t> input(256);
//    for (int i = 0; i < 256; i++) {
//        input(i) = rand();
//    }
//    Var x;
//
//    // reinterpret cast
//    Func f1;
//    f1(x) = reinterpret<float>(input(x));
//    Buffer<float> im1 = f1.realize({256});
//
//    for (int x = 0; x < 256; x++) {
//        float halide = im1(x);
//        float c = Halide::Internal::reinterpret_bits<float>(input(x));
//        if (halide != c && std::isnan(halide) ^ std::isnan(c)) {
//            printf("reinterpret<float>(%x) -> %f instead of %f\n", input(x), halide, c);
//            return -1;
//        }
//    }
//
//    // bitwise xor
//    Func f2;
//    f2(x) = input(x) ^ input(x + 1);
//    Buffer<uint32_t> im2 = f2.realize({128});
//    for (int x = 0; x < 128; x++) {
//        uint32_t correct = input(x) ^ input(x + 1);
//        if (im2(x) != correct) {
//            printf("%x ^ %x -> %x instead of %x\n",
//                   input(x), input(x + 1), im2(x), correct);
//            return -1;
//        }
//    }
//
//    // bitwise and
//    Func f3;
//    f3(x) = input(x) & input(x + 1);
//    Buffer<uint32_t> im3 = f3.realize({128});
//    for (int x = 0; x < 128; x++) {
//        uint32_t correct = input(x) & input(x + 1);
//        if (im3(x) != correct) {
//            printf("%x & %x -> %x instead of %x\n",
//                   input(x), input(x + 1), im3(x), correct);
//            return -1;
//        }
//    }
//
//    // bitwise or
//    Func f4;
//    f4(x) = input(x) | input(x + 1);
//    Buffer<uint32_t> im4 = f4.realize({128});
//    for (int x = 0; x < 128; x++) {
//        uint32_t correct = input(x) | input(x + 1);
//        if (im4(x) != correct) {
//            printf("%x | %x -> %x instead of %x\n",
//                   input(x), input(x + 1), im4(x), correct);
//            return -1;
//        }
//    }
//
//    // bitwise not
//    Func f5;
//    f5(x) = ~input(x);
//    Buffer<uint32_t> im5 = f5.realize({128});
//    for (int x = 0; x < 128; x++) {
//        uint32_t correct = ~input(x);
//        if (im5(x) != correct) {
//            printf("~%x = %x instead of %x\n",
//                   input(x), im5(x), correct);
//            return -1;
//        }
//    }
//
//    // shift left combined with masking
//    Func f6;
//    f6(x) = input(x) << (input(x + 1) & 0xf);
//    Buffer<uint32_t> im6 = f6.realize({128});
//    for (int x = 0; x < 128; x++) {
//        uint32_t correct = input(x) << (input(x + 1) & 0xf);
//        if (im6(x) != correct) {
//            printf("%x << (%x & 0xf) -> %x instead of %x\n",
//                   input(x), input(x + 1), im6(x), correct);
//            return -1;
//        }
//    }
//
//    // logical shift right
//    Func f7;
//    f7(x) = input(x) >> (input(x + 1) & 0xf);
//    Buffer<uint32_t> im7 = f7.realize({128});
//    for (int x = 0; x < 128; x++) {
//        uint32_t correct = input(x) >> (input(x + 1) & 0xf);
//        if (im7(x) != correct) {
//            printf("%x >> (%x & 0xf) -> %x instead of %x\n",
//                   input(x), input(x + 1), im7(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift right
//    Func f8;
//    Expr a = reinterpret<int>(input(x));
//    Expr b = reinterpret<unsigned>(input(x + 1));
//    f8(x) = a >> (b & 0x1f);
//    Buffer<int> im8 = f8.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) >> (((int)(input(x + 1))) & 0x1f);
//        if (im8(x) != correct) {
//            printf("%x >> uint32(%x & 0x1f) -> %x instead of %x\n",
//                   input(x), input(x + 1), im8(x), correct);
//            return -1;
//        }
//    }
//
//    // bit shift on mixed types
//    Func f9;
//    Expr a32 = cast<int32_t>(input(x));
//    Expr b8 = cast<int32_t>(min(31, cast<uint8_t>(input(x + 1))));
//    f9(x) = a32 >> b8;
//    Buffer<int> im9 = f9.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int lhs = (int)input(x);
//        int shift_amount = (uint8_t)(input(x + 1));
//        shift_amount = std::min(31, shift_amount);
//        int correct = lhs >> shift_amount;
//        if (im9(x) != correct) {
//            printf("%d >> %d -> %d instead of %d\n",
//                   input(x), shift_amount, im9(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift left with signed type (positive value)
//    Func f10;
//    Expr a10 = cast<int>(input(x));
//    Expr b10 = cast<int>(input(x + 1));
//    f10(x) = a10 << (b10 & 0x1f);
//    Buffer<int> im10 = f10.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) << (((int)(input(x + 1))) & 0x1f);
//        if (im10(x) != correct) {
//            printf("%x << (%x & 0x1f) -> %x instead of %x\n",
//                   input(x), input(x + 1), im10(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift right with signed type (positive value) and mixed types
//    Func f11;
//    Expr a11 = cast<int>(input(x));
//    Expr b11 = cast<int>(input(x + 1));
//    f11(x) = a11 >> cast<int16_t>(b11 & 0x0f);
//    Buffer<int> im11 = f11.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) >> (((int)(input(x + 1))) & 0x0f);
//        if (im11(x) != correct) {
//            printf("%x >> (%x & 0x1f) -> %x instead of %x\n",
//                   input(x), input(x + 1), im11(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift left with signed type (negative value)
//    Func f12;
//    Expr a12 = cast<int>(input(x));
//    Expr b12 = cast<int>(input(x + 1));
//    f12(x) = a12 << (-1 * (b12 & 0x1f));
//    Buffer<int> im12 = f12.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) >> (((int)(input(x + 1))) & 0x1f);
//        if (im12(x) != correct) {
//            printf("%x << (-1 * (%x & 0x1f)) -> %x instead of %x\n",
//                   input(x), input(x + 1), im12(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift right with signed type (negative value)
//    Func f13;
//    Expr a13 = cast<int>(input(x));
//    Expr b13 = cast<int>(input(x + 1));
//    f13(x) = a13 >> (-1 * (b13 & 0x1f));
//    Buffer<int> im13 = f13.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) << (((int)(input(x + 1))) & 0x1f);
//        if (im13(x) != correct) {
//            printf("%x >> (-1 * (%x & 0x1f)) -> %x instead of %x\n",
//                   input(x), input(x + 1), im13(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift left with signed type (constant positive value)
//    Func f14;
//    Expr a14 = cast<int>(input(x));
//    int b14 = 4;
//    f14(x) = a14 << b14;
//    Buffer<int> im14 = f14.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) << 4;
//        if (im14(x) != correct) {
//            printf("%x << %x -> %x instead of %x\n",
//                   input(x), b14, im14(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift right with signed type (constant positive value)
//    Func f15;
//    Expr a15 = cast<int>(input(x));
//    int b15 = 4;
//    f15(x) = a15 >> b15;
//    Buffer<int> im15 = f15.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) >> 4;
//        if (im15(x) != correct) {
//            printf("%x >> %x -> %x instead of %x\n",
//                   input(x), b15, im15(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift left with signed type (constant negative value)
//    Func f16;
//    Expr a16 = cast<int>(input(x));
//    int b16 = -4;
//    f16(x) = a16 << b16;
//    Buffer<int> im16 = f16.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) >> 4;
//        if (im16(x) != correct) {
//            printf("%x << %x -> %x instead of %x\n",
//                   input(x), b16, im16(x), correct);
//            return -1;
//        }
//    }
//
//    // arithmetic shift right with signed type (constant negative value)
//    Func f17;
//    Expr a17 = cast<int>(input(x));
//    int b17 = -4;
//    f17(x) = a17 >> b17;
//    Buffer<int> im17 = f17.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int correct = ((int)(input(x))) << 4;
//        if (im17(x) != correct) {
//            printf("%x >> %x -> %x instead of %x\n",
//                   input(x), b17, im17(x), correct);
//            return -1;
//        }
//    }
//
//    // bitwise and on mixed types
//    Func f18;
//    Expr a8 = cast<int8_t>(input(x));
//    f18(x) = a8 & cast<int8_t>(0xf0);
//    Buffer<int8_t> im18 = f18.realize({128});
//    for (int x = 0; x < 128; x++) {
//        int8_t correct = (int8_t)(input(x)) & 0xf0;
//        if (im18(x) != correct) {
//            printf("(int8_t)%x & 0xf0 -> %x instead of %x\n",
//                   input(x), im18(x), correct);
//            return -1;
//        }
//    }
//
//    // bitwise xor scalar/vector
//    Expr vec = cast(UInt(8).with_lanes(4), 42) ^ 3;
//    assert(vec.type().lanes() == 4);
//
//    // Ensure signedness is preserved.
//    Expr vec2 = cast(UInt(8).with_lanes(4), 42) & 3;
//    assert(vec.type().is_uint());
//
//    // Ensure that bitwise op is commutative re: type.  (This was not
//    // true at least for some time, which is problematic given that
//    // simplification and other things assume expressions can be
//    // reordered.)
//    {
//        Expr a = cast(UInt(8), 42);
//        Expr b = cast(UInt(16), 199);
//
//        Expr a_then_b = a ^ b;
//        Expr b_then_a = b ^ a;
//
//        assert(a_then_b.type() == b_then_a.type());
//    }
//
//    printf("Success!\n");
//    return 0;
//}
