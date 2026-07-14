const builtin = @import("builtin");

const has_avx512bw = builtin.cpu.has(.x86, .avx512bw);
const has_avx512vnni = builtin.cpu.has(.x86, .avx512vnni);
const has_avxvnni = builtin.cpu.has(.x86, .avxvnni);
const has_avx2 = builtin.cpu.has(.x86, .avx2);
const has_ssse3 = builtin.cpu.has(.x86, .ssse3);
const has_dotprod = builtin.cpu.has(.aarch64, .dotprod);

/// One packed dot-product instruction consumes four byte pairs per i32 lane.
/// Use the widest instruction family enabled for this build target.
pub const byte_lanes: comptime_int = if (has_avx512bw)
    64
else if (has_avx2)
    32
else
    16;

pub const output_lanes: comptime_int = byte_lanes / 4;
pub const U8Vec = @Vector(byte_lanes, u8);
pub const I8Vec = @Vector(byte_lanes, i8);
pub const I16Vec = @Vector(byte_lanes / 2, i16);
pub const I32Vec = @Vector(output_lanes, i32);

/// Repeat one four-byte activation group once for each output lane.
pub inline fn splatGroup(group: *align(1) const [4]u8) U8Vec {
    var result: U8Vec = undefined;
    inline for (0..output_lanes) |output| {
        inline for (0..4) |lane| {
            result[output * 4 + lane] = group[lane];
        }
    }
    return result;
}

inline fn fallbackDotAdd(sum: I32Vec, inputs: U8Vec, weights: I8Vec) I32Vec {
    var result = sum;
    inline for (0..output_lanes) |output| {
        var dot: i32 = 0;
        inline for (0..4) |lane| {
            dot += @as(i32, inputs[output * 4 + lane]) *
                @as(i32, weights[output * 4 + lane]);
        }
        result[output] += dot;
    }
    return result;
}

inline fn x86VnniDotAdd(sum: I32Vec, inputs: U8Vec, weights: I8Vec) I32Vec {
    const intrinsic = switch (byte_lanes) {
        64 => "llvm.x86.avx512.vpdpbusd.512",
        32 => "llvm.x86.avx512.vpdpbusd.256",
        else => unreachable,
    };
    return @extern(*const fn (I32Vec, I32Vec, I32Vec) callconv(.c) I32Vec, .{
        .name = intrinsic,
    }).*(sum, @bitCast(inputs), @bitCast(weights));
}

inline fn x86PackedDotAdd(sum: I32Vec, inputs: U8Vec, weights: I8Vec) I32Vec {
    const maddubs_name = switch (byte_lanes) {
        64 => "llvm.x86.avx512.pmaddubs.w.512",
        32 => "llvm.x86.avx2.pmadd.ub.sw",
        16 => "llvm.x86.ssse3.pmadd.ub.sw.128",
        else => unreachable,
    };
    const maddwd_name = switch (byte_lanes) {
        64 => "llvm.x86.avx512.pmaddw.d.512",
        32 => "llvm.x86.avx2.pmadd.wd",
        16 => "llvm.x86.sse2.pmadd.wd",
        else => unreachable,
    };
    const pair_sums = @extern(*const fn (U8Vec, I8Vec) callconv(.c) I16Vec, .{
        .name = maddubs_name,
    }).*(inputs, weights);
    const ones: I16Vec = @splat(1);
    const dot = @extern(*const fn (I16Vec, I16Vec) callconv(.c) I32Vec, .{
        .name = maddwd_name,
    }).*(pair_sums, ones);
    return sum + dot;
}

inline fn armDotAdd(sum: I32Vec, inputs: U8Vec, weights: I8Vec) I32Vec {
    return @extern(*const fn (I32Vec, I8Vec, I8Vec) callconv(.c) I32Vec, .{
        .name = "llvm.aarch64.neon.sdot.v4i32.v16i8",
    }).*(sum, @bitCast(inputs), weights);
}

/// Add four packed u8 x i8 products to each i32 output lane.
///
/// V7 pooled activations are in [0, 127], so the signed ARM dot-product path
/// is exact after bitcasting them to i8. On AVX2, every adjacent pair is also
/// bounded within i16, so the saturating maddubs intermediate cannot saturate.
pub inline fn dotAdd(sum: I32Vec, inputs: U8Vec, weights: I8Vec) I32Vec {
    if (comptime has_dotprod) return armDotAdd(sum, inputs, weights);
    if (comptime ((has_avx512vnni and byte_lanes == 64) or
        (has_avxvnni and byte_lanes == 32))) return x86VnniDotAdd(sum, inputs, weights);
    if (comptime (has_avx512bw or has_avx2 or has_ssse3)) return x86PackedDotAdd(sum, inputs, weights);
    return fallbackDotAdd(sum, inputs, weights);
}
