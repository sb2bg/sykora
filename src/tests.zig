const std = @import("std");

test {
    inline for (.{
        @import("search.zig"),
        @import("search/heuristics.zig"),
        @import("search/move_picker.zig"),
        @import("search/tt.zig"),
        @import("gensfen.zig"),
        @import("nnue.zig"),
        @import("full_threats_v1.zig"),
        @import("bench.zig"),
        @import("interface.zig"),
    }) |module| std.testing.refAllDecls(module);
}
