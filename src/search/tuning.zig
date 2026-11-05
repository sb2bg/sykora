const std = @import("std");

/// Search controls exposed through UCI for OpenBench tuning. Percentage fields
/// are fixed-point integers where 100 represents 1.00; margins are centipawns.
/// Defaults track the latest accepted SPSA candidate.
pub const SearchTuning = struct {
    // Late move reductions. The base and table are hundredths of a ply.
    lmr_base_x100: i32 = 0,
    lmr_scale_pct: i32 = 140,
    lmr_history_scale_pct: i32 = 95,
    lmr_full_depth_moves: u32 = 3,
    lmr_min_depth: u32 = 3,
    // Reduction relief in 1/1024 ply.
    lmr_killer: i32 = 1024,
    lmr_counter: i32 = 1024,
    lmr_improving: i32 = 1024,
    lmr_pv: i32 = 1024,
    lmr_cut_node: i32 = 1024,

    // Late move pruning.
    lmp_move_scale_pct: i32 = 90,
    lmp_max_depth: u32 = 3,
    lmp_improving_bonus: u32 = 2,

    // History.
    history_bonus_mult: i32 = 160,
    history_bonus_offset: i32 = 96,
    history_max_bonus: u32 = 1600,

    // Static-eval correction history weights (pawn, per-side non-pawn).
    corr_pawn_weight: i32 = 200,
    corr_non_pawn_weight: i32 = 100,
    corr_minor_weight: i32 = 100,
    corr_major_weight: i32 = 100,

    // Aspiration windows.
    asp_delta: i32 = 25,
    asp_widen_pct: i32 = 200,

    // Reverse futility pruning: margin * depth - improving_margin * improving.
    rfp_margin: i32 = 100,
    rfp_improving_margin: i32 = 100,
    rfp_max_depth: u32 = 7,

    // Futility pruning: base + mult * depth (+ improving).
    futility_base: i32 = 200,
    futility_mult: i32 = 120,
    futility_improving: i32 = 80,
    futility_max_depth: u32 = 3,

    razor_margin: i32 = 500,

    // Null move pruning.
    nmp_min_depth: u32 = 3,
    nmp_base_reduction: u32 = 3,
    nmp_eval_margin: i32 = 80,
    nmp_eval_reduce_margin: i32 = 200,
    nmp_depth_divisor: u32 = 3,

    // ProbCut.
    probcut_min_depth: u32 = 6,
    probcut_margin: i32 = 200,

    // Singular extensions: margin = base + per_ply * depth.
    singular_min_depth: u32 = 8,
    singular_margin_base: i32 = 30,
    singular_margin_per_ply: i32 = 25,

    // SEE pruning. Main-search margins scale with depth.
    main_see_margin: i32 = 70,
    main_see_max_depth: u32 = 4,
    quiet_see_margin: i32 = 20,
    quiet_see_max_depth: u32 = 7,
    qs_see_threshold: i32 = -80,
    qs_delta_margin: i32 = 300,

    // Adaptive soft time limit, in permille.
    tm_node_base: i32 = 1480,
    tm_node_scale: i32 = 1680,
    tm_stability_base: i32 = 1100,
    tm_stability_step: i32 = 30,

    /// 0 disables check extensions, 1 extends every checked node, and 2 only
    /// extends checked PV nodes.
    check_extension_mode: u32 = 2,
};

pub const Group = enum {
    /// Smooth search-shape parameters; the default SPSA input.
    search,
    /// Time management. Tune separately from search shape.
    time,
    /// Small integer domains (depth gates, move counts). Their rounding noise
    /// dominates simultaneous SPSA gradients, so test them with focused SPRTs.
    discrete,
    /// Mode switches; never tuned.
    mode,
};

pub const Tunable = struct {
    /// UCI option name; OpenBench requires a single whitespace-free token.
    name: []const u8,
    field: []const u8,
    min: i32,
    max: i32,
    group: Group = .search,
    /// SPSA C_end; zero means one twentieth of the range.
    c_end: f64 = 0,

    pub fn default(comptime self: Tunable) i32 {
        return @intCast(@field(SearchTuning{}, self.field));
    }

    pub fn defaultString(comptime self: Tunable) []const u8 {
        return struct {
            const text = blk: {
                @setEvalBranchQuota(10_000);
                break :blk std.fmt.comptimePrint("{d}", .{self.default()});
            };
        }.text;
    }

    pub fn cEnd(comptime self: Tunable) f64 {
        if (self.c_end != 0) return self.c_end;
        return @as(f64, @floatFromInt(self.max - self.min)) / 20.0;
    }
};

pub const tunables = [_]Tunable{
    .{ .name = "LMRBase", .field = "lmr_base_x100", .min = -100, .max = 150 },
    .{ .name = "LMRScale", .field = "lmr_scale_pct", .min = 50, .max = 200 },
    .{ .name = "LMRHistoryScale", .field = "lmr_history_scale_pct", .min = 0, .max = 300 },
    .{ .name = "LMRFullDepthMoves", .field = "lmr_full_depth_moves", .min = 1, .max = 8, .group = .discrete },
    .{ .name = "LMRKiller", .field = "lmr_killer", .min = 0, .max = 2048 },
    .{ .name = "LMRCounter", .field = "lmr_counter", .min = 0, .max = 2048 },
    .{ .name = "LMRImproving", .field = "lmr_improving", .min = 0, .max = 2048 },
    .{ .name = "LMRPV", .field = "lmr_pv", .min = 0, .max = 2048 },
    .{ .name = "LMRCutNode", .field = "lmr_cut_node", .min = 0, .max = 3072 },
    .{ .name = "LMRMinDepth", .field = "lmr_min_depth", .min = 1, .max = 6, .group = .discrete },
    .{ .name = "LMPMoveScale", .field = "lmp_move_scale_pct", .min = 50, .max = 200 },
    .{ .name = "LMPMaxDepth", .field = "lmp_max_depth", .min = 0, .max = 3, .group = .discrete },
    .{ .name = "LMPImprovingBonus", .field = "lmp_improving_bonus", .min = 0, .max = 8, .group = .discrete },
    .{ .name = "HistoryBonusMult", .field = "history_bonus_mult", .min = 40, .max = 400 },
    .{ .name = "HistoryBonusOffset", .field = "history_bonus_offset", .min = 0, .max = 400 },
    .{ .name = "HistoryMaxBonus", .field = "history_max_bonus", .min = 400, .max = 4000 },
    .{ .name = "CorrPawnWeight", .field = "corr_pawn_weight", .min = 0, .max = 400 },
    .{ .name = "CorrNonPawnWeight", .field = "corr_non_pawn_weight", .min = 0, .max = 300 },
    .{ .name = "CorrMinorWeight", .field = "corr_minor_weight", .min = 0, .max = 300 },
    .{ .name = "CorrMajorWeight", .field = "corr_major_weight", .min = 0, .max = 300 },
    .{ .name = "AspDelta", .field = "asp_delta", .min = 8, .max = 60 },
    .{ .name = "AspWidenPct", .field = "asp_widen_pct", .min = 130, .max = 300 },
    .{ .name = "RFPMargin", .field = "rfp_margin", .min = 40, .max = 200 },
    .{ .name = "RFPImprovingMargin", .field = "rfp_improving_margin", .min = 0, .max = 200 },
    .{ .name = "RFPMaxDepth", .field = "rfp_max_depth", .min = 1, .max = 12, .group = .discrete },
    .{ .name = "FutilityBase", .field = "futility_base", .min = 50, .max = 400 },
    .{ .name = "FutilityMult", .field = "futility_mult", .min = 40, .max = 250 },
    .{ .name = "FutilityImproving", .field = "futility_improving", .min = 0, .max = 200 },
    .{ .name = "FutilityMaxDepth", .field = "futility_max_depth", .min = 0, .max = 8, .group = .discrete },
    .{ .name = "RazorMargin", .field = "razor_margin", .min = 200, .max = 900 },
    .{ .name = "NMPMinDepth", .field = "nmp_min_depth", .min = 1, .max = 6, .group = .discrete },
    .{ .name = "NMPBaseReduction", .field = "nmp_base_reduction", .min = 1, .max = 6, .group = .discrete },
    .{ .name = "NMPEvalMargin", .field = "nmp_eval_margin", .min = 0, .max = 200 },
    .{ .name = "NMPEvalReduceMargin", .field = "nmp_eval_reduce_margin", .min = 50, .max = 400 },
    .{ .name = "NMPDepthDivisor", .field = "nmp_depth_divisor", .min = 2, .max = 8, .group = .discrete },
    .{ .name = "ProbCutMinDepth", .field = "probcut_min_depth", .min = 3, .max = 10, .group = .discrete },
    .{ .name = "ProbCutMargin", .field = "probcut_margin", .min = 80, .max = 350 },
    .{ .name = "SingularMinDepth", .field = "singular_min_depth", .min = 4, .max = 12, .group = .discrete },
    .{ .name = "SingularMarginBase", .field = "singular_margin_base", .min = 0, .max = 80 },
    .{ .name = "SingularMarginPerPly", .field = "singular_margin_per_ply", .min = 5, .max = 50 },
    .{ .name = "MainSEEMargin", .field = "main_see_margin", .min = 20, .max = 150 },
    .{ .name = "MainSEEMaxDepth", .field = "main_see_max_depth", .min = 0, .max = 8, .group = .discrete },
    .{ .name = "QuietSEEMargin", .field = "quiet_see_margin", .min = 0, .max = 80 },
    .{ .name = "QuietSEEMaxDepth", .field = "quiet_see_max_depth", .min = 0, .max = 12, .group = .discrete },
    .{ .name = "QSSEEThreshold", .field = "qs_see_threshold", .min = -250, .max = 50 },
    .{ .name = "QSDeltaMargin", .field = "qs_delta_margin", .min = 100, .max = 600 },
    .{ .name = "TMNodeBase", .field = "tm_node_base", .min = 1200, .max = 1800, .group = .time },
    .{ .name = "TMNodeScale", .field = "tm_node_scale", .min = 1000, .max = 2500, .group = .time },
    .{ .name = "TMStabilityBase", .field = "tm_stability_base", .min = 900, .max = 1400, .group = .time },
    .{ .name = "TMStabilityStep", .field = "tm_stability_step", .min = 0, .max = 60, .group = .time },
    .{ .name = "CheckExtension", .field = "check_extension_mode", .min = 0, .max = 2, .group = .mode },
};

comptime {
    for (tunables) |t| {
        if (!@hasField(SearchTuning, t.field)) @compileError("unknown tuning field " ++ t.field);
        if (t.default() < t.min or t.default() > t.max) @compileError(t.name ++ " default is out of range");
        if (@typeInfo(@FieldType(SearchTuning, t.field)).int.signedness == .unsigned and t.min < 0) {
            @compileError(t.name ++ " is unsigned but allows negative values");
        }
    }
}

/// Writes OpenBench SPSA input lines (`name, int, default, min, max, C_end, R_end`).
pub fn writeSpsaInput(writer: *std.Io.Writer, groups: []const Group) std.Io.Writer.Error!void {
    inline for (tunables) |t| {
        if (std.mem.indexOfScalar(Group, groups, t.group) != null) {
            try writer.print("{s}, int, {d}, {d}, {d}, {d:.1}, 0.002\n", .{ t.name, t.default(), t.min, t.max, t.cEnd() });
        }
    }
}

test "tunable defaults match UCI ranges and spsa output" {
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try writeSpsaInput(&writer, &.{.mode});
    try std.testing.expectEqualStrings("CheckExtension, int, 2, 0, 2, 0.1, 0.002\n", writer.buffered());
}
