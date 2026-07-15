const std = @import("std");
const nnue = @import("nnue.zig");
const options = @import("options.zig");
const Option = options.Option;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
const Uci = @import("interface.zig").Uci;

pub fn registerOptions(uci: *Uci) !void {
    try uci.options.items.append(uci.allocator, Option{
        .name = "Debug Log File",
        .type = .string,
        .default_value = "<empty>",
        .on_changed = handleLogFileChange,
        .context = uci,
    });
    try uci.options.items.append(uci.allocator, Option{
        .name = "UseNNUE",
        .type = .check,
        .default_value = "true",
        .on_changed = handleUseNnueChange,
        .context = uci,
    });
    try uci.options.items.append(uci.allocator, Option{
        .name = "EvalFile",
        .type = .string,
        .default_value = "<empty>",
        .on_changed = handleEvalFileChange,
        .context = uci,
    });
    try uci.options.items.append(uci.allocator, Option{
        .name = "NnueBlend",
        .type = .spin,
        .default_value = "100",
        .min_value = 0,
        .max_value = 100,
        .on_changed = handleNnueBlendChange,
        .context = uci,
    });
    try uci.options.items.append(uci.allocator, Option{
        .name = "NnueScale",
        .type = .spin,
        .default_value = "100",
        .min_value = 10,
        .max_value = 400,
        .on_changed = handleNnueScaleChange,
        .context = uci,
    });
    try uci.options.items.append(uci.allocator, Option{
        .name = "Threads",
        .type = .spin,
        .default_value = "1",
        .min_value = 1,
        .max_value = 64,
        .on_changed = handleThreadsChange,
        .context = uci,
    });
    try uci.options.items.append(uci.allocator, Option{
        .name = "Hash",
        .type = .spin,
        .default_value = "128",
        .min_value = 1,
        .max_value = 4096,
        .on_changed = handleHashChange,
        .context = uci,
    });
    try uci.options.items.append(uci.allocator, Option{
        .name = "Move Overhead",
        .type = .spin,
        .default_value = "30",
        .min_value = 0,
        .max_value = 5000,
        .on_changed = handleMoveOverheadChange,
        .context = uci,
    });

    // OpenBench parameter names must be single whitespace-free tokens. Scale
    // options are fixed-point integers where 100 represents logical 1.00.
    try registerSpin(uci, "LMRScale", "100", 50, 200, handleLmrScaleChange);
    try registerSpin(uci, "LMRHistoryScale", "100", 0, 300, handleLmrHistoryScaleChange);
    try registerSpin(uci, "LMPMoveScale", "100", 50, 200, handleLmpMoveScaleChange);
    try registerSpin(uci, "HistoryMaxBonus", "400", 50, 1600, handleHistoryMaxBonusChange);
}

fn registerSpin(
    uci: *Uci,
    name: []const u8,
    default_value: []const u8,
    min_value: i32,
    max_value: i32,
    on_changed: *const fn (*Uci, []const u8) UciError!void,
) !void {
    try uci.options.items.append(uci.allocator, Option{
        .name = name,
        .type = .spin,
        .default_value = default_value,
        .min_value = min_value,
        .max_value = max_value,
        .on_changed = on_changed,
        .context = uci,
    });
}

pub fn handleLogFileChange(self: *Uci, value: []const u8) UciError!void {
    if (self.log_file) |file| {
        file.close();
        self.log_file = null;
    }

    if (std.mem.eql(u8, value, "<empty>") or value.len == 0) {
        return;
    }

    self.log_file = std.fs.cwd().openFile(value, .{ .mode = .read_write }) catch
        std.fs.cwd().createFile(value, .{}) catch return UciError.IOError;

    self.log_file.?.seekFromEnd(0) catch return UciError.IOError;
}

pub fn handleUseNnueChange(self: *Uci, value: []const u8) UciError!void {
    if (std.mem.eql(u8, value, "true")) {
        self.use_nnue = true;
        return;
    }
    if (std.mem.eql(u8, value, "false")) {
        self.use_nnue = false;
        return;
    }
    return UciError.InvalidArgument;
}

pub fn handleEvalFileChange(self: *Uci, value: []const u8) UciError!void {
    if (std.mem.eql(u8, value, "<empty>") or value.len == 0) {
        if (self.eval_file_path) |old_path| {
            self.allocator.free(old_path);
            self.eval_file_path = null;
        }
        self.reloadEmbeddedNetwork();
        return;
    }

    var loaded = nnue.Network.loadFromFile(self.allocator, value) catch |err| {
        return switch (err) {
            nnue.LoadError.OutOfMemory => UciError.OutOfMemory,
            nnue.LoadError.InvalidNetwork,
            nnue.LoadError.UnsupportedVersion,
            nnue.LoadError.NetworkTooLarge,
            nnue.LoadError.AccumulatorBoundsExceeded,
            => UciError.InvalidArgument,
            else => UciError.IOError,
        };
    };

    const dup_path = self.allocator.dupe(u8, value) catch {
        loaded.deinit();
        return UciError.OutOfMemory;
    };

    if (self.nnue_network) |*old_network| {
        old_network.deinit();
    }
    if (self.eval_file_path) |old_path| {
        self.allocator.free(old_path);
    }

    self.nnue_network = loaded;
    self.eval_file_path = dup_path;
}

pub fn handleNnueBlendChange(self: *Uci, value: []const u8) UciError!void {
    const parsed = std.fmt.parseInt(i32, value, 10) catch return UciError.InvalidArgument;
    if (parsed < 0 or parsed > 100) {
        return UciError.InvalidArgument;
    }
    self.nnue_blend = parsed;
}

pub fn handleNnueScaleChange(self: *Uci, value: []const u8) UciError!void {
    const parsed = std.fmt.parseInt(i32, value, 10) catch return UciError.InvalidArgument;
    if (parsed < 10 or parsed > 400) {
        return UciError.InvalidArgument;
    }
    self.nnue_scale = parsed;
}

pub fn handleThreadsChange(self: *Uci, value: []const u8) UciError!void {
    const parsed = std.fmt.parseInt(usize, value, 10) catch return UciError.InvalidArgument;
    if (parsed < 1 or parsed > 64) {
        return UciError.InvalidArgument;
    }
    self.num_threads = parsed;
}

pub fn handleMoveOverheadChange(self: *Uci, value: []const u8) UciError!void {
    const parsed = std.fmt.parseInt(u64, value, 10) catch return UciError.InvalidArgument;
    if (parsed > 5000) return UciError.InvalidArgument;
    self.move_overhead_ms = parsed;
}

pub fn handleHashChange(self: *Uci, value: []const u8) UciError!void {
    const parsed = std.fmt.parseInt(usize, value, 10) catch return UciError.InvalidArgument;
    if (parsed < 1 or parsed > 4096) {
        return UciError.InvalidArgument;
    }
    self.hash_size_mb = parsed;
    self.tt.resize(parsed) catch return UciError.OutOfMemory;
}

fn parseTuningInt(value: []const u8, min_value: i32, max_value: i32) UciError!i32 {
    const parsed = std.fmt.parseInt(i32, value, 10) catch return UciError.InvalidArgument;
    if (parsed < min_value or parsed > max_value) return UciError.InvalidArgument;
    return parsed;
}

pub fn handleLmrScaleChange(self: *Uci, value: []const u8) UciError!void {
    self.search_tuning.lmr_scale_pct = try parseTuningInt(value, 50, 200);
}

pub fn handleLmrHistoryScaleChange(self: *Uci, value: []const u8) UciError!void {
    self.search_tuning.lmr_history_scale_pct = try parseTuningInt(value, 0, 300);
}

pub fn handleLmpMoveScaleChange(self: *Uci, value: []const u8) UciError!void {
    self.search_tuning.lmp_move_scale_pct = try parseTuningInt(value, 50, 200);
}

pub fn handleHistoryMaxBonusChange(self: *Uci, value: []const u8) UciError!void {
    self.search_tuning.history_max_bonus = @intCast(try parseTuningInt(value, 50, 1600));
}
