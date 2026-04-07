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
        .default_value = "38",
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

pub fn handleHashChange(self: *Uci, value: []const u8) UciError!void {
    const parsed = std.fmt.parseInt(usize, value, 10) catch return UciError.InvalidArgument;
    if (parsed < 1 or parsed > 4096) {
        return UciError.InvalidArgument;
    }
    self.hash_size_mb = parsed;
    self.tt.resize(parsed) catch return UciError.OutOfMemory;
}
