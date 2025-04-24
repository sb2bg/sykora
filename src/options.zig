const std = @import("std");
const Uci = @import("interface.zig").Uci;
const UciError = @import("uci_error.zig").UciError;

pub const OptionType = enum {
    check,
    spin,
    combo,
    button,
    string,
};

pub const Option = struct {
    name: []const u8,
    type: OptionType,
    default_value: ?[]const u8 = null,
    min_value: ?i32 = null,
    max_value: ?i32 = null,
    var_values: ?[][]const u8 = null,
    current_value: ?[]const u8 = null,
    on_changed: ?*const fn (*Uci, []const u8) UciError!void = null,
    context: ?*Uci = null,
};

pub const Options = struct {
    items: std.ArrayList(Option),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Options {
        return Options{
            .items = std.ArrayList(Option).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Options) void {
        for (self.items.items) |*option| {
            if (option.var_values) |vars| {
                self.allocator.free(vars);
            }
        }
        self.items.deinit();
    }

    pub fn getOption(self: *Options, name: []const u8) ?*Option {
        for (self.items.items) |*option| {
            if (std.mem.eql(u8, option.name, name)) {
                return option;
            }
        }
        return null;
    }

    pub fn addOption(self: *Options, option: Option) UciError!void {
        self.items.append(option) catch return UciError.OutOfMemory;
    }

    pub fn setOption(self: *Options, name: []const u8, value: []const u8) UciError!void {
        if (self.getOption(name)) |option| {
            switch (option.type) {
                .check => {
                    if (std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "false")) {
                        option.current_value = value;
                    }
                },
                .spin => {
                    if (option.min_value) |min| {
                        if (option.max_value) |max| {
                            const val = std.fmt.parseInt(i32, value, 10) catch return;
                            if (val >= min and val <= max) {
                                option.current_value = value;
                            }
                        }
                    }
                },
                .combo => {
                    if (option.var_values) |vars| {
                        for (vars) |variant| {
                            if (std.mem.eql(u8, variant, value)) {
                                option.current_value = value;
                                break;
                            }
                        }
                    }
                },
                .button => {
                    // Button options don't have values
                },
                .string => {
                    option.current_value = value;
                },
            }

            if (option.on_changed) |callback| {
                if (option.context) |ctx| {
                    try callback(ctx, value);
                }
            }
        }
    }

    pub fn printOptions(self: *Options, writer: anytype) UciError!void {
        return self.printOptionsGenericError(writer) catch UciError.IOError;
    }

    fn printOptionsGenericError(self: *Options, writer: anytype) !void {
        for (self.items.items) |option| {
            try writer.print("option name {s} type {s}", .{ option.name, @tagName(option.type) });

            if (option.default_value) |default| {
                try writer.print(" default {s}", .{default});
            }

            if (option.min_value) |min| {
                try writer.print(" min {d}", .{min});
            }

            if (option.max_value) |max| {
                try writer.print(" max {d}", .{max});
            }

            if (option.var_values) |vars| {
                for (vars) |variant| {
                    try writer.print(" var {s}", .{variant});
                }
            }

            try writer.writeAll("\n");
        }
    }
};
