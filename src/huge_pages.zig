const std = @import("std");
const builtin = @import("builtin");

const HUGE_PAGE_SIZE = 2 * 1024 * 1024;

/// Ask Linux to back the 2 MiB-aligned interior of `memory` with transparent
/// huge pages. Kernels in THP "madvise" mode (Ubuntu's default) otherwise use
/// 4 KiB pages, and the large weight tables and TT then miss the TLB on most
/// random accesses. Call before the memory is first written. No-op elsewhere.
pub fn advise(memory: []u8) void {
    if (builtin.os.tag != .linux) return;
    const start = std.mem.alignForward(usize, @intFromPtr(memory.ptr), HUGE_PAGE_SIZE);
    const end = std.mem.alignBackward(usize, @intFromPtr(memory.ptr) + memory.len, HUGE_PAGE_SIZE);
    if (end <= start) return;
    const aligned: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(start);
    std.posix.madvise(aligned, end - start, std.posix.MADV.HUGEPAGE) catch {};
}
