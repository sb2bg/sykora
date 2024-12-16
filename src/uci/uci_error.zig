pub const UciError = error{
    UnknownCommand,
    UnexpectedEOF,
    CommandTooLong,
    Unimplemented,
    IOError,
    OutOfMemory,
};

pub fn getErrorDescriptor(err: UciError) []const u8 {
    return switch (err) {
        UciError.UnknownCommand => "Unknown command.",
        UciError.UnexpectedEOF => "Unexpected EOF.",
        UciError.CommandTooLong => "Command too long.",
        UciError.Unimplemented => "Unimplemented.",
        UciError.IOError => "IO error.",
    };
}
