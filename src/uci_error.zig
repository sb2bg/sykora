pub const UciError = error{
    UnknownCommand,
    UnexpectedEOF,
    CommandTooLong,
    InvalidArgument,
    InvalidFen,
    InvalidMove,
    Unimplemented,
    UnsupportedCommand,
    IOError,
    OutOfMemory,
    ThreadCreationFailed,
    Quit,
};

pub fn getErrorDescriptor(err: UciError) []const u8 {
    return switch (err) {
        UciError.UnknownCommand => "unknown command",
        UciError.UnexpectedEOF => "unexpected EOF",
        UciError.CommandTooLong => "command too long",
        UciError.InvalidArgument => "invalid argument",
        UciError.InvalidFen => "invalid FEN",
        UciError.InvalidMove => "invalid move",
        UciError.Unimplemented => "unimplemented",
        UciError.IOError => "IO error",
        UciError.OutOfMemory => "out of memory",
        UciError.ThreadCreationFailed => "thread creation failed",
        UciError.UnsupportedCommand => "unsupported command",
        UciError.Quit => "quit",
    };
}
