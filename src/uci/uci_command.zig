const std = @import("std");

pub const ToEngine = enum {
    uci,
    debug,
    isready,
    setoption,
    register,
    ucinewgame,
    position,
    go,
    stop,
    ponderhit,
    quit,
    display,
};

/// These are all the commands the engine recieves from the interface, plus
/// some novelty commands that are not part of the official protocol. The
/// engine will also send commands to the interface, through use of the ToGui enum.
pub const ToEngineCommand = union(ToEngine) {
    /// **uci**
    ///
    /// tell engine to use the uci (universal chess interface), this
    /// will be sent once as a first command after program boot to
    /// tell the engine to switch to uci mode. After receiving the uci
    /// command the engine must identify itself with the "id" command
    /// and send the "option" commands to tell the GUI which engine
    /// settings the engine supports if any. After that the engine
    /// should send "uciok" to acknowledge the uci mode. If no uciok
    /// is sent within a certain time period, the engine task will be
    /// killed by the GUI.
    uci,
    /// **debug** [on / off]
    ///
    /// switch the debug mode of the engine on and off. In debug mode
    /// the engine should send additional infos to the GUI, e. g. with
    /// the "info string" command, to help debugging, e. g. the commands
    /// that the engine has received etc. This mode should be switched
    /// off by default and this command can be sent any time, also when
    /// the engine is thinking.
    debug: bool,
    /// **isready**
    ///
    /// this is used to synchronize the engine with the GUI. When the
    /// GUI has sent a command or multiple commands that can take some
    /// time to complete, this command can be used to wait for the
    /// engine to be ready again or to ping the engine to find out if
    /// it is still alive. E. g. this should be sent after setting the
    /// path to the tablebases as this can take some time. This command
    /// is also required once before the engine is asked to do any search
    /// to wait for the engine to finish initializing. This command must
    ///  always be answered with "readyok" and can be sent also when the
    ///  engine is calculating in which case the engine should also
    /// immediately answer with "readyok" without stopping the search.
    isready,
    /// **setoption** name [value]
    ///
    /// this is sent to the engine when the user wants to change the
    /// internal parameters of the engine. For the "button" type no value
    /// is needed. One string will be sent for each parameter and this will
    /// only be sent when the engine is waiting. The name and value of the
    /// option in <id> should not be case sensitive and can inlude spaces.
    /// The substrings "value" and "name" should be avoided in \<id> and \<x>.
    setoption: SetOptionOptions,
    /// **register** [later | name | code]
    ///
    /// this is the command to try to register an engine or to tell the engine
    /// that registration will be done later. This command should always be
    /// sent if the engine has sent "registration error" at program startup.
    register: RegisterOptions,
    /// **ucinewgame**
    ///
    /// TODO
    ucinewgame,
    /// **position** [startpos | fen <fenstring>]  moves <move1> .... <movei>
    ///
    /// TODO
    position: PositionOptions,
    /// **go**
    ///
    /// TODO
    go: GoOptions,
    /// **stop**
    ///
    /// TODO
    stop,
    /// **ponderhit**
    ///
    /// TODO
    ponderhit,
    /// **quit**
    ///
    /// TODO
    quit,
    /// **display**
    ///
    /// display the current position in the terminal.
    ///
    /// **Note:** this command is not part of the official UCI protocol and is a novelty,
    /// quality of life feature.
    display,
};

pub const SetOptionOptions = struct {
    name: []const u8,
    value: []const u8,
};

pub const RegisterOptions = union(enum) {
    later,
    now: struct {
        name: []const u8,
        code: []const u8,
    },
};

pub const PositionOptions = struct {
    value: union(enum) {
        startpos,
        fen: []const u8,
    },
    moves: ?[][]const u8,
};

pub const GoOptions = struct {
    searchMoves: [][]const u8,
    ponder: bool,
    wtime: u64,
    btime: u64,
    winc: u64,
    binc: u64,
    movesToGo: u64,
    depth: u64,
    nodes: u64,
    mate: u64,
    moveTime: u64,
    infinite: bool,
};