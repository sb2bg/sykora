# Sykora

![Sykora Logo](https://github.com/sb2bg/sykora/blob/main/assets/logo.png)

Sykora is a chess engine written in Zig that implements the Universal Chess Interface (UCI) protocol. It provides a robust and efficient implementation of chess game logic and UCI communication.

## Features

- Currently in progress, full feature list coming soon

## Prerequisites

- [Zig](https://ziglang.org/) compiler (latest stable version recommended)
- A UCI-compatible chess GUI (like Arena, Cutechess, or similar)

## Building

To build the project, run:

```bash
zig build
```

This will create an executable named `sykora` in the `zig-out/bin` directory.

## Running

To run the engine:

```bash
zig build run
```

Or directly run the executable:

```bash
./zig-out/bin/sykora
```

## Testing

To run the test suite:

```bash
zig build test
```

## Documentation

- `engine-interface.md` - Detailed documentation of the engine interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
