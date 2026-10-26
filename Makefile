ZIG ?= zig
EXE ?= sykora
# OpenBench passes EVALFILE= when a test selects an uploaded network.
EVALFILE ?=

# OpenBench currently passes its selected compiler as CC=. Honor that when it
# selected Zig, while keeping normal local builds independent of CC.
ZIG_CMD := $(if $(findstring zig,$(notdir $(CC))),$(CC),$(ZIG))

.PHONY: all clean

all:
	$(ZIG_CMD) build -Doptimize=ReleaseFast -Dopenbench-exe=$(EXE) $(if $(EVALFILE),-Devalfile=$(EVALFILE)) --prefix .

clean:
	$(RM) $(EXE) $(EXE).exe
