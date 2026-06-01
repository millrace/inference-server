"""Phase-2 safetensors loader + verification gate (ARCHITECTURE.md §5.1).

Parses the safetensors header (8-byte little-endian length + JSON), locates named
tensors, and reads bf16 elements upcast to f32 — the mechanism the weight loader
will use to put Qwen2's weights on the GPU. Verifies, against values dumped by
torch (`loader-capture`), that for several real tensors the dtype is BF16, the
shape matches, and the first 8 flat elements decode bit-exactly.

bf16 -> f32 is exact: bf16 is the top 16 bits of an f32, so f32_bits = u16 << 16.

Needs `pixi run loader-capture` first (writes tests/fixtures/loader/, gitignored
— it references the ~1 GB checkpoint in the HF cache). Mismatch → non-zero exit.
"""

# JSON byte constants
comptime QUOTE = 34
comptime LBRACE = 123
comptime RBRACE = 125
comptime LBRACK = 91
comptime RBRACK = 93
comptime COLON = 58
comptime COMMA = 44
comptime TOL = Float32(1.0e-6)


@fieldwise_init
struct TensorEntry(Copyable, Movable):
    var name: String
    var dtype: String
    var shape: List[Int]
    var begin: Int
    var end: Int


def is_ws(c: Int) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13


def skip_ws(buf: List[UInt8], mut pos: Int):
    while pos < len(buf) and is_ws(Int(buf[pos])):
        pos += 1


def expect(buf: List[UInt8], mut pos: Int, ch: Int) raises:
    if pos >= len(buf) or Int(buf[pos]) != ch:
        raise Error("parse error: expected '" + chr(ch) + "' at byte " + String(pos))
    pos += 1


def parse_string(buf: List[UInt8], mut pos: Int) raises -> String:
    expect(buf, pos, QUOTE)
    var s = String("")
    while pos < len(buf) and Int(buf[pos]) != QUOTE:
        s += chr(Int(buf[pos]))
        pos += 1
    expect(buf, pos, QUOTE)
    return s^


def parse_uint(buf: List[UInt8], mut pos: Int) raises -> Int:
    var start = pos
    var val = 0
    while pos < len(buf):
        var c = Int(buf[pos])
        if c >= 48 and c <= 57:
            val = val * 10 + (c - 48)
            pos += 1
        else:
            break
    if pos == start:
        raise Error("parse error: expected integer at byte " + String(pos))
    return val


def parse_int_array(buf: List[UInt8], mut pos: Int) raises -> List[Int]:
    var out = List[Int]()
    expect(buf, pos, LBRACK)
    skip_ws(buf, pos)
    if Int(buf[pos]) == RBRACK:
        pos += 1
        return out^
    while True:
        skip_ws(buf, pos)
        out.append(parse_uint(buf, pos))
        skip_ws(buf, pos)
        if Int(buf[pos]) == COMMA:
            pos += 1
            continue
        break
    expect(buf, pos, RBRACK)
    return out^


def skip_value(buf: List[UInt8], mut pos: Int) raises:
    skip_ws(buf, pos)
    var c = Int(buf[pos])
    if c == QUOTE:
        _ = parse_string(buf, pos)
    elif c == LBRACE:
        skip_object(buf, pos)
    elif c == LBRACK:
        expect(buf, pos, LBRACK)
        skip_ws(buf, pos)
        if Int(buf[pos]) == RBRACK:
            pos += 1
            return
        while True:
            skip_value(buf, pos)
            skip_ws(buf, pos)
            if Int(buf[pos]) == COMMA:
                pos += 1
                continue
            break
        expect(buf, pos, RBRACK)
    else:
        while pos < len(buf):
            var d = Int(buf[pos])
            if d == COMMA or d == RBRACE or d == RBRACK or is_ws(d):
                break
            pos += 1


def skip_object(buf: List[UInt8], mut pos: Int) raises:
    expect(buf, pos, LBRACE)
    skip_ws(buf, pos)
    if Int(buf[pos]) == RBRACE:
        pos += 1
        return
    while True:
        skip_ws(buf, pos)
        _ = parse_string(buf, pos)
        skip_ws(buf, pos)
        expect(buf, pos, COLON)
        skip_value(buf, pos)
        skip_ws(buf, pos)
        if Int(buf[pos]) == COMMA:
            pos += 1
            continue
        break
    expect(buf, pos, RBRACE)


def parse_header(buf: List[UInt8]) raises -> List[TensorEntry]:
    var entries = List[TensorEntry]()
    var pos = 0
    skip_ws(buf, pos)
    expect(buf, pos, LBRACE)
    skip_ws(buf, pos)
    if Int(buf[pos]) == RBRACE:
        return entries^
    while True:
        skip_ws(buf, pos)
        var name = parse_string(buf, pos)
        skip_ws(buf, pos)
        expect(buf, pos, COLON)
        skip_ws(buf, pos)
        if name == "__metadata__":
            skip_object(buf, pos)
        else:
            expect(buf, pos, LBRACE)
            var dtype = String("")
            var shape = List[Int]()
            var begin = 0
            var end = 0
            skip_ws(buf, pos)
            if Int(buf[pos]) != RBRACE:
                while True:
                    skip_ws(buf, pos)
                    var fkey = parse_string(buf, pos)
                    skip_ws(buf, pos)
                    expect(buf, pos, COLON)
                    skip_ws(buf, pos)
                    if fkey == "dtype":
                        dtype = parse_string(buf, pos)
                    elif fkey == "shape":
                        shape = parse_int_array(buf, pos)
                    elif fkey == "data_offsets":
                        var offs = parse_int_array(buf, pos)
                        begin = offs[0]
                        end = offs[1]
                    else:
                        skip_value(buf, pos)
                    skip_ws(buf, pos)
                    if Int(buf[pos]) == COMMA:
                        pos += 1
                        continue
                    break
            expect(buf, pos, RBRACE)
            entries.append(TensorEntry(name, dtype^, shape^, begin, end))
        skip_ws(buf, pos)
        if pos < len(buf) and Int(buf[pos]) == COMMA:
            pos += 1
            continue
        break
    return entries^


def bf16_to_f32(lo: Int, hi: Int) -> Float32:
    var bits: UInt32 = (UInt32(hi) << 24) | (UInt32(lo) << 16)
    return UnsafePointer(to=bits).bitcast[Float32]()[0]


def to_list(raw: List[UInt8]) -> List[UInt8]:
    return raw.copy()


def read_f32_file(path: String) raises -> List[Float32]:
    var out = List[Float32]()
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var p = raw.unsafe_ptr().bitcast[Float32]()
        for i in range(len(raw) // 4):
            out.append(p[i])
    return out^


def main() raises:
    var fixdir = "tests/fixtures/loader/"

    # --- read meta.txt: path, count, then "<name> <ndim> <dims...>" ---
    var text: String
    with open(fixdir + "meta.txt", "r") as f:
        text = f.read()
    var lines = text.split("\n")
    if len(lines) < 2:
        raise Error("loader fixtures missing — run `pixi run loader-capture`")
    var path = String(lines[0]).strip()
    var count = Int(atol(String(lines[1]).strip()))

    var spec_names = List[String]()
    var spec_shapes = List[List[Int]]()
    for i in range(count):
        var parts = String(lines[2 + i]).split(" ")
        spec_names.append(String(parts[0]))
        var ndim = Int(atol(String(parts[1]).strip()))
        var dims = List[Int]()
        for j in range(ndim):
            dims.append(Int(atol(String(parts[2 + j]).strip())))
        spec_shapes.append(dims^)

    var expected = read_f32_file(fixdir + "expected.bin")

    print("safetensors loader spike — vs torch (", path, "):", sep="")

    var all_ok = True
    with open(path, "r") as f:
        var lenb = f.read_bytes(8)
        var header_len: UInt64 = 0
        for i in range(8):
            header_len |= UInt64(Int(lenb[i])) << UInt64(8 * i)
        var hdr = to_list(f.read_bytes(Int(header_len)))
        var entries = parse_header(hdr)
        var data_start = 8 + Int(header_len)

        for i in range(count):
            var name = spec_names[i]
            # find entry
            var idx = -1
            for e in range(len(entries)):
                if entries[e].name == name:
                    idx = e
                    break
            if idx < 0:
                print("  ", name, " [FAIL — not in header]", sep="")
                all_ok = False
                continue
            ref entry = entries[idx]

            # dtype + shape
            var shape_ok = len(entry.shape) == len(spec_shapes[i])
            if shape_ok:
                for d in range(len(entry.shape)):
                    if entry.shape[d] != spec_shapes[i][d]:
                        shape_ok = False
            var dtype_ok = entry.dtype == "BF16"

            # first 8 elements
            _ = f.seek(UInt64(data_start + entry.begin))
            var raw = f.read_bytes(16)
            var worst = Float32(0.0)
            for e in range(8):
                var got = bf16_to_f32(Int(raw[2 * e]), Int(raw[2 * e + 1]))
                var diff = abs(got - expected[i * 8 + e])
                if diff > worst:
                    worst = diff

            var ok = dtype_ok and shape_ok and worst <= TOL
            var tag = "OK" if ok else "FAIL"
            print(
                "  ", name, " dtype=", entry.dtype, " shape_ok=", shape_ok,
                " max_abs=", worst, " [", tag, "]", sep="",
            )
            all_ok = all_ok and ok

    if not all_ok:
        raise Error("safetensors loader does NOT match torch — spike FAILED")
    print("OK — Mojo safetensors header parse + bf16→f32 match torch on all tensors")
