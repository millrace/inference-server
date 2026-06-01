"""Chat-template rendering via minja2 (ARCHITECTURE.md §5.3).

Renders the model's real Jinja chat template (assets/qwen2.5-chat-template.jinja)
with the ../minja2 engine, replacing the hardcoded no-tools template the CLI and
server used. The messages context is built as JSON and parsed into a minja2
`Value` (simpler than constructing values by hand). Compile once, render many.

Built with `-I ../minja2/src` so minja2's modules resolve (it compiles cleanly
under the same 1.0.0b2 nightly the GPU engine needs — unlike flare, §11 #11).
"""

from template import Template
from json import parse_json


def json_escape(s: String) -> String:
    var out = String("")
    var sb = s.as_bytes()
    for i in range(len(sb)):
        var c = Int(sb[i])
        if c == 34:
            out += "\\\""
        elif c == 92:
            out += "\\\\"
        elif c == 10:
            out += "\\n"
        elif c == 13:
            out += "\\r"
        elif c == 9:
            out += "\\t"
        else:
            out += chr(c)
    return out^


def load_chat_template(path: String) raises -> Template:
    with open(path, "r") as f:
        return Template.compile(f.read())


def render_chat(tmpl: Template, user: String) raises -> String:
    """Render the template for a single user turn with add_generation_prompt."""
    var ctx_json = (
        String('{"messages":[{"role":"user","content":"')
        + json_escape(user)
        + '"}],"add_generation_prompt":true,"tools":null}'
    )
    var ctx = parse_json(ctx_json)
    return tmpl.render(ctx^, 0)
