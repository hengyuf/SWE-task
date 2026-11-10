import sys, json
from jinja2 import sandbox, StrictUndefined
from jinja2.ext import loopcontrols

def load(path):
    env = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters['tojson'] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    return env.from_string(open(path).read())

tools = [{"type":"function","function":{"name":"bash","description":"run cmd","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
         {"type":"function","function":{"name":"read","description":"read file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]
msgs = [
 {"role":"system","content":"You are an agent."},
 {"role":"user","content":"Fix the bug."},
 {"role":"assistant","content":None,"reasoning_content":"I should look.","tool_calls":[
    {"id":"call_b","type":"function","function":{"name":"read","arguments":{"path":"a.py"}}},
    {"id":"call_a","type":"function","function":{"name":"bash","arguments":{"command":"ls"}}}]},
 {"role":"tool","tool_call_id":"call_a","content":"a.py b.py"},
 {"role":"tool","tool_call_id":"call_b","content":"print(1)"},
 {"role":"assistant","content":"Done.","reasoning_content":"All good."},
 {"role":"user","content":"Thanks"},
]
for path in sys.argv[1:]:
    t = load(path)
    for kw in [dict(), dict(clear_thinking=True, reasoning_effort="low")]:
        out = t.render(messages=msgs, tools=tools, add_generation_prompt=True, **kw)
        print("=====", path.split('/')[-1], kw)
        print(out)
