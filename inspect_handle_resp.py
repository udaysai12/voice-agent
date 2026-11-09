import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('web_bot.py', 'r', encoding='utf-8') as f:
    text = f.read()

start_html = text.find('HTML_PAGE =')
end_html = text.find('OPENAPI_SPEC =')
html_str = text[start_html:end_html]

script_start = html_str.find('<script>')
script_end = html_str.find('</script>')
js_code = html_str[script_start+8:script_end]

lines = js_code.split('\n')

for i in range(1005, 1165):
    if i < len(lines):
        print(f'{i+1:4d}: {lines[i]}')

