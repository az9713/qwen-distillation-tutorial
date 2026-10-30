"""Build index.html from tutorial.md and the preserved interactive widgets."""
from html import escape
from pathlib import Path
import re

import markdown
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
TITLE = 'Qwen Model Distillation Techniques and Prompt Engineering Results'


def build():
    source = (ROOT / 'tutorial.md').read_text(encoding='utf-8')
    equations = []

    def equation(match):
        equations.append(match.group(1).strip())
        return f'\n\n<div class="math-equation">{escape(equations[-1])}</div>\n\n'

    source = re.sub(r'```math\n(.*?)\n```', equation, source, flags=re.S)
    # Preserve executable visualizations as authored on the selected Page.
    for name in ['results', 'training-update', 'soft-targets']:
        pattern = r'!\[([^\]]*)\]\(assets/' + name + r'\.html\)'
        widget = (ROOT / 'assets' / f'{name}.html').read_text(encoding='utf-8')
        source = re.sub(pattern, lambda m: f'\n\n<section class="visualization" aria-label="{escape(m.group(1), quote=True)}">\n{widget}\n</section>\n\n', source)
    rendered = markdown.markdown(source, extensions=['tables', 'fenced_code', 'sane_lists'])
    soup = BeautifulSoup(rendered, 'html.parser')
    outline = []
    used = {element['id'] for element in soup.find_all(attrs={'id': True})}
    for heading in soup.find_all(re.compile(r'^h[1-6]$')):
        if heading.find_parent(class_='visualization'):
            continue
        level = int(heading.name[1])
        heading.name = f'h{min(level + 1, 6)}'
        slug = re.sub(r'[^a-z0-9]+', '-', heading.get_text().lower()).strip('-')
        while slug in used:
            slug += '-section'
        used.add(slug)
        heading['id'] = slug
        if level == 1:
            outline.append((slug, heading.get_text()))
    for table in soup.find_all('table'):
        wrapper = soup.new_tag('div', attrs={'class': 'table-scroll', 'tabindex': '0', 'role': 'region', 'aria-label': 'Scrollable comparison table'})
        table.wrap(wrapper)
        for th in table.find_all('th'):
            th['scope'] = 'col'
    for image in soup.find_all('img'):
        image['loading'] = 'lazy'
        image['decoding'] = 'async'
    toc = ''.join(f'<li><a href="#{slug}">{escape(text)}</a></li>' for slug, text in outline)
    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <meta name="description" content="An interactive tutorial on Qwen agent distillation: teacher answers, soft targets, on-policy supervision, and the accuracy versus efficiency trade-off.">
  <title>{TITLE}</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="assets/site.css">
  <link rel="stylesheet" href="assets/katex/katex.min.css">
  <script defer src="assets/katex/katex.min.js"></script>
  <script defer src="assets/site.js"></script>
</head>
<body>
<a class="skip-link" href="#tutorial">Skip to tutorial</a>
<header class="site-header">
  <div class="eyebrow">Qwen distillation tutorial</div>
  <h1>{TITLE}</h1>
  <p class="lead">How teacher supervision changes a small agent’s answers, token probabilities, and execution habits.</p>
  <nav aria-label="Project links"><a href="https://www.youtube.com/watch?v=hOitZIkMT2Y">Watch the source video</a><a href="https://github.com/az9713/qwen-distillation-tutorial">View on GitHub</a></nav>
</header>
<div class="reading-layout">
  <aside><details open><summary>On this page</summary><nav aria-label="Tutorial contents"><ol>{toc}</ol></nav></details></aside>
  <main id="tutorial" tabindex="-1">{soup}</main>
</div>
<footer>Educational commentary based on Prompt Engineering’s experiment. Reported training results are attributed to the source; interactive diagrams are teaching aids. <a href="#">Back to top</a></footer>
</body>
</html>
'''
    (ROOT / 'index.html').write_text(html, encoding='utf-8')
    print(f'Built index.html: {len(outline)} sections, {len(equations)} equations, 3 interactive visualizations')


if __name__ == '__main__':
    build()
