document.addEventListener('DOMContentLoaded', () => {
  if (!window.katex) return;
  document.querySelectorAll('.math-equation').forEach(element => {
    const formula = element.textContent;
    katex.render(formula, element, { displayMode: true, throwOnError: false, output: 'htmlAndMathml' });
  });
});
