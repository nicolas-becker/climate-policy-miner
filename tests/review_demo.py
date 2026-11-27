"""
Run the validation, results and social card pages without Azure, LangChain
or Unstructured. Creates a sample PDF and a fake pipeline run.

    python tests/review_demo.py            # serves on http://localhost:5055
    open http://localhost:5055/results/demo-task-0001

The fake LLM returns a canned summary, so "Finish and rebuild results" works
offline.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import fitz  # noqa: E402
from flask import Flask  # noqa: E402

import review  # noqa: E402

TASK_ID = 'demo-task-0001'
BASE = os.path.join(tempfile.gettempdir(), 'review_demo')

PAGES = {
    3: ['Foreword', 'This updated NDC replaces the submission of September 2025 following a revised greenhouse gas inventory.'],
    5: ['2. Mitigation contribution',
        'The country will reduce greenhouse gas emissions by 30% relative to the business-as-usual scenario by 2035, of which 12% is unconditional.',
        'Net zero emissions will be reached by 2060.'],
    7: ['4.3 Electric mobility strategy',
        'The strategy foresees the installation of approximately 5,000 charging points or stations across all regions.',
        'All new public buses procured will be electric by 2035, subject to international support.',
        'A bus rapid transit corridor will be developed in the capital to shift trips from private cars.'],
    9: ['5. Adaptation',
        'Coastal road sections exposed to flooding will be raised and reinforced.',
        'Cleaner transport will reduce air pollution and improve public health in cities.'],
}


def make_pdf(path, n_pages=10):
    doc = fitz.open()
    for p in range(1, n_pages + 1):
        page = doc.new_page()
        lines = PAGES.get(p, [f'Page {p}', 'National circumstances and other context without transport content.'])
        y = 72
        for i, line in enumerate(lines):
            rect = fitz.Rect(72, y, 520, y + 120)
            page.insert_textbox(rect, line, fontsize=16 if i == 0 else 11, fontname='helv')
            y += 40 if i == 0 else 70
    doc.save(path)


def records():
    """What classified.to_dict('records') looks like after the pipeline."""
    T, F = 'True', 'False'
    return [
        dict(quote='reduce greenhouse gas emissions by 30% relative to the business-as-usual scenario by 2035', page=5,
             target=T, mitigation_measure=F, adaptation_measure=F, target_area='Overall mitigation target',
             ghg_target='GHG', conditionality='Unclear conditionality', category='--', purpose='--', instrument='--',
             asi='--', verification_status='Verified'),
        dict(quote='Net zero emissions will be reached by 2060.', page=5, target=T, mitigation_measure=F,
             adaptation_measure=F, target_area='Net zero target', ghg_target='GHG', conditionality='Unconditional',
             verification_status='Verified'),
        dict(quote='installation of approximately 5,000 charging points or stations', page=7, target=F,
             mitigation_measure=T, adaptation_measure=F, category='Electrification', purpose='--',
             instrument='--', asi='Improve', verification_status='Verified'),
        dict(quote='All new public buses procured will be electric by 2035, subject to international support.', page=7,
             target=T, mitigation_measure=T, adaptation_measure=F, target_area='Transport sector mitigation target',
             ghg_target='Non-GHG', conditionality='Conditional', category='Electrification', instrument='--',
             asi='Improve', verification_status='Verified'),
        dict(quote='A bus rapid transit corridor will be developed in the capital', page=7, target=F,
             mitigation_measure=T, adaptation_measure=F, category='Mode shift and demand management', asi='Shift',
             verification_status='Verified'),
        dict(quote='Coastal road sections exposed to flooding will be raised and reinforced.', page=8, target=F,
             mitigation_measure=F, adaptation_measure=T, category='Structural and Technical',
             instrument='Transport Infrastructure Resilience', verification_status='Unverified'),
    ]


class FakeLLM:
    model_name = 'fake'

    def invoke(self, messages):
        class R:
            content = """### Executive Summary
The NDC sets an economy-wide target against BAU for 2035 (p. 5) and relies on electric mobility in transport (p. 7).
### Key Findings
- Commits to reduce emissions by 30% against BAU by 2035, 12% unconditional (p. 5)
- Plans about 5,000 charging points (p. 7)
- Aims for all new public buses to be electric by 2035, conditional on support (p. 7)
### Economy-wide Targets
- Targets net zero by 2060 (p. 5)
### Transport Targets
- Electrifies the public bus fleet by 2035 (p. 7)
### Mitigation Actions
- Develops a BRT corridor in the capital (p. 7)
### Adaptation Actions
- Raises coastal road sections exposed to flooding (p. 9)
### Benefits
- None identified
"""
        return R()


def build_app():
    os.makedirs(BASE, exist_ok=True)
    app = Flask(__name__, template_folder=os.path.join(ROOT, 'src', 'templates'),
                static_folder=os.path.join(ROOT, 'src', 'static'))
    tasks = {}
    folder_fn = lambda tid: os.path.join(BASE, f'results_{tid}')
    review.init_review(app, tasks, folder_fn, FakeLLM())

    @app.route('/')
    def home():
        return '<a href="/results/demo-task-0001">results</a> | <a href="/validate/demo-task-0001">validate</a>'

    @app.route('/results/<task_id>')
    def results(task_id):
        return review.results_page(task_id)

    pdf = os.path.join(BASE, 'sample.pdf')
    make_pdf(pdf)
    if not os.path.exists(os.path.join(folder_fn(TASK_ID), 'review', 'pipeline.json')) or os.environ.get('RESET'):
        import shutil
        shutil.rmtree(folder_fn(TASK_ID), ignore_errors=True)
        with app.app_context():
            review.save_pipeline_snapshot(TASK_ID, pdf, 'Example_Updated_NDC_3.0.pdf', records(),
                                          {'summary_sections': {'executive_summary': 'Pipeline summary text.',
                                                                'key_findings': ['Converts 1,000 kt to 1 Mt CO2eq']},
                                           'summary_text': ''}, 10, 444.4, 182340)
    return app


app = build_app()

if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5055)), debug=False)
