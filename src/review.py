"""
Review layer for the Transport Policy Miner
===========================================

Adds four things on top of the extraction pipeline in flask_app.py:

1. A validation page (/validate/<task_id>) where analysts confirm, edit, flag,
   delete or add rows, with the PDF shown next to the rows.
2. A results page built from the rows (pipeline rows by default, reviewed rows
   once the analyst has pressed "Finish and rebuild results").
3. A social media card maker (/social-card/<task_id>).
4. A pre-filled GitHub "new issue" link for error reports.

Everything a review needs is written to disk in the task folder, so a review
does not depend on the in-memory ``processing_tasks`` dict:

    results_<task_id>/
        review/source.pdf        copy of the analysed PDF
        review/pipeline.json     rows + summary as produced by the pipeline
        review/validation.json   the latest saved review state
        output/*_validated_*     files rebuilt from the reviewed rows

The JSON format of pipeline.json / validation.json is documented in
docs/validation-json.md.
"""

import io
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from urllib.parse import quote as urlquote

from flask import (Blueprint, abort, jsonify, redirect,
                   render_template, request, send_file, url_for)
from werkzeug.utils import secure_filename

# Python 3.9 does not know .mjs; PDF.js is shipped as ES modules and browsers
# refuse module scripts served as application/octet-stream.
mimetypes.add_type('text/javascript', '.mjs')

SCHEMA = 'ndc-tracker-validation/1'
DOMAINS = ['targets', 'mitigation', 'adaptation', 'benefits']
GITHUB_REPO = os.environ.get('GITHUB_ISSUE_REPO', 'nicolas-becker/climate-policy-miner')
APP_VERSION = os.environ.get('APP_VERSION', 'unknown')

review_bp = Blueprint('review', __name__)

# Filled by init_review()
_ctx = {
    'processing_tasks': None,
    'results_folder': None,
    'llm': None,
}


def init_review(app, processing_tasks, results_folder_fn, llm=None):
    """Register the blueprint and give it access to the app's task store."""
    _ctx['processing_tasks'] = processing_tasks
    _ctx['results_folder'] = results_folder_fn
    _ctx['llm'] = llm
    app.register_blueprint(review_bp)

    @app.context_processor
    def _inject_issue_url():
        return {'report_issue_url': report_issue_url, 'app_version': APP_VERSION}


# ---------------------------------------------------------------------------
# Paths and storage
# ---------------------------------------------------------------------------

_TASK_ID_RE = re.compile(r'^[A-Za-z0-9-]{8,64}$')


def _check_task_id(task_id):
    if not _TASK_ID_RE.match(task_id or ''):
        abort(404)


def review_dir(task_id):
    _check_task_id(task_id)
    return os.path.join(_ctx['results_folder'](task_id), 'review')


def output_dir(task_id):
    return os.path.join(_ctx['results_folder'](task_id), 'output')


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def source_pdf_path(task_id):
    path = os.path.join(review_dir(task_id), 'source.pdf')
    if os.path.exists(path):
        return path
    task = (_ctx['processing_tasks'] or {}).get(task_id) or {}
    fallback = task.get('file_path')
    if fallback and os.path.exists(fallback):
        return fallback
    return None


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

_taxonomy_cache = {}


def load_taxonomy():
    if 'data' not in _taxonomy_cache:
        path = os.path.join(os.path.dirname(__file__), 'static', 'review', 'taxonomy', 'ndc_taxonomy.json')
        with open(path, 'r', encoding='utf-8') as f:
            _taxonomy_cache['data'] = json.load(f)
    return _taxonomy_cache['data']


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _is_true(value):
    if isinstance(value, str):
        return value.strip().lower() == 'true'
    return bool(value)


def _clean(value):
    """Pipeline uses '--' (and sometimes NaN/None) for 'not set'."""
    if value is None:
        return ''
    if isinstance(value, float) and value != value:  # NaN
        return ''
    text = str(value).strip()
    return '' if text in ('--', 'nan', 'None') else text


# Pipeline labels that are not taxonomy values (classification_utils.py).
# Fallbacks mean "nothing found" and are left empty for the analyst to fill.
_LABEL_FIXES = {
    'No concrete category': '',
    'No concrete purpose': '',
    'No concrete ASI category': '',
    'Freight transport shifting to rail of inland waterways': 'Freight transport shifting to rail or inland waterways',
}


def _label(value):
    text = _clean(value)
    return _LABEL_FIXES.get(text, text)


def _page(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def new_row(domain, **fields):
    row = {
        'id': uuid.uuid4().hex[:12],
        'domain': domain,
        'verbatim': '',
        'translation': '',
        'page': None,
        'source': 'pipeline',
        'status': 'pending',      # pending | confirmed | flagged
        'deleted': False,
        'edited': False,
        'note': '',
        'pipeline_verified': None,  # True/False when the pipeline checked the quote
        'pdf_found': None,          # True/False once the validation page searched the PDF
        'pdf_page': None,           # page where the validation page found the quote
        # targets
        'target_area': '', 'target_scope': '', 'ghg_target': '', 'target_type': '',
        'conditionality': '', 'target_year': '',
        # mitigation
        'category': '', 'purpose': '', 'instrument': '', 'asi': '',
        # adaptation
        'adaptation_category': '', 'adaptation_measure': '',
        # benefits
        'benefit': '',
    }
    row.update(fields)
    return row


def rows_from_pipeline(records):
    """Turn classified_quotes records into one review row per domain."""
    rows = []
    for rec in records or []:
        quote = _clean(rec.get('quote'))
        if not quote:
            continue
        base = {
            'verbatim': quote,
            'page': _page(rec.get('page')),
            'pipeline_verified': (rec.get('verification_status') == 'Verified')
            if rec.get('verification_status') else None,
        }
        if _is_true(rec.get('target')):
            rows.append(new_row('targets', **base,
                                target_area=_clean(rec.get('target_area')),
                                ghg_target=_clean(rec.get('ghg_target')),
                                conditionality=_clean(rec.get('conditionality'))))
        if _is_true(rec.get('mitigation_measure')):
            rows.append(new_row('mitigation', **base,
                                category=_label(rec.get('category')),
                                purpose=_label(rec.get('purpose')),
                                instrument=_label(rec.get('instrument')),
                                asi=_label(rec.get('asi'))))
        if _is_true(rec.get('adaptation_measure')):
            # postprocess_results() exports category/instrument as Category/Measure
            rows.append(new_row('adaptation', **base,
                                adaptation_category=_label(rec.get('category')),
                                adaptation_measure=_label(rec.get('instrument'))))
    return rows


def default_document(filename):
    return {
        'document_id': '', 'country_code': '', 'country': '', 'region': '',
        'type_of_document': 'NDC', 'document_name': '',
        'generation': '', 'version': '',
        'date': '', 'status': '', 'language': '', 'url': '',
        'note': '',            # exported as "Comments"
        'backend_status': '',  # free text, exported as "Backend status"
        # '' means "derive from rows"; 'yes'/'no' or a label overrides
        'flags': {k: '' for k in FLAG_KEYS},
    }


FLAG_KEYS = [
    'transport_content', 'transport_target', 'transport_measures',
    'summary_transport_target', 'contains_transport_mitigation_target',
    'ghg_transport_target_type', 'contains_transport_adaptation_target',
    'contains_transport_mitigation_measures', 'contains_transport_adaptation_measures',
    'contains_benefits', 'contains_just_transition',
]

FLAG_COLUMNS = {
    'contains_transport_mitigation_target': 'Contains transport mitigation target',
    'summary_transport_target': 'Summary transport target ',
    'ghg_transport_target_type': 'GHG transport target type',
    'contains_transport_adaptation_target': 'Contains transport adaptation target',
    'contains_transport_mitigation_measures': 'Contains transport mitigation measures',
    'contains_transport_adaptation_measures': 'Contains transport adaptation measures',
    'contains_benefits': 'Contains benefits',
    'contains_just_transition': 'Contains reference to just transition',
    'transport_content': 'Transport content',
    'transport_target': 'Transport target (mitigation or/and adaptation)',
    'transport_measures': 'Transport measures (mitigation or/and adaptation)',
}


def active_rows(rows):
    return [r for r in rows if not r.get('deleted')]


def derive_flags(rows):
    """Document-sheet flags as the database defines them, computed from rows.

    Mirrors deriveFlags() in static/review/validate.js. Keep both in sync.
    """
    rows = active_rows(rows)
    t_mit = [r for r in rows if r['domain'] == 'targets' and r.get('target_area') == 'Transport sector mitigation target']
    t_ada = [r for r in rows if r['domain'] == 'targets' and r.get('target_area') == 'Transport sector adaptation target']
    has_mit = any(r['domain'] == 'mitigation' for r in rows)
    has_ada = any(r['domain'] == 'adaptation' for r in rows)
    has_ben = any(r['domain'] == 'benefits' for r in rows)
    yn = lambda b: 'yes' if b else 'no'

    ghg_kinds = {r.get('ghg_target') for r in t_mit}
    if not t_mit:
        summary = 'No transport target'
    elif 'GHG' in ghg_kinds and 'Non-GHG' in ghg_kinds:
        summary = 'GHG and non-GHG target'
    elif 'GHG' in ghg_kinds:
        summary = 'GHG target'
    else:
        summary = 'non-GHG target'

    ghg_rows = [r for r in t_mit if r.get('ghg_target') == 'GHG']
    conds = {r.get('conditionality') for r in ghg_rows}
    if not t_mit:
        ghg_type = 'No transport target'
    elif not ghg_rows:
        ghg_type = 'No GHG transport target'
    elif conds == {'Unconditional'}:
        ghg_type = 'GHG unconditional only'
    elif conds == {'Conditional'}:
        ghg_type = 'GHG conditional only'
    elif {'Unconditional', 'Conditional'} <= conds:
        ghg_type = 'GHG unconditional and conditional'
    else:
        ghg_type = 'Conditionality unclear'

    target = bool(t_mit or t_ada)
    measures = has_mit or has_ada
    return {
        'transport_content': yn(target or measures or has_ben),
        'transport_target': yn(target),
        'transport_measures': yn(measures),
        'summary_transport_target': summary,
        'contains_transport_mitigation_target': yn(bool(t_mit)),
        'ghg_transport_target_type': ghg_type,
        'contains_transport_adaptation_target': yn(bool(t_ada)),
        'contains_transport_mitigation_measures': yn(has_mit),
        'contains_transport_adaptation_measures': yn(has_ada),
        'contains_benefits': yn(has_ben),
        'contains_just_transition': '',
    }


def effective_flags(state):
    derived = derive_flags(state.get('rows', []))
    overrides = (state.get('document') or {}).get('flags') or {}
    return {k: (overrides.get(k) or derived.get(k, '')) for k in FLAG_KEYS}


def load_countries():
    if 'countries' not in _taxonomy_cache:
        path = os.path.join(os.path.dirname(__file__), 'static', 'review', 'taxonomy', 'countries.json')
        with open(path, 'r', encoding='utf-8') as f:
            _taxonomy_cache['countries'] = json.load(f)
    return _taxonomy_cache['countries']


def _ordinal(n):
    n = int(n)
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def document_name(doc):
    """Database 'Document name' from type and generation, e.g. '3rd NDC'.
    Mirrors documentName() in static/review/validate.mjs."""
    kind = doc.get('type_of_document') or 'NDC'
    gen = str(doc.get('generation') or '').strip()
    if kind == 'NDC' and gen.isdigit():
        return f'{_ordinal(gen)} NDC'
    if kind == 'NDC':
        return ''
    return kind


def effective_document(doc):
    """Document details with the derived values filled in."""
    doc = dict(doc or {})
    code = str(doc.get('country_code') or '').strip().upper()
    doc['country_code'] = code
    hit = load_countries().get(code)
    if hit:
        doc['country'] = hit['country']
        doc['region'] = hit['region']
    doc['document_name'] = document_name(doc)
    return doc


def version_number(doc):
    """Database 'Version number' (e.g. 'NDC 3.0') from generation + version."""
    gen, ver = str(doc.get('generation') or '').strip(), str(doc.get('version') or '').strip()
    if not gen:
        return ''
    return f"{doc.get('type_of_document') or 'NDC'} {gen}.{ver or '0'}"


# ---------------------------------------------------------------------------
# Pipeline snapshot and review state
# ---------------------------------------------------------------------------

def save_pipeline_snapshot(task_id, file_path, filename, classified_records, summary_data,
                           document_pages=None, runtime_seconds=None, tokens_used=None):
    """Called at the end of process_document(). Never raises."""
    try:
        folder = review_dir(task_id)
        os.makedirs(folder, exist_ok=True)
        if file_path and os.path.exists(file_path):
            shutil.copyfile(file_path, os.path.join(folder, 'source.pdf'))
        snapshot = {
            'schema': SCHEMA,
            'task_id': task_id,
            'created': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'filename': filename,
            'document_pages': document_pages,
            'runtime_seconds': runtime_seconds,
            'tokens_used': tokens_used,
            'document': default_document(filename),
            'rows': rows_from_pipeline(classified_records),
            'summary': _summary_for_json(summary_data),
            'validated': False,
        }
        _write_json(os.path.join(folder, 'pipeline.json'), snapshot)
    except Exception as e:  # the review layer must never break a finished run
        logging.error(f"[REVIEW {task_id}] snapshot failed: {e}")


def _summary_for_json(summary_data):
    if not summary_data:
        return None
    return {
        'sections': summary_data.get('summary_sections') or {},
        'text': summary_data.get('summary_text') or '',
        'based_on': 'pipeline',
    }


def load_snapshot(task_id):
    """Pipeline snapshot from disk, or built on the fly for runs finished
    before this module existed (only while the task is still in memory)."""
    snap = _read_json(os.path.join(review_dir(task_id), 'pipeline.json'))
    if snap:
        return snap
    task = (_ctx['processing_tasks'] or {}).get(task_id)
    if not task or task.get('result') is None:
        return None
    records = (task.get('partial_results') or {}).get('classified_quotes') or []
    save_pipeline_snapshot(task_id, task.get('file_path'), task.get('original_filename'),
                           records, task.get('summary'), task.get('document_pages'),
                           task.get('total_runtime_seconds'), task.get('total_tokens_used'))
    return _read_json(os.path.join(review_dir(task_id), 'pipeline.json'))


def load_state(task_id):
    """Latest saved review state, else the pipeline snapshot."""
    saved = _read_json(os.path.join(review_dir(task_id), 'validation.json'))
    if saved:
        return saved
    return load_snapshot(task_id)


def normalise_state(data, fallback=None):
    """Accept our own format and tolerant variants from other tools."""
    fallback = fallback or {}
    if isinstance(data, list):
        data = {'rows': data}
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object or a list of rows')

    rows_in = data.get('rows')
    if rows_in is None:  # {"targets": [...], "mitigation": [...], ...}
        rows_in = []
        for d in DOMAINS:
            for r in data.get(d, []) or []:
                r = dict(r)
                r.setdefault('domain', d)
                rows_in.append(r)

    rows = [normalise_row(r) for r in rows_in]
    doc = dict(default_document(fallback.get('filename')))
    doc.update(fallback.get('document') or {})
    doc.update({k: v for k, v in (data.get('document') or {}).items() if v is not None})
    doc['flags'] = {**{k: '' for k in FLAG_KEYS}, **(doc.get('flags') or {})}

    out = dict(fallback)
    out.update({k: v for k, v in data.items() if k not in ('rows', 'document') + tuple(DOMAINS)})
    out['schema'] = SCHEMA
    out['rows'] = rows
    out['document'] = doc
    return out


_ROW_ALIASES = {
    'verbatim': ['verbatim', 'quote', 'content', 'text', 'verbatim_en'],
    'page': ['page', 'page_number', 'pagenumber', 'p'],
    'domain': ['domain', 'type', 'sheet', 'kind'],
    'translation': ['translation', 'translation_en'],
    'target_year': ['target_year', 'year'],
    'ghg_target': ['ghg_target', 'ghg'],
    'adaptation_measure': ['adaptation_measure', 'measure'],
    'benefit': ['benefit', 'type_of_benefit'],
}

_DOMAIN_ALIASES = {
    'target': 'targets', 'targets': 'targets',
    'mitigation': 'mitigation', 'mitigation_measure': 'mitigation', 'measure': 'mitigation',
    'adaptation': 'adaptation', 'adaptation_measure': 'adaptation',
    'benefit': 'benefits', 'benefits': 'benefits',
}


def normalise_row(raw):
    raw = {str(k).strip().lower().replace(' ', '_').replace('-', '_'): v for k, v in (raw or {}).items()}
    row = new_row('targets')
    if raw.get('id'):
        row['id'] = str(raw['id'])
    for key in row:
        if key == 'id':
            continue
        for alias in _ROW_ALIASES.get(key, [key]):
            if alias in raw and raw[alias] is not None:
                row[key] = raw[alias]
                break
    row['domain'] = _DOMAIN_ALIASES.get(str(row['domain']).strip().lower(), 'targets')
    row['page'] = _page(row['page'])
    row['pdf_page'] = _page(row['pdf_page'])
    # colleague's console used confirmed/flagged booleans
    if 'status' not in raw:
        if _is_true(raw.get('confirmed')):
            row['status'] = 'confirmed'
        elif _is_true(raw.get('flagged')):
            row['status'] = 'flagged'
    if row['status'] not in ('pending', 'confirmed', 'flagged'):
        row['status'] = 'pending'
    for key in ('deleted', 'edited'):
        row[key] = _is_true(row[key])
    for key in ('pipeline_verified', 'pdf_found'):
        if row[key] not in (None, ''):
            row[key] = _is_true(row[key])
        else:
            row[key] = None
    keep = ('page', 'pdf_page', 'deleted', 'edited', 'pipeline_verified', 'pdf_found')
    for key, value in list(row.items()):
        if key in keep:
            continue
        if value is None:
            row[key] = ''
        elif not isinstance(value, str):
            row[key] = str(value)
    row['source'] = row['source'] or 'import'
    return row


# ---------------------------------------------------------------------------
# Results view model
# ---------------------------------------------------------------------------

def _split_asi(value):
    parts = [p.strip().capitalize() for p in re.split(r'[,/;]| and ', value or '') if p.strip()]
    return [p for p in parts if p in ('Avoid', 'Shift', 'Improve')]


def results_context(task_id, state):
    rows = active_rows(state.get('rows', []))
    by = {d: [r for r in rows if r['domain'] == d] for d in DOMAINS}
    pages = state.get('document_pages') or 0
    try:
        pages = int(pages)
    except (TypeError, ValueError):
        pages = 0

    # mitigation by category with instruments
    cats = {}
    for r in by['mitigation']:
        c = r.get('category') or 'Not classified'
        cats.setdefault(c, {'label': c, 'n': 0, 'instruments': {}})
        cats[c]['n'] += 1
        inst = r.get('instrument') or 'No concrete instrument'
        cats[c]['instruments'][inst] = cats[c]['instruments'].get(inst, 0) + 1
    cat_list = sorted(cats.values(), key=lambda c: -c['n'])
    top = max([c['n'] for c in cat_list] or [1])
    for c in cat_list:
        c['pct'] = round(100 * c['n'] / top)
        c['instruments'] = sorted(c['instruments'].items(), key=lambda kv: -kv[1])

    asi = {'Avoid': 0, 'Shift': 0, 'Improve': 0}
    for r in by['mitigation']:
        for p in _split_asi(r.get('asi')):
            asi[p] += 1

    cond = {'Unconditional': 0, 'Conditional': 0, 'Unclear conditionality': 0}
    for r in by['targets']:
        c = r.get('conditionality') or 'Unclear conditionality'
        cond[c] = cond.get(c, 0) + 1

    # page strip: dominant domain per page
    page_hits = {}
    for r in rows:
        if r.get('page'):
            page_hits.setdefault(r['page'], []).append(r['domain'])
    max_page = max([pages] + list(page_hits.keys()) + [1])
    max_hits = max([len(v) for v in page_hits.values()] or [1])
    strip = []
    for p in range(1, max_page + 1):
        hits = page_hits.get(p, [])
        dom = max(set(hits), key=hits.count) if hits else ''
        strip.append({'page': p, 'n': len(hits), 'domain': dom,
                      'h': 6 if not hits else 25 + round(75 * len(hits) / max_hits)})

    def _found(r):
        # the check on the validation page is newer than the pipeline's own
        return r.get('pdf_found') if r.get('pdf_found') is not None else r.get('pipeline_verified')
    checked = [r for r in rows if _found(r) is not None]
    verified = sum(1 for r in checked if _found(r))

    status_counts = {s: sum(1 for r in rows if r['status'] == s) for s in ('pending', 'confirmed', 'flagged')}
    all_rows = state.get('rows', [])
    review_stats = {
        'confirmed': status_counts['confirmed'],
        'edited': sum(1 for r in rows if r.get('edited')),
        'removed': sum(1 for r in all_rows if r.get('deleted')),
        'added': sum(1 for r in rows if r.get('source') not in ('pipeline',)),
        'sources': sorted({source_label(r.get('source')) for r in rows if r.get('source') not in ('pipeline',)}),
        'missing': len(missing_fields(rows)),
    }

    doc = effective_document(state.get('document'))
    return {
        'task_id': task_id,
        'run': run_stats(state, pages),
        'state': state,
        'doc': doc,
        'version_number': version_number(doc),
        'flags': effective_flags(state),
        'validated': bool(state.get('validated')),
        'validated_at': state.get('validated_at'),
        'filename': state.get('filename') or 'document.pdf',
        'pages': pages or None,
        'counts': {d: len(by[d]) for d in DOMAINS},
        'rows': sorted(rows, key=lambda r: (r.get('page') or 10 ** 6)),
        'categories': cat_list,
        'asi': asi,
        'asi_total': sum(asi.values()),
        'conditionality': cond,
        'strip': strip,
        'quote_check': {'checked': len(checked), 'verified': verified},
        'summary': state.get('summary'),
        'review': review_stats,
        'has_pdf': source_pdf_path(task_id) is not None,
        'downloads': available_downloads(task_id, bool(state.get('validated'))),
    }


def run_stats(state, pages):
    """Monitoring figures of the pipeline run (empty for reviews started without one)."""
    secs = state.get('runtime_seconds')
    tokens = state.get('tokens_used')
    runtime = ''
    if isinstance(secs, (int, float)):
        m, s = divmod(int(round(secs)), 60)
        runtime = f'{m} min {s} s'
    return {
        'pages': pages or '',
        'runtime': runtime,
        'tokens': str(int(tokens)) if isinstance(tokens, (int, float)) else '',
    }


TRANSPORT_TARGET_AREAS = ('Transport sector mitigation target', 'Transport sector adaptation target')


def source_label(source):
    """Display name of a row source: rows from the pipeline are shown as 'miner'."""
    return 'miner' if source in ('pipeline', '', None) else source


REQUIRED = {
    'targets': ['target_area', 'conditionality', 'target_year'],
    'mitigation': ['category', 'instrument'],
    'adaptation': ['adaptation_category', 'adaptation_measure'],
    'benefits': ['benefit'],
}


def missing_fields(rows):
    """Rows that cannot be exported to the database as they are."""
    out = []
    for r in active_rows(rows):
        missing = [f for f in REQUIRED.get(r['domain'], []) if not str(r.get(f) or '').strip()]
        if not r.get('verbatim') or not r.get('page'):
            missing.append('verbatim/page')
        if missing:
            out.append((r, missing))
    return out


def available_downloads(task_id, validated):
    folder = output_dir(task_id)
    found = {}
    if not os.path.isdir(folder):
        return found
    for name in os.listdir(folder):
        is_val = '_validated' in name
        if validated != is_val:
            continue
        if name.endswith('.xlsx'):
            found['excel'] = name
        elif name.endswith('_highlighted.pdf') or name.endswith('_highlighted_validated.pdf'):
            found['pdf'] = name
        elif name.endswith('.txt'):
            found['summary'] = name
    return found


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

_PAGE_REF = re.compile(r'\((?:p|pp|page|pages)\.?\s*([\d][\d,\s\-–and]*)\)', re.IGNORECASE)


@review_bp.app_template_filter('pagelinks')
def pagelinks(text):
    """Turn '(p. 12, 40)' in summary text into links that open the page in the preview."""
    from markupsafe import Markup, escape
    out, last = [], 0
    raw = str(text or '')
    for m in _PAGE_REF.finditer(raw):
        out.append(str(escape(raw[last:m.start()])))
        pages = [p for p in re.findall(r'\d+', m.group(1))]
        links = ', '.join(f'<a href="#preview" data-goto="{p}">p. {p}</a>' for p in pages)
        out.append(f'<span class="pageref">({links})</span>')
        last = m.end()
    out.append(str(escape(raw[last:])))
    return Markup(''.join(out))


ROW_LABEL_FIELDS = {
    'targets': ['target_area', 'conditionality', 'target_year'],
    'mitigation': ['category', 'instrument', 'asi'],
    'adaptation': ['adaptation_category', 'adaptation_measure'],
    'benefits': ['benefit'],
}


@review_bp.app_template_filter('rowtag')
def rowtag(row):
    """(css class, label) of a row's tag: transport targets get their own colour."""
    d = row.get('domain')
    if d == 'targets' and row.get('target_area') in TRANSPORT_TARGET_AREAS:
        return ('transport-target', 'Transport target')
    return (d, {'targets': 'Target', 'mitigation': 'Mitigation', 'adaptation': 'Adaptation', 'benefits': 'Benefit'}.get(d, d))


@review_bp.app_template_filter('rowlabels')
def rowlabels(row):
    return ', '.join(str(row.get(f)) for f in ROW_LABEL_FIELDS.get(row.get('domain'), []) if row.get(f)) or 'Not classified'


# ---------------------------------------------------------------------------
# Issue link
# ---------------------------------------------------------------------------

def report_issue_url(task_id=None, page=None, filename=None, step=None, error=None):
    """Pre-filled GitHub issue. Everything in it becomes public."""
    lines = [
        '**What happened?**', '', '<!-- Describe what you did and what went wrong -->', '',
        '**What did you expect?**', '', '', '---',
        f'- Page: `{page or request.path}`',
        f'- Task ID: `{task_id or "n/a"}`',
        f'- Document: `{filename or "n/a"}`',
        f'- Step: `{step or "n/a"}`',
        f'- App version: `{APP_VERSION}`',
        f'- Time (UTC): `{datetime.utcnow().strftime("%Y-%m-%d %H:%M")}`',
    ]
    if error:
        lines += ['', '<details><summary>Error message</summary>', '', '```', str(error)[:1500], '```', '</details>']
    lines += ['', '_Issues in this repository are public. Do not paste confidential content._']
    title = f'Problem report: {step or page or request.path}'
    return (f'https://github.com/{GITHUB_REPO}/issues/new?labels=bug'
            f'&title={urlquote(title)}&body={urlquote(chr(10).join(lines))}')


# ---------------------------------------------------------------------------
# Routes: pages
# ---------------------------------------------------------------------------

@review_bp.route('/validate/<task_id>')
def validate_page(task_id):
    state = load_state(task_id)
    if not state:
        return render_template('review_missing.html', task_id=task_id), 404
    return render_template('validate.html', task_id=task_id,
                           filename=state.get('filename') or '',
                           has_pdf=source_pdf_path(task_id) is not None)


@review_bp.route('/validate', methods=['GET', 'POST'])
def validate_start():
    """Start a review without running the pipeline: PDF + optional JSON."""
    if request.method == 'GET':
        return render_template('validate_start.html')

    pdf = request.files.get('pdf_file')
    if not pdf or not pdf.filename.lower().endswith('.pdf'):
        return render_template('validate_start.html', error='Please choose a PDF file.'), 400

    task_id = 'r-' + str(uuid.uuid4())
    folder = review_dir(task_id)
    os.makedirs(folder, exist_ok=True)
    filename = secure_filename(pdf.filename)
    pdf.save(os.path.join(folder, 'source.pdf'))

    pages = None
    try:
        import fitz
        with fitz.open(os.path.join(folder, 'source.pdf')) as d:
            pages = len(d)
    except Exception:
        pass

    base = {'schema': SCHEMA, 'task_id': task_id, 'filename': filename, 'document_pages': pages,
            'document': default_document(filename), 'rows': [], 'summary': None, 'validated': False,
            'created': datetime.utcnow().isoformat(timespec='seconds') + 'Z'}

    upload = request.files.get('json_file')
    if upload and upload.filename:
        try:
            data = json.loads(upload.read().decode('utf-8'))
            state = normalise_state(data, base)
            source = request.form.get('source_name', '').strip()
            if source:
                for r in state['rows']:
                    if r['source'] in ('', 'import', 'pipeline'):
                        r['source'] = source
        except Exception as e:
            shutil.rmtree(_ctx['results_folder'](task_id), ignore_errors=True)
            return render_template('validate_start.html', error=f'Could not read the JSON file: {e}'), 400
    else:
        state = base

    _write_json(os.path.join(folder, 'pipeline.json'), state)
    return redirect(url_for('review.validate_page', task_id=task_id))


@review_bp.route('/social-card/<task_id>')
def social_card(task_id):
    state = load_state(task_id)
    if not state:
        return render_template('review_missing.html', task_id=task_id), 404
    ctx = results_context(task_id, state)
    card = {
        'country': ctx['doc'].get('country') or '',
        'document_name': ' '.join(x for x in [ctx['doc'].get('document_name'), ctx['version_number'] and f"({ctx['version_number']})"] if x) or ctx['filename'],
        'date': ctx['doc'].get('date') or '',
        'validated': ctx['validated'],
        'asi': ctx['asi'],
        'points': _card_points(ctx),
    }
    return render_template('social_card.html', ctx=ctx, card=card, task_id=task_id)


def _card_points(ctx):
    """Candidate key points: summary findings first, then confirmed target rows."""
    points = []
    summary = ctx.get('summary') or {}
    for text in (summary.get('sections') or {}).get('key_findings') or []:
        m = re.search(r'\(p+\.\s*([\d,\s]+)\)', text)
        points.append({'text': re.sub(r'\s*\(p+\.\s*[\d,\s]+\)', '', text).strip(),
                       'page': ('p. ' + m.group(1).strip()) if m else ''})
    for r in ctx['rows']:
        if r['domain'] == 'targets' and r['status'] == 'confirmed':
            points.append({'text': r['verbatim'][:140], 'page': f"p. {r['page']}" if r.get('page') else ''})
    return points[:8]


# ---------------------------------------------------------------------------
# Routes: data
# ---------------------------------------------------------------------------

@review_bp.route('/api/review/<task_id>/state', methods=['GET'])
def get_state(task_id):
    state = load_state(task_id)
    if not state:
        return jsonify({'error': 'not found'}), 404
    return jsonify(state)


@review_bp.route('/api/review/<task_id>/state', methods=['POST'])
def save_state(task_id):
    base = load_state(task_id)
    if not base:
        return jsonify({'error': 'not found'}), 404
    try:
        state = normalise_state(request.get_json(force=True), base)
    except Exception as e:
        return jsonify({'error': f'invalid state: {e}'}), 400
    state['saved_at'] = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    _write_json(os.path.join(review_dir(task_id), 'validation.json'), state)
    return jsonify({'ok': True, 'saved_at': state['saved_at']})


@review_bp.route('/api/review/<task_id>/taxonomy')
def taxonomy(task_id=None):
    return jsonify(load_taxonomy())


@review_bp.route('/review/<task_id>/pdf')
def serve_pdf(task_id):
    path = source_pdf_path(task_id)
    if not path:
        abort(404)
    return send_file(path, mimetype='application/pdf', conditional=True)


@review_bp.route('/review/<task_id>/download/<kind>')
def download(task_id, kind):
    state = load_state(task_id)
    if not state:
        abort(404)
    if kind == 'json':
        buf = io.BytesIO(json.dumps(state, ensure_ascii=False, indent=2).encode('utf-8'))
        stem = os.path.splitext(state.get('filename') or 'review')[0]
        return send_file(buf, mimetype='application/json', as_attachment=True,
                         download_name=f'{stem}_review.json')
    files = available_downloads(task_id, bool(state.get('validated')))
    name = files.get(kind)
    if not name:
        abort(404)
    return send_file(os.path.join(output_dir(task_id), name), as_attachment=True, download_name=name)


ZIP_README = """TRANSPORT POLICY MINER - RESULTS PACKAGE
==========================================
Task ID: {task_id}
Document: {filename}
Generated: {generated}
Validated: {validated}

CONTENTS:
- output/: Final analysis results (Excel, highlighted PDF, AI summary).
  Files ending in _validated were rebuilt from the reviewed rows.
- by-products/: Intermediate processing files (JSON, CSV)
- vector_stores/: Semantic search indexes
- logs/: Processing logs
- review/: Analysed PDF (source.pdf), rows as extracted (pipeline.json)
  and the latest review (validation.json, can be imported again on
  the validation page)

Each task has its own isolated folder to prevent conflicts.
"""


@review_bp.route('/review/<task_id>/download-all')
def download_all(task_id):
    """Whole task folder as one ZIP, like the /download route of the old results page."""
    import zipfile
    folder = _ctx['results_folder'](task_id)
    _check_task_id(task_id)
    if not os.path.isdir(folder):
        abort(404)
    state = load_state(task_id) or {}
    filename = state.get('filename') or 'document.pdf'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.endswith('.tmp'):
                    continue
                path = os.path.join(root, name)
                zf.write(path, os.path.relpath(path, folder))
        zf.writestr('README.txt', ZIP_README.format(
            task_id=task_id, filename=filename, generated=time.strftime('%Y-%m-%d %H:%M:%S'),
            validated=(state.get('validated_at') or '')[:10] if state.get('validated') else 'no'))
    buf.seek(0)
    stem = os.path.splitext(secure_filename(filename))[0] or 'transport_policy'
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                     download_name=f'{stem}_analysis.zip')


@review_bp.route('/api/review/<task_id>/finalize', methods=['POST'])
def finalize(task_id):
    base = load_state(task_id)
    if not base:
        return jsonify({'error': 'not found'}), 404
    try:
        state = normalise_state(request.get_json(force=True), base)
    except Exception as e:
        return jsonify({'error': f'invalid state: {e}'}), 400

    warnings = []
    folder = output_dir(task_id)
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(secure_filename(state.get('filename') or 'document'))[0] or 'document'
    # remove earlier validated outputs so downloads always match this run
    for name in os.listdir(folder):
        if '_validated' in name:
            os.remove(os.path.join(folder, name))

    write_excel(state, os.path.join(folder, f'{stem}_results_validated.xlsx'))

    pdf = source_pdf_path(task_id)
    if pdf:
        try:
            write_highlighted_pdf(pdf, state, os.path.join(folder, f'{stem}_highlighted_validated.pdf'))
        except Exception as e:
            warnings.append(f'Highlighted PDF could not be rebuilt: {e}')
    else:
        warnings.append('Source PDF not available, highlighted PDF not rebuilt.')

    if request.args.get('summary', '1') != '0' and _ctx['llm'] is not None and active_rows(state['rows']):
        try:
            state['summary'] = generate_review_summary(_ctx['llm'], state)
        except Exception as e:
            warnings.append(f'Summary could not be regenerated: {e}')
            if state.get('summary'):
                state['summary']['based_on'] = 'pipeline'
    write_summary_txt(state, os.path.join(folder, f'{stem}_ai_summary_validated.txt'))

    state['validated'] = True
    state['validated_at'] = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    state['saved_at'] = state['validated_at']
    _write_json(os.path.join(review_dir(task_id), 'validation.json'), state)
    return jsonify({'ok': True, 'warnings': warnings,
                    'redirect': url_for('review.results_page', task_id=task_id)})


@review_bp.route('/review/<task_id>/results')
def results_page(task_id):
    """Results from disk. flask_app's /results/<task_id> delegates here."""
    state = load_state(task_id)
    if not state:
        return render_template('review_missing.html', task_id=task_id), 404
    view = request.args.get('view')
    if view == 'pipeline':
        state = load_snapshot(task_id) or state
    return render_template('results.html', **results_context(task_id, state))


# ---------------------------------------------------------------------------
# Rebuilding outputs from reviewed rows
# ---------------------------------------------------------------------------

DOC_PREFIX = ['Document ID', 'Country Code', 'Country', 'Type of Document ', 'Document name',
              'Version number', 'Status', 'Date']

MODE_COLUMNS = ['Mode: Not defined', 'Informal transport', 'Active mobility', 'Walking', 'Cycling', 'Road',
                'Two-/Three-wheelers', 'Cars', 'Private cars', 'Taxis', 'Truck', 'Bus', 'Rail', 'Heavy rail',
                'High-speed rail', 'Transit rail', 'Water', 'Coastal shipping', 'Inland shipping',
                'International maritime', 'Aviation', 'Domestic aviation', 'International aviation',
                'Geography: Not defined', 'Urban', 'Rural', 'Inter-city', 'Activity type']


def _doc_values(doc):
    return {
        'Document ID': doc.get('document_id', ''), 'Country Code': doc.get('country_code', ''),
        'Country': doc.get('country', ''), 'Type of Document ': doc.get('type_of_document', ''),
        'Document name': doc.get('document_name', ''), 'Version number': version_number(doc),
        'Status': doc.get('status', ''), 'Date': doc.get('date', ''), 'Region': doc.get('region', ''),
    }


def write_excel(state, path):
    """Workbook whose sheets use the column headers of the tracker database."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    doc = effective_document(state.get('document'))
    dv = _doc_values(doc)
    flags = effective_flags(state)
    rows = active_rows(state.get('rows', []))

    wb = Workbook()
    ws = wb.active
    ws.title = 'Document'
    flag_order = ['contains_transport_mitigation_target', 'summary_transport_target', 'ghg_transport_target_type',
                  'contains_transport_adaptation_target', 'contains_transport_mitigation_measures',
                  'contains_transport_adaptation_measures', 'contains_benefits', 'contains_just_transition',
                  'transport_content', 'transport_target', 'transport_measures']
    doc_row = [
        ('Document ID', dv['Document ID']), ('Country Code', dv['Country Code']), ('Country', dv['Country']),
        ('Check active', ''), ('Type of document', doc.get('type_of_document', '')),
        ('Document name', dv['Document name']), ('Last submission', ''),  # filled from the Country sheet
        ('Version number', dv['Version number']), ('Date', dv['Date']), ('Status', dv['Status']),
    ] + [(FLAG_COLUMNS[k], flags.get(k, '')) for k in flag_order] + [
        ('Comments', doc.get('note', '')), ('Backend status', doc.get('backend_status', '')),
        ('URL', doc.get('url', '')),
    ]
    ws.append([c for c, _ in doc_row])
    ws.append([v for _, v in doc_row])

    def sheet(title, cols, items):
        s = wb.create_sheet(title)
        s.append(cols)
        for it in items:
            s.append([it.get(c, '') for c in cols])
        return s

    sheet('Targets', DOC_PREFIX + ['Target area', 'Target scope', 'GHG target?', 'Target type', 'Conditionality',
                                   'Target Year', 'Content', 'Page Number'],
          [{**dv, 'Target area': r['target_area'], 'Target scope': r['target_scope'], 'GHG target?': r['ghg_target'],
            'Target type': r['target_type'], 'Conditionality': r['conditionality'],
            'Target Year': r['target_year'], 'Content': r['verbatim'], 'Page Number': r['page']}
           for r in rows if r['domain'] == 'targets'])
    sheet('Mitigation', ['Document ID', 'Country Code', 'Country', 'Region', 'Type of Document ', 'Document name',
                         'Version number', 'Status', 'Date', 'Category', 'Purpose', 'Instrument', 'Quote', 'A-S-I',
                         'Page Number'] + MODE_COLUMNS + ['Status of measure'],
          [{**dv, 'Category': r['category'], 'Purpose': r['purpose'], 'Instrument': r['instrument'],
            'Quote': r['verbatim'], 'A-S-I': r['asi'], 'Page Number': r['page']}
           for r in rows if r['domain'] == 'mitigation'])
    sheet('Adaptation', DOC_PREFIX + ['Category', 'Measure', 'Quote', 'Page Number'] + MODE_COLUMNS,
          [{**dv, 'Category': r['adaptation_category'], 'Measure': r['adaptation_measure'],
            'Quote': r['verbatim'], 'Page Number': r['page']}
           for r in rows if r['domain'] == 'adaptation'])
    sheet('Benefits', DOC_PREFIX[:-1] + ['Type of benefit', 'Quote', 'Page Number'],
          [{**dv, 'Type of benefit': r['benefit'], 'Quote': r['verbatim'], 'Page Number': r['page']}
           for r in rows if r['domain'] == 'benefits'])

    # Review trail: not part of the database, useful for checking
    s = sheet('Review', ['Row ID', 'Sheet', 'Status', 'Source', 'Edited', 'Page Number', 'Verbatim',
                         'English translation', 'Note', 'Missing for export'],
              [])
    missing = {r['id']: m for r, m in missing_fields(rows)}
    for r in rows:
        s.append([r['id'], r['domain'], r['status'], source_label(r['source']), 'yes' if r.get('edited') else '',
                  r['page'], r['verbatim'], r['translation'], r['note'], ', '.join(missing.get(r['id'], []))])
    s.append([])
    s.append(['Document language', doc.get('language', '')])

    bold = Font(bold=True)
    head = PatternFill('solid', fgColor='E0F6FE')
    for s in wb.worksheets:
        for cell in s[1]:
            cell.font = bold
            cell.fill = head
        for col in s.columns:
            width = max(len(str(c.value or '')) for c in list(col)[:200]) + 2
            s.column_dimensions[col[0].column_letter].width = min(max(width, 10), 60)
        s.freeze_panes = 'A2'
    wb.save(path)


def write_highlighted_pdf(src, state, path):
    import pandas as pd
    try:
        from .general_utils import create_highlighted_pdf
    except ImportError:
        from general_utils import create_highlighted_pdf
    rows = active_rows(state.get('rows', []))
    df = pd.DataFrame([{
        'quote': r['verbatim'],
        'target': 'True' if r['domain'] == 'targets' else 'False',
        'measure': 'True' if r['domain'] in ('mitigation', 'adaptation') else 'False',
    } for r in rows if r.get('verbatim')])
    if df.empty:
        shutil.copyfile(src, path)
        return
    df = df.drop_duplicates(subset=['quote']).reset_index(drop=True)
    create_highlighted_pdf(src, quotes=df['quote'], output_path=path, df=df)


def write_summary_txt(state, path):
    doc = effective_document(state.get('document'))
    summary = state.get('summary') or {}
    sections = summary.get('sections') or {}
    rows = active_rows(state.get('rows', []))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('TRANSPORT POLICY ANALYSIS, AI SUMMARY OF REVIEWED ROWS\n')
        f.write('=' * 54 + '\n\n')
        f.write(f"Document: {doc.get('document_name') or state.get('filename')}\n")
        f.write(f"Country: {doc.get('country', '')} {('(' + doc['country_code'] + ')') if doc.get('country_code') else ''}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Summary based on: {summary.get('based_on', 'n/a')} rows\n\n")
        for d in DOMAINS:
            f.write(f"{d.capitalize()}: {sum(1 for r in rows if r['domain'] == d)} rows\n")
        f.write('\n')
        if sections.get('executive_summary'):
            f.write('EXECUTIVE SUMMARY\n' + sections['executive_summary'] + '\n\n')
        for key, title in SUMMARY_SECTIONS[1:]:
            items = sections.get(key) or []
            if items:
                f.write(title.upper() + '\n')
                for it in items:
                    f.write(f'- {it}\n')
                f.write('\n')


SUMMARY_SECTIONS = [
    ('executive_summary', 'Executive Summary'),
    ('key_findings', 'Key Findings'),
    ('economy_wide_targets', 'Economy-wide Targets'),
    ('transport_targets', 'Transport Targets'),
    ('mitigation_actions', 'Mitigation Actions'),
    ('adaptation_actions', 'Adaptation Actions'),
    ('benefits', 'Benefits'),
]


def generate_review_summary(llm, state):
    """Summary of the reviewed rows. Differs from the pipeline summary in two
    ways: each bullet cites its pages, and figures are kept as written."""
    rows = active_rows(state.get('rows', []))
    doc = effective_document(state.get('document'))

    def block(domain, cond=None):
        sel = [r for r in rows if r['domain'] == domain and (cond is None or cond(r))]
        return '\n'.join(f"- (p. {r.get('page') or '?'}) {r['verbatim']}" for r in sel) or '- none'

    economy = lambda r: r.get('target_area') in ('Net zero target', 'Overall mitigation target', 'Energy sector target')
    prompt = f"""You are a climate policy analyst specialised in transport. Summarise the reviewed extraction
results of this document for a policy audience.

Document: {doc.get('document_name') or state.get('filename')}
Country: {doc.get('country') or 'unknown'}

ECONOMY-WIDE TARGETS:
{block('targets', economy)}

TRANSPORT TARGETS:
{block('targets', lambda r: not economy(r))}

MITIGATION MEASURES:
{block('mitigation')}

ADAPTATION MEASURES:
{block('adaptation')}

BENEFITS:
{block('benefits')}

Answer in exactly this format:
### Executive Summary
[2-3 sentences]
### Key Findings
- [finding] (p. N)
### Economy-wide Targets
- ...
### Transport Targets
- ...
### Mitigation Actions
- ...
### Adaptation Actions
- ...
### Benefits
- ...

Rules:
- End every bullet with the page numbers of the quotes it is based on, e.g. "(p. 12)" or "(p. 12, 40)".
- Keep every figure and unit exactly as written in the quotes. Do not convert units.
- Start bullets with a verb describing what the country does or plans.
- If a section has no quotes, write "- None identified".
- Do not add information that is not in the quotes.
"""
    try:
        from langchain.schema import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
    except ImportError:
        response = llm.invoke(prompt)
    text = getattr(response, 'content', str(response))
    return {'sections': parse_summary(text), 'text': text, 'based_on': 'reviewed'}


def parse_summary(text):
    sections = {k: ([] if k != 'executive_summary' else '') for k, _ in SUMMARY_SECTIONS}
    current = None
    titles = {t.lower(): k for k, t in SUMMARY_SECTIONS}
    for line in (text or '').splitlines():
        s = line.strip()
        if s.startswith('#'):
            current = titles.get(s.lstrip('#').strip().lower())
            continue
        if not s or current is None:
            continue
        if current == 'executive_summary':
            sections[current] = (sections[current] + ' ' + s).strip()
        elif s[:2] in ('- ', '* ', '• '):
            item = s[2:].strip()
            if item.lower().rstrip('.') != 'none identified':
                sections[current].append(item)
    return sections
