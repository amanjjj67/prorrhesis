'use strict';

/* PRORRHESIS - the local web app. No CDN, no build step: every chart below is
   plain SVG written by this file. */

const SCREENS = [
  { id: 'fixtures', label: 'Fixtures' },
  { id: 'arena', label: 'Arena' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'simulator', label: 'Simulator' },
  { id: 'ledger', label: 'Ledger' },
  { id: 'health', label: 'Data Health' },
];

const state = {
  screen: 'fixtures',
  league: 'E0',
  days: 7,
  leagues: [],
  match: null,
  calibrationModel: 'ensemble_cal',
  timeMachineDate: '',
};

const view = document.getElementById('view');
const statusEl = document.getElementById('status');

/* -- small helpers -------------------------------------------------------- */

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const key in (attrs || {})) {
    if (key === 'class') node.className = attrs[key];
    else if (key === 'text') node.textContent = attrs[key];
    else if (key === 'html') node.innerHTML = attrs[key];
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), attrs[key]);
    else if (attrs[key] !== null && attrs[key] !== undefined) node.setAttribute(key, attrs[key]);
  }
  (children || []).forEach((child) => {
    if (child === null || child === undefined) return;
    node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  });
  return node;
}

function pct(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return (100 * value).toFixed(digits === undefined ? 1 : digits) + '%';
}

function num(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return Number(value).toFixed(digits === undefined ? 4 : digits);
}

function title(text) {
  return { title: text };
}

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try {
    body = await response.json();
  } catch (err) {
    body = null;
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : response.statusText;
    throw new Error(detail);
  }
  return body;
}

function setStatus(text) {
  statusEl.textContent = text || '';
}

function showError(message, retry) {
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'error-box' }, [
    el('strong', { text: 'Something went wrong. ' }),
    el('span', { text: message }),
  ]));
  if (retry) {
    view.appendChild(el('div', { class: 'row', style: 'margin-top:12px' }, [
      el('button', { class: 'icon-btn', text: 'Try again', onclick: retry }),
    ]));
  }
}

function showEmpty(message) {
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty', text: message }));
}

/* -- charts --------------------------------------------------------------- */

function probabilityBar(pH, pD, pA, opts) {
  const options = opts || {};
  const h = Number.isFinite(pH) ? pH : 0;
  const d = Number.isFinite(pD) ? pD : 0;
  const a = Number.isFinite(pA) ? pA : 0;
  const bar = el('div', { class: 'pbar', role: 'img', 'aria-label':
    `Home ${pct(h)}, draw ${pct(d)}, away ${pct(a)}` });
  [['b-h', h, 'H'], ['b-d', d, 'D'], ['b-a', a, 'A']].forEach(([cls, value, key]) => {
    const span = el('span', { class: cls, title: `${key}: ${pct(value)}` });
    span.style.width = Math.max(0, Math.min(100, value * 100)) + '%';
    if (value > 0.16) span.textContent = pct(value, 0);
    bar.appendChild(span);
  });
  if (options.legend) {
    return el('div', {}, [
      bar,
      el('div', { class: 'legend', style: 'margin-top:4px' }, [
        el('span', {}, [el('span', { class: 'swatch', style: 'background:var(--home)' }), 'Home ' + pct(h)]),
        el('span', {}, [el('span', { class: 'swatch', style: 'background:var(--draw)' }), 'Draw ' + pct(d)]),
        el('span', {}, [el('span', { class: 'swatch', style: 'background:var(--away)' }), 'Away ' + pct(a)]),
      ]),
    ]);
  }
  return bar;
}

function svgRoot(width, height) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  node.setAttribute('viewBox', `0 0 ${width} ${height}`);
  node.setAttribute('width', '100%');
  node.setAttribute('height', height);
  node.setAttribute('role', 'img');
  return node;
}

function svgEl(tag, attrs, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const key in (attrs || {})) node.setAttribute(key, attrs[key]);
  if (text !== undefined) node.textContent = text;
  return node;
}

/* A horizontal bar chart: rows of {label, value, note}. Lower is better for
   log loss, so the bars are drawn from the right. */
function barChart(rows, opts) {
  const options = opts || {};
  const width = options.width || 640;
  const rowHeight = 26;
  const height = Math.max(40, rows.length * rowHeight + 10);
  const left = options.left || 150;
  const right = 90;
  const inner = width - left - right;
  const svg = svgRoot(width, height);
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);

  rows.forEach((row, index) => {
    const y = index * rowHeight + 4;
    const length = (Math.abs(row.value) / max) * inner;
    svg.appendChild(svgEl('text', {
      x: left - 8, y: y + 15, 'text-anchor': 'end', 'font-size': 12,
      fill: 'currentColor', opacity: 0.8,
    }, row.label));
    svg.appendChild(svgEl('rect', {
      x: left, y: y + 4, width: Math.max(length, 1), height: rowHeight - 12,
      rx: 3, fill: row.color || 'var(--accent)',
    }));
    svg.appendChild(svgEl('text', {
      x: left + length + 6, y: y + 15, 'font-size': 12, fill: 'currentColor',
    }, row.note || num(row.value)));
  });
  return svg;
}

/* The scoreline heatmap: goals 0..6 on both axes. */
function heatmap(grid, maxGoals) {
  const limit = maxGoals || 7;
  const table = el('table', { class: 'heatmap' });
  const head = el('tr', {}, [el('th', { text: '' })]);
  for (let g = 0; g < limit; g++) head.appendChild(el('th', { text: String(g) }));
  table.appendChild(head);

  for (let h = 0; h < limit; h++) {
    const row = el('tr', {}, [el('th', { text: String(h) })]);
    for (let a = 0; a < limit; a++) {
      const value = grid && grid[h] && grid[h][a] !== undefined ? grid[h][a] : 0;
      const intensity = Math.min(1, Math.pow(value, 0.45));
      const cell = el('td', {}, [el('div', {
        title: `${h}-${a}: ${pct(value, 2)}`,
        style: `background: rgba(47,111,179,${0.08 + intensity * 0.92});` +
               (intensity > 0.6 ? 'color:#fff' : 'color:inherit'),
      }, value > 0.005 ? pct(value, 1) : '')]);
      row.appendChild(cell);
    }
    table.appendChild(row);
  }
  return table;
}

/* The reliability diagram: observed rate against predicted probability. */
function reliabilityChart(bins) {
  const width = 340;
  const height = 300;
  const pad = 34;
  const inner = width - pad * 2;
  const svg = svgRoot(width, height);
  svg.appendChild(svgEl('rect', {
    x: pad, y: pad, width: inner, height: inner, fill: 'none',
    stroke: 'currentColor', opacity: 0.25,
  }));
  svg.appendChild(svgEl('line', {
    x1: pad, y1: pad + inner, x2: pad + inner, y2: pad,
    stroke: 'currentColor', 'stroke-dasharray': '4 4', opacity: 0.5,
  }));
  for (let i = 0; i <= 4; i++) {
    const t = i / 4;
    svg.appendChild(svgEl('text', {
      x: pad + t * inner, y: height - 10, 'text-anchor': 'middle',
      'font-size': 11, fill: 'currentColor', opacity: 0.7,
    }, pct(t, 0)));
    svg.appendChild(svgEl('text', {
      x: 10, y: pad + inner - t * inner + 4, 'font-size': 11,
      fill: 'currentColor', opacity: 0.7,
    }, pct(t, 0)));
  }
  svg.appendChild(svgEl('text', {
    x: pad + inner / 2, y: height - 1, 'text-anchor': 'middle', 'font-size': 11,
    fill: 'currentColor', opacity: 0.7,
  }, 'predicted'));

  const points = (bins || []).filter((b) => b.count > 0);
  if (points.length) {
    const path = points.map((b, index) => {
      const x = pad + b.mean_predicted * inner;
      const y = pad + inner - b.observed_rate * inner;
      return (index === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1);
    }).join(' ');
    svg.appendChild(svgEl('path', {
      d: path, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2,
    }));
    points.forEach((b) => {
      svg.appendChild(svgEl('circle', {
        cx: pad + b.mean_predicted * inner, cy: pad + inner - b.observed_rate * inner,
        r: 3.5, fill: 'var(--accent)',
      }, ));
      const titleNode = svgEl('title', {},
        `predicted ${pct(b.mean_predicted)} -> observed ${pct(b.observed_rate)} (${b.count} rows)`);
      svg.lastChild.appendChild(titleNode);
    });
  }
  return svg;
}

/* -- navigation ----------------------------------------------------------- */

function renderTabs() {
  const tabs = document.getElementById('tabs');
  tabs.innerHTML = '';
  SCREENS.forEach((screen) => {
    tabs.appendChild(el('button', {
      type: 'button',
      text: screen.label,
      'aria-current': state.screen === screen.id ? 'page' : null,
      onclick: () => go(screen.id),
    }));
  });
}

function go(screen) {
  state.screen = screen;
  renderTabs();
  window.scrollTo(0, 0);
  route();
}

function route() {
  const renderers = {
    fixtures: renderFixtures,
    match: renderMatch,
    arena: renderArena,
    calibration: renderCalibration,
    simulator: renderSimulator,
    ledger: renderLedger,
    health: renderHealth,
  };
  const renderer = renderers[state.screen] || renderFixtures;
  renderer();
}

/* -- fixtures ------------------------------------------------------------- */

async function renderFixtures() {
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty', text: 'Loading fixtures...' }));
  try {
    if (state.leagues.length === 0) {
      const data = await api('/api/leagues');
      state.leagues = data.leagues || [];
    }
    if (!state.leagues.length) {
      showEmpty('No leagues are configured yet.');
      return;
    }
    if (!state.leagues.some((l) => l.code === state.league)) {
      state.league = state.leagues[0].code;
    }
    const data = await api(`/api/fixtures?league=${encodeURIComponent(state.league)}&days=${state.days}`);

    view.innerHTML = '';
    const panel = el('div', { class: 'panel' });
    panel.appendChild(el('h2', { text: 'Upcoming fixtures' }));

    const tabs = el('div', { class: 'row', role: 'tablist' });
    state.leagues.forEach((league) => {
      tabs.appendChild(el('button', {
        class: 'icon-btn',
        type: 'button',
        text: league.code,
        'aria-current': league.code === state.league ? 'true' : null,
        onclick: () => { state.league = league.code; renderFixtures(); },
      }));
    });
    panel.appendChild(tabs);

    panel.appendChild(el('div', { class: 'row', style: 'margin:10px 0' }, [
      el('label', { class: 'small muted', text: 'days ahead' }),
      el('select', {
        onchange: (event) => { state.days = Number(event.target.value); renderFixtures(); },
      }, [3, 7, 14, 30].map((n) => el('option', {
        value: String(n), text: String(n),
        selected: n === state.days ? 'selected' : null,
      }))),
      el('span', { class: 'small muted', text:
        `${data.count || 0} fixture(s)` + (data.method ? ` - ensemble: ${data.method}` : '') }),
    ]));
    view.appendChild(panel);

    if (!data.fixtures || !data.fixtures.length) {
      view.appendChild(el('div', { class: 'empty' }, [
        el('div', { text: 'No upcoming fixtures in this window.' }),
        el('div', { class: 'small muted', style: 'margin-top:6px',
          text: 'The remaining fixtures are derived from the season when the fixtures feed has none.' }),
      ]));
      return;
    }

    const list = el('div', { class: 'panel' });
    data.fixtures.forEach((fixture) => {
      const row = el('div', {
        class: 'fixture', tabindex: '0', role: 'button',
        title: 'Open the match detail',
      });
      row.appendChild(el('div', { class: 'date' }, [
        el('div', { text: fixture.date }),
        fixture.estimated ? el('div', { class: 'small muted', text: 'estimated' }) : null,
      ]));
      row.appendChild(el('div', { class: 'teams' }, [
        el('span', { text: fixture.home }),
        el('span', { class: 'sep', text: 'v' }),
        el('span', { text: fixture.away }),
      ]));
      row.appendChild(probabilityBar(fixture.pH, fixture.pD, fixture.pA));
      row.appendChild(el('div', {}, [
        fixture.favourite ? el('span', {
          class: 'chip',
          text: `${fixture.favourite.outcome} ${pct(fixture.favourite.probability, 0)}`,
          title: `Most likely outcome: ${fixture.favourite.outcome}`,
        }) : null,
        fixture.top_scoreline ? el('div', { class: 'small muted', text:
          `${fixture.top_scoreline.home_goals}-${fixture.top_scoreline.away_goals}` }) : null,
      ]));
      row.appendChild(el('div', { title: fixture.disagree ? 'The models disagree' : 'The models agree' }, [
        fixture.disagree ? el('span', { class: 'dot' }) : el('span', { class: 'small muted', text: '' }),
      ]));

      const open = () => openMatch(fixture);
      row.addEventListener('click', open);
      row.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
      });
      list.appendChild(row);
    });
    view.appendChild(list);
  } catch (err) {
    showError(err.message, renderFixtures);
  }
}

/* -- match detail --------------------------------------------------------- */

function openMatch(fixture) {
  state.match = {
    league: fixture.league, date: fixture.date,
    home: fixture.home, away: fixture.away,
    home_id: fixture.home_id, away_id: fixture.away_id,
  };
  go('match');
}

async function renderMatch() {
  const match = state.match;
  if (!match) {
    showEmpty('Pick a fixture to see the detail.');
    return;
  }
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty', text: 'Loading match...' }));
  try {
    const query = new URLSearchParams({
      league: match.league, date: match.date, home: match.home, away: match.away,
    });
    const data = await api('/api/match?' + query.toString());
    view.innerHTML = '';

    const head = el('div', { class: 'panel' });
    head.appendChild(el('div', { class: 'row', style: 'justify-content:space-between' }, [
      el('h2', { text: `${match.home} v ${match.away}` }),
      el('button', { class: 'icon-btn', text: 'Back to fixtures', onclick: () => go('fixtures') }),
    ]));
    head.appendChild(el('div', { class: 'small muted', text: `${match.league} - ${match.date}` }));
    view.appendChild(head);

    if (data.error) {
      view.appendChild(el('div', { class: 'error-box', text: data.error }));
      return;
    }

    const grid = el('div', { class: 'grid2' });

    const bar = el('div', { class: 'panel' });
    bar.appendChild(el('h2', { text: 'Ensemble' }));
    bar.appendChild(probabilityBar(data.ensemble.pH, data.ensemble.pD, data.ensemble.pA, { legend: true }));
    bar.appendChild(el('div', { class: 'small muted', style: 'margin-top:6px', text:
      `method: ${data.method || 'equal'}` + (data.calibration_method ? ` - calibration: ${data.calibration_method}` : '') }));
    if (data.uncertainty) {
      bar.appendChild(el('div', { class: 'notice warn', style: 'margin-top:10px', text: data.uncertainty }));
    }
    (data.reasons || []).forEach((reason) => {
      bar.appendChild(el('div', { class: 'small', style: 'margin-top:6px', text: '- ' + reason }));
    });
    grid.appendChild(bar);

    const heat = el('div', { class: 'panel' });
    heat.appendChild(el('h2', { text: 'Scoreline heatmap' }));
    if (data.grid) {
      heat.appendChild(heatmap(data.grid, 7));
      heat.appendChild(el('div', { class: 'small muted', style: 'margin-top:6px', text:
        'rows are home goals, columns are away goals' }));
      if (data.top_scoreline) {
        heat.appendChild(el('div', { class: 'small', style: 'margin-top:6px', text:
          `Most likely: ${data.top_scoreline.home_goals}-${data.top_scoreline.away_goals} (${pct(data.top_scoreline.probability)})` }));
      }
    } else {
      heat.appendChild(el('div', { class: 'empty', text: 'No goal grid for this match.' }));
    }
    grid.appendChild(heat);
    view.appendChild(grid);

    const table = el('div', { class: 'panel' });
    table.appendChild(el('h2', { text: 'Every model' }));
    const rows = Object.keys(data.models || {}).sort().map((name) => {
      const triple = data.models[name];
      return { name: name, triple: triple };
    });
    const body = el('tbody');
    rows.forEach(({ name, triple }) => {
      body.appendChild(el('tr', {}, [
        el('td', { text: name }),
        el('td', { text: triple.status === 'ok' ? pct(triple.pH) : 'abstain' }),
        el('td', { text: triple.status === 'ok' ? pct(triple.pD) : '-' }),
        el('td', { text: triple.status === 'ok' ? pct(triple.pA) : '-' }),
        el('td', { text: triple.status }),
      ]));
    });
    body.appendChild(el('tr', {}, [
      el('td', { text: 'ensemble' }),
      el('td', { text: pct(data.ensemble.pH) }),
      el('td', { text: pct(data.ensemble.pD) }),
      el('td', { text: pct(data.ensemble.pA) }),
      el('td', { text: 'ok' }),
    ]));
    if (data.market_open) {
      body.appendChild(el('tr', {}, [
        el('td', { text: 'market (open)' }),
        el('td', { text: pct(data.market_open.pH) }),
        el('td', { text: pct(data.market_open.pD) }),
        el('td', { text: pct(data.market_open.pA) }),
        el('td', { text: 'ok' }),
      ]));
    }
    table.appendChild(el('table', {}, [
      el('thead', {}, [el('tr', {}, [
        el('th', { text: 'model' }), el('th', { text: 'Home' }),
        el('th', { text: 'Draw' }), el('th', { text: 'Away' }), el('th', { text: 'status' }),
      ])]),
      body,
    ]));
    view.appendChild(table);

    view.appendChild(whatifPanel(match));
  } catch (err) {
    showError(err.message, renderMatch);
  }
}

function whatifPanel(match) {
  const panel = el('div', { class: 'panel' });
  panel.appendChild(el('h2', { text: 'What-if' }));
  panel.appendChild(el('div', { class: 'small muted', text:
    'Nudge one side and watch the numbers move. This is a manual judgement and is never stored.' }));

  const attack = el('input', { type: 'range', min: '-0.25', max: '0.25', step: '0.01', value: '0' });
  const defence = el('input', { type: 'range', min: '-0.25', max: '0.25', step: '0.01', value: '0' });
  const attackOut = el('output', { text: '0.00' });
  const defenceOut = el('output', { text: '0.00' });
  const result = el('div', { style: 'margin-top:10px' });

  const update = () => {
    attackOut.textContent = Number(attack.value).toFixed(2);
    defenceOut.textContent = Number(defence.value).toFixed(2);
  };
  attack.addEventListener('input', update);
  defence.addEventListener('input', update);

  const send = async () => {
    try {
      const data = await api('/api/whatif', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          league: match.league, home_id: match.home_id, away_id: match.away_id,
          attack: Number(attack.value), defence: Number(defence.value), side: 'home',
        }),
      });
      result.innerHTML = '';
      if (data.error) {
        result.appendChild(el('div', { class: 'error-box', text: data.error }));
        return;
      }
      result.appendChild(probabilityBar(data.pH, data.pD, data.pA, { legend: true }));
      result.appendChild(el('div', { class: 'small muted', style: 'margin-top:6px',
        text: `expected goals ${num(data.lambdas[0], 2)} v ${num(data.lambdas[1], 2)}` }));
    } catch (err) {
      result.innerHTML = '';
      result.appendChild(el('div', { class: 'error-box', text: err.message }));
    }
  };

  panel.appendChild(el('div', { class: 'slider-row' }, [
    el('label', { text: 'Home attack' }), attack, attackOut,
  ]));
  panel.appendChild(el('div', { class: 'slider-row' }, [
    el('label', { text: 'Home defence' }), defence, defenceOut,
  ]));
  panel.appendChild(el('div', { class: 'row' }, [
    el('button', { class: 'icon-btn', text: 'Apply', onclick: send }),
  ]));
  panel.appendChild(result);
  return panel;
}

/* -- arena ---------------------------------------------------------------- */

async function renderArena() {
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty', text: 'Loading the arena...' }));
  try {
    const data = await api('/api/arena');
    view.innerHTML = '';
    if (!data.run_id) {
      showEmpty(data.note || 'No evaluation run yet. Run: python -m prorrhesis backtest --kind tuning');
      return;
    }
    if (data.holdout_broken) {
      view.appendChild(el('div', { class: 'notice bad' }, [
        el('strong', { text: 'holdout_broken: ' }),
        el('span', { text: 'the configuration changed after the headline run was frozen. ' +
          'Re-declare the holdout before trusting the headline numbers.' }),
      ]));
    }

    const head = el('div', { class: 'panel' });
    head.appendChild(el('h2', { text: 'Arena' }));
    head.appendChild(el('div', { class: 'small muted', text:
      `run ${data.run_id} (${data.run_kind})` + (data.window && data.window.start
        ? ` - ${data.window.start} .. ${data.window.end}` : '') }));
    view.appendChild(head);

    const table = el('div', { class: 'panel' });
    table.appendChild(el('h2', { text: 'Models by log loss' }));
    const body = el('tbody');
    (data.models || []).forEach((model) => {
      const hasCi = model.ci_low !== null && model.ci_low !== undefined &&
        !Number.isNaN(model.ci_low);
      body.appendChild(el('tr', {}, [
        el('td', { text: model.model }),
        el('td', { text: model.n || 0 }),
        el('td', { text: model.log_loss === null || model.log_loss === undefined
          ? '-' : num(model.log_loss) }),
        el('td', { text: hasCi ? `[${num(model.ci_low)}, ${num(model.ci_high)}]` : '-' }),
        el('td', { text: model.rps === undefined ? '-' : num(model.rps) }),
        el('td', { text: model.brier === undefined ? '-' : num(model.brier) }),
        el('td', { text: model.accuracy === undefined ? '-' : pct(model.accuracy) }),
        el('td', { text: model.ece === undefined ? '-' : num(model.ece) }),
        el('td', { text: model.skill_vs_freq === undefined || model.skill_vs_freq === null
          ? '-' : pct(model.skill_vs_freq) }),
      ]));
    });
    table.appendChild(el('table', {}, [
      el('thead', {}, [el('tr', {}, [
        el('th', { text: 'model' }), el('th', { text: 'n' }), el('th', { text: 'log loss' }),
        el('th', { text: '95% CI' }), el('th', { text: 'RPS' }), el('th', { text: 'Brier' }),
        el('th', { text: 'accuracy' }), el('th', { text: 'ECE' }), el('th', { text: 'skill vs freq' }),
      ])]),
      body,
    ]));
    view.appendChild(table);

    const chartRows = (data.models || [])
      .filter((m) => m.log_loss !== null && m.log_loss !== undefined && !Number.isNaN(m.log_loss))
      .map((m) => ({
        label: m.model,
        value: m.log_loss,
        note: num(m.log_loss),
        color: m.model === 'ensemble' ? 'var(--accent-2)'
          : (m.model === data.benchmark ? 'var(--away)' : 'var(--accent)'),
      }));
    if (chartRows.length) {
      const chartPanel = el('div', { class: 'panel' });
      chartPanel.appendChild(el('h2', { text: 'Log loss (lower is better)' }));
      chartPanel.appendChild(barChart(chartRows));
      view.appendChild(chartPanel);
    }

    if (data.paired && data.paired.n) {
      const paired = el('div', { class: 'panel' });
      paired.appendChild(el('h2', { text: `Paired comparison: ${data.paired.a} v ${data.paired.b}` }));
      paired.appendChild(el('dl', { class: 'kv' }, [
        el('dt', { text: 'mean loss difference' }), el('dd', { text: num(data.paired.mean_diff) }),
        el('dt', { text: '95% CI' }),
        el('dd', { text: `[${num(data.paired.ci_low)}, ${num(data.paired.ci_high)}]` }),
        el('dt', { text: 'blocks' }), el('dd', { text: String(data.paired.blocks) }),
        el('dt', { text: 'verdict' }),
        el('dd', {}, [el('span', {
          class: 'pill ' + (data.paired.significant ? 'good' : ''),
          text: data.paired.significant
            ? (data.paired.a_better ? `${data.paired.a} is better` : `${data.paired.b} is better`)
            : 'not significant',
        })]),
      ]));
      view.appendChild(paired);
    }

    if (data.per_league && data.per_league.length) {
      const perLeague = el('div', { class: 'panel' });
      perLeague.appendChild(el('h2', { text: 'By league' }));
      const byLeague = {};
      data.per_league.forEach((row) => {
        (byLeague[row.league] = byLeague[row.league] || []).push(row);
      });
      const select = el('select', {
        onchange: (event) => drawLeague(byLeague[event.target.value]),
      }, Object.keys(byLeague).map((code) => el('option', { value: code, text: code })));
      perLeague.appendChild(el('div', { class: 'row' }, [el('span', { class: 'small muted', text: 'league' }), select]));
      const holder = el('div');
      perLeague.appendChild(holder);
      view.appendChild(perLeague);

      function drawLeague(code) {
        holder.innerHTML = '';
        const body = el('tbody');
        (byLeague[code] || []).forEach((row) => {
          body.appendChild(el('tr', {}, [
            el('td', { text: row.model }),
            el('td', { text: row.n }),
            el('td', { text: num(row.log_loss) }),
            el('td', { text: num(row.rps) }),
            el('td', { text: pct(row.accuracy) }),
          ]));
        });
        holder.appendChild(el('table', {}, [
          el('thead', {}, [el('tr', {}, [
            el('th', { text: 'model' }), el('th', { text: 'n' }), el('th', { text: 'log loss' }),
            el('th', { text: 'RPS' }), el('th', { text: 'accuracy' }),
          ])]),
          body,
        ]));
      }
      drawLeague(Object.keys(byLeague)[0]);
    }

    view.appendChild(timeMachinePanel());
  } catch (err) {
    showError(err.message, renderArena);
  }
}

function timeMachinePanel() {
  const panel = el('div', { class: 'panel' });
  panel.appendChild(el('h2', { text: 'Time machine' }));
  panel.appendChild(el('div', { class: 'small muted', text:
    'Replay any past matchweek: the models as they were then, and what actually happened.' }));
  const date = el('input', { type: 'date', value: state.timeMachineDate || '' });
  const holder = el('div', { style: 'margin-top:10px' });
  const load = async () => {
    state.timeMachineDate = date.value;
    if (!date.value) { holder.innerHTML = ''; return; }
    holder.innerHTML = '';
    holder.appendChild(el('div', { class: 'small muted', text: 'Loading...' }));
    try {
      const data = await api('/api/time-machine?date=' + encodeURIComponent(date.value));
      holder.innerHTML = '';
      if (!data.matches || !data.matches.length) {
        holder.appendChild(el('div', { class: 'empty', text: 'No evaluated matches on that date.' }));
        return;
      }
      const body = el('tbody');
      data.matches.forEach((match) => {
        body.appendChild(el('tr', {}, [
          el('td', { text: `${match.home} v ${match.away}` }),
          el('td', { text: pct(match.pH) }),
          el('td', { text: pct(match.pD) }),
          el('td', { text: pct(match.pA) }),
          el('td', { text: match.result || '-' }),
          el('td', { text: match.score || '-' }),
          el('td', { text: match.loss === null || match.loss === undefined ? '-' : num(match.loss, 3) }),
        ]));
      });
      holder.appendChild(el('table', {}, [
        el('thead', {}, [el('tr', {}, [
          el('th', { text: 'match' }), el('th', { text: 'Home' }), el('th', { text: 'Draw' }),
          el('th', { text: 'Away' }), el('th', { text: 'result' }), el('th', { text: 'score' }),
          el('th', { text: 'log loss' }),
        ])]),
        body,
      ]));
    } catch (err) {
      holder.innerHTML = '';
      holder.appendChild(el('div', { class: 'error-box', text: err.message }));
    }
  };
  date.addEventListener('change', load);
  panel.appendChild(el('div', { class: 'row' }, [
    el('span', { class: 'small muted', text: 'date' }), date,
    el('button', { class: 'icon-btn', text: 'Replay', onclick: load }),
  ]));
  panel.appendChild(holder);
  return panel;
}

/* -- calibration ---------------------------------------------------------- */

async function renderCalibration() {
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty', text: 'Loading calibration...' }));
  try {
    const data = await api('/api/calibration?model=' + encodeURIComponent(state.calibrationModel));
    view.innerHTML = '';
    const panel = el('div', { class: 'panel' });
    panel.appendChild(el('h2', { text: 'Calibration' }));
    panel.appendChild(el('div', { class: 'row' }, [
      el('span', { class: 'small muted', text: 'model' }),
      el('select', {
        onchange: (event) => { state.calibrationModel = event.target.value; renderCalibration(); },
      }, ['ensemble', 'ensemble_cal', 'market_close', 'gbm', 'dc', 'elo'].map((name) =>
        el('option', { value: name, text: name, selected: name === state.calibrationModel ? 'selected' : null }))),
    ]));
    view.appendChild(panel);

    if (!data.outcomes || !data.outcomes.length) {
      view.appendChild(el('div', { class: 'empty', text: data.note || 'No calibrated rows in this run yet.' }));
      return;
    }

    const grid = el('div', { class: 'grid2' });
    data.outcomes.forEach((outcome) => {
      const card = el('div', { class: 'panel' });
      card.appendChild(el('h2', { text: `Outcome ${outcome.outcome}` }));
      card.appendChild(el('div', { class: 'small muted', text:
        `ECE ${num(outcome.ece)} - ${outcome.n} evaluated match(es)` }));
      card.appendChild(reliabilityChart(outcome.bins));
      grid.appendChild(card);
    });
    view.appendChild(grid);
  } catch (err) {
    showError(err.message, renderCalibration);
  }
}

/* -- simulator ------------------------------------------------------------ */

async function renderSimulator() {
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty', text: 'Running the simulation...' }));
  try {
    if (state.leagues.length === 0) {
      const data = await api('/api/leagues');
      state.leagues = data.leagues || [];
    }
    const league = state.league;
    const data = await api(`/api/simulate?league=${encodeURIComponent(league)}&n=10000`);
    view.innerHTML = '';

    const panel = el('div', { class: 'panel' });
    panel.appendChild(el('h2', { text: 'Season simulation' }));
    const tabs = el('div', { class: 'row' });
    state.leagues.forEach((item) => {
      tabs.appendChild(el('button', {
        class: 'icon-btn', type: 'button', text: item.code,
        'aria-current': item.code === league ? 'true' : null,
        onclick: () => { state.league = item.code; renderSimulator(); },
      }));
    });
    panel.appendChild(tabs);
    panel.appendChild(el('div', { class: 'small muted', style: 'margin-top:8px', text:
      `${data.n} simulations - season ${data.season || '-'} - ` +
      `${data.remaining_fixtures || 0} fixture(s) left to play` }));
    view.appendChild(panel);

    if (data.disabled) {
      view.appendChild(el('div', { class: 'empty', text: data.reason || 'The simulation is not available.' }));
      return;
    }
    if (!data.teams || !data.teams.length) {
      view.appendChild(el('div', { class: 'empty', text: 'No teams to simulate.' }));
      return;
    }

    const table = el('div', { class: 'panel' });
    const body = el('tbody');
    data.teams.forEach((team) => {
      body.appendChild(el('tr', {}, [
        el('td', { text: team.team || team.team_id }),
        el('td', { text: num(team.expected_points, 1) }),
        el('td', { text: num(team.points_now, 1) }),
        el('td', {}, [probabilityBar(team.p_title, team.p_top4, team.p_relegation)]),
        el('td', { text: pct(team.p_title) }),
        el('td', { text: pct(team.p_top4) }),
        el('td', { text: pct(team.p_relegation) }),
      ]));
    });
    table.appendChild(el('table', {}, [
      el('thead', {}, [el('tr', {}, [
        el('th', { text: 'club' }), el('th', { text: 'expected pts' }),
        el('th', { text: 'pts so far' }), el('th', { text: 'title / top4 / relegation' }),
        el('th', { text: 'title' }), el('th', { text: 'top 4' }), el('th', { text: 'relegation' }),
      ])]),
      body,
    ]));
    view.appendChild(table);
  } catch (err) {
    showError(err.message, renderSimulator);
  }
}

/* -- ledger --------------------------------------------------------------- */

async function renderLedger() {
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty', text: 'Loading the ledger...' }));
  try {
    const data = await api('/api/ledger?limit=200');
    view.innerHTML = '';

    const panel = el('div', { class: 'panel' });
    panel.appendChild(el('div', { class: 'row', style: 'justify-content:space-between' }, [
      el('h2', { text: 'Ledger' }),
      el('span', {
        class: 'pill ' + (data.chain_ok ? 'good' : 'bad'),
        text: data.chain_ok ? 'chain intact' : 'chain broken',
      }),
    ]));
    panel.appendChild(el('div', { class: 'small muted', text:
      `${data.count} row(s) - ${data.settled} settled - ${data.open} still open` }));
    view.appendChild(panel);

    const rolling = data.running_log_loss || {};
    const summary = el('div', { class: 'panel' });
    summary.appendChild(el('h2', { text: 'Running log loss' }));
    const ens = rolling.ensemble || {};
    const bench = rolling.market_close || {};
    summary.appendChild(el('dl', { class: 'kv' }, [
      el('dt', { text: 'ensemble' }),
      el('dd', { text: ens.n ? `${num(ens.log_loss)} (${ens.n} settled)` : 'too few to judge' }),
      el('dt', { text: 'market_close' }),
      el('dd', { text: bench.n ? `${num(bench.log_loss)} (${bench.n} settled)` : 'too few to judge' }),
    ]));
    if (ens.n && ens.n < (rolling.threshold || 100)) {
      summary.appendChild(el('div', { class: 'notice warn', style: 'margin-top:8px', text:
        `Too few settled predictions to judge (${ens.n} of ${rolling.threshold || 100}).` }));
    }
    view.appendChild(summary);

    if (!data.rows || !data.rows.length) {
      view.appendChild(el('div', { class: 'empty' }, [
        el('div', { text: 'The ledger is empty.' }),
        el('div', { class: 'small muted', style: 'margin-top:6px',
          text: 'Run: python -m prorrhesis predict' }),
      ]));
      return;
    }

    const table = el('div', { class: 'panel' });
    const body = el('tbody');
    data.rows.forEach((row) => {
      body.appendChild(el('tr', {}, [
        el('td', { text: row.created_at }),
        el('td', { text: row.league }),
        el('td', { text: row.date }),
        el('td', { text: row.model }),
        el('td', { text: pct(row.pH) }),
        el('td', { text: pct(row.pD) }),
        el('td', { text: pct(row.pA) }),
        el('td', { text: row.result || 'open' }),
        el('td', { text: row.loss === null || row.loss === undefined ? '-' : num(row.loss, 3) }),
        el('td', { class: 'small muted', text: (row.hash || '').slice(0, 10) }),
      ]));
    });
    table.appendChild(el('table', {}, [
      el('thead', {}, [el('tr', {}, [
        el('th', { text: 'created' }), el('th', { text: 'league' }), el('th', { text: 'date' }),
        el('th', { text: 'model' }), el('th', { text: 'Home' }), el('th', { text: 'Draw' }),
        el('th', { text: 'Away' }), el('th', { text: 'result' }), el('th', { text: 'loss' }),
        el('th', { text: 'hash' }),
      ])]),
      body,
    ]));
    view.appendChild(table);
  } catch (err) {
    showError(err.message, renderLedger);
  }
}

/* -- data health ---------------------------------------------------------- */

async function renderHealth() {
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty', text: 'Loading health...' }));
  try {
    const data = await api('/api/health');
    view.innerHTML = '';

    const panel = el('div', { class: 'panel' });
    panel.appendChild(el('h2', { text: 'Data health' }));
    panel.appendChild(el('dl', { class: 'kv' }, [
      el('dt', { text: 'status' }),
      el('dd', {}, [el('span', {
        class: 'pill ' + (data.status === 'ok' ? 'good' : 'warn'), text: data.status })]),
      el('dt', { text: 'config hash' }), el('dd', { class: 'small', text: data.config_hash }),
      el('dt', { text: 'database' }),
      el('dd', { text: `${data.database.matches} matches, ${data.database.teams} clubs, ` +
        `${data.database.aliases} aliases` }),
      el('dt', { text: 'unresolved names' }),
      el('dd', { text: String(data.unresolved_names.count) }),
    ]));
    view.appendChild(panel);

    (data.warnings || []).forEach((warning) => {
      view.appendChild(el('div', { class: 'notice warn', text: warning }));
    });

    const sources = el('div', { class: 'panel' });
    sources.appendChild(el('h2', { text: 'Sources' }));
    const body = el('tbody');
    Object.keys(data.sources || {}).forEach((name) => {
      const source = data.sources[name];
      body.appendChild(el('tr', {}, [
        el('td', { text: name }),
        el('td', { text: source.status || '-' }),
        el('td', { text: source.last_ingest || source.latest_snapshot || '-' }),
        el('td', { class: 'small muted', text: source.last_error || '' }),
      ]));
    });
    sources.appendChild(el('table', {}, [
      el('thead', {}, [el('tr', {}, [
        el('th', { text: 'source' }), el('th', { text: 'status' }),
        el('th', { text: 'last update' }), el('th', { text: 'note' }),
      ])]),
      body,
    ]));
    view.appendChild(sources);

    const coverage = el('div', { class: 'panel' });
    coverage.appendChild(el('h2', { text: 'Column coverage per season' }));
    if (!data.column_coverage || !data.column_coverage.length) {
      coverage.appendChild(el('div', { class: 'empty', text: 'No coverage data yet.' }));
    } else {
      const cbody = el('tbody');
      data.column_coverage.forEach((row) => {
        cbody.appendChild(el('tr', {}, [
          el('td', { text: row.league }),
          el('td', { text: row.season }),
          el('td', { text: row.matches }),
          el('td', { text: pct(row.close_odds_pct, 0) }),
          el('td', { text: pct(row.open_odds_pct, 0) }),
          el('td', { text: pct(row.shots_pct, 0) }),
          el('td', { text: pct(row.cards_pct, 0) }),
        ]));
      });
      coverage.appendChild(el('table', {}, [
        el('thead', {}, [el('tr', {}, [
          el('th', { text: 'league' }), el('th', { text: 'season' }), el('th', { text: 'matches' }),
          el('th', { text: 'closing odds' }), el('th', { text: 'open odds' }),
          el('th', { text: 'shots' }), el('th', { text: 'cards' }),
        ])]),
        cbody,
      ]));
    }
    view.appendChild(coverage);
  } catch (err) {
    showError(err.message, renderHealth);
  }
}

/* -- theme and refresh ---------------------------------------------------- */

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('prorrhesis-theme', theme); } catch (err) { /* private mode */ }
}

function startRefresh() {
  setStatus('Refreshing...');
  api('/api/refresh', { method: 'POST' })
    .then((job) => {
      setStatus('Refresh job ' + job.id);
      const timer = setInterval(() => {
        api('/api/jobs/' + job.id)
          .then((current) => {
            if (current.status === 'done' || current.status === 'failed') {
              clearInterval(timer);
              setStatus(current.status === 'done' ? 'Refresh finished' : 'Refresh failed');
              if (current.error) setStatus('Refresh failed: ' + current.error);
              route();
            }
          })
          .catch(() => { clearInterval(timer); setStatus('Refresh status unavailable'); });
      }, 2000);
    })
    .catch((err) => setStatus('Refresh failed: ' + err.message));
}

document.getElementById('theme').addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

document.getElementById('refresh').addEventListener('click', startRefresh);

let stored = 'light';
try { stored = localStorage.getItem('prorrhesis-theme') || 'light'; } catch (err) { /* ignore */ }
applyTheme(stored);

renderTabs();
route();
