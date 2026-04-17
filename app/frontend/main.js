const API = '/api';
let agents = {};
let currentMode = 'hvh';
let gameState = null;
let gameId = null;
let autoPlayTimer = null;
let isAutoPlaying = false;
let hoveredCol = null;

// Tournament state variables
let cancelTourney = false;

// ── Boot ───────────────────────────────────────────────────────────── //
async function boot() {
  try {
    const res = await fetch(`${API}/agents`);
    agents = await res.json();
    populateAgentDropdowns();
  } catch(e) {
    console.error('Failed to load agents', e);
  }
}

function populateAgentDropdowns() {
  const humanOpt = '<option value="human">Human</option>';
  const agentOpts = Object.entries(agents).map(([k,v]) =>
    `<option value="${k}">${v.label}</option>`
  ).join('');

  // Setup view: start with all options, mode selection will filter
  document.getElementById('p1-type').innerHTML = humanOpt + agentOpts;
  document.getElementById('p2-type').innerHTML = humanOpt + agentOpts;

  // Tournament view dropdowns (Agents only — no humans in tournament)
  document.getElementById('tourney-p1-type').innerHTML = agentOpts;
  document.getElementById('tourney-p2-type').innerHTML = agentOpts;

  // Set default unique agents for tournament if possible
  const keys = Object.keys(agents);
  if (keys.length > 0) {
    document.getElementById('tourney-p1-type').value = keys[0];
    document.getElementById('tourney-p2-type').value = keys[Math.min(1, keys.length - 1)];
  }

  updatePlayerUI('p1');
  updatePlayerUI('p2');
  updatePlayerUI('tourney-p1');
  updatePlayerUI('tourney-p2');
}

// ── Mode selection ──────────────────────────────────────────────────── //
function selectMode(mode) {
  currentMode = mode;
  ['hvh','hva','ava'].forEach(m =>
    document.getElementById(`mode-${m}`).classList.toggle('selected', m === mode)
  );

  const p1sel = document.getElementById('p1-type');
  const p2sel = document.getElementById('p2-type');
  const humanOpt = '<option value="human">Human</option>';
  const agentOpts = Object.entries(agents).map(([k,v]) =>
    `<option value="${k}">${v.label}</option>`
  ).join('');

  if (mode === 'hvh') {
    // Both must be human
    p1sel.innerHTML = humanOpt;
    p2sel.innerHTML = humanOpt;
    p1sel.value = 'human';
    p2sel.value = 'human';
    p1sel.disabled = true;
    p2sel.disabled = true;

  } else if (mode === 'hva') {
    // P1 can be human or agent; P2 can be human or agent.
    // But if P1 is human, P2 can't also be human (and vice versa).
    // We start P1 = human, P2 = first agent.
    p1sel.innerHTML = humanOpt + agentOpts;
    p2sel.innerHTML = agentOpts; // P2 starts without human since P1 is human
    p1sel.disabled = false;
    p2sel.disabled = false;
    p1sel.value = 'human';
    const firstAgent = Object.keys(agents)[0];
    if (firstAgent) p2sel.value = firstAgent;

    // Attach mutual-exclusion listener for HvA mode
    p1sel.onchange = function() { onHvaPlayerChange('p1'); updatePlayerUI('p1'); };
    p2sel.onchange = function() { onHvaPlayerChange('p2'); updatePlayerUI('p2'); };

  } else { // ava — no humans allowed
    p1sel.innerHTML = agentOpts;
    p2sel.innerHTML = agentOpts;
    p1sel.disabled = false;
    p2sel.disabled = false;
    // Remove any HvA listeners
    p1sel.onchange = function() { updatePlayerUI('p1'); };
    p2sel.onchange = function() { updatePlayerUI('p2'); };

    const keys = Object.keys(agents);
    if (keys.length > 0) {
      p1sel.value = keys[0];
      p2sel.value = keys[Math.min(1, keys.length - 1)];
    }
  }

  updatePlayerUI('p1');
  updatePlayerUI('p2');
}

/**
 * Called when a player dropdown changes in HvA mode.
 * Ensures the opposite player cannot also be human.
 */
function onHvaPlayerChange(changedPlayer) {
  const p1sel = document.getElementById('p1-type');
  const p2sel = document.getElementById('p2-type');
  const humanOpt = '<option value="human">Human</option>';
  const agentOpts = Object.entries(agents).map(([k,v]) =>
    `<option value="${k}">${v.label}</option>`
  ).join('');

  if (changedPlayer === 'p1') {
    if (p1sel.value === 'human') {
      // P2 must be an agent
      const prev = p2sel.value;
      p2sel.innerHTML = agentOpts;
      if (agents[prev]) p2sel.value = prev;
      else {
        const firstAgent = Object.keys(agents)[0];
        if (firstAgent) p2sel.value = firstAgent;
      }
    } else {
      // P1 is agent — P2 can be human or agent
      const prev = p2sel.value;
      p2sel.innerHTML = humanOpt + agentOpts;
      p2sel.value = prev || 'human';
    }
  } else {
    if (p2sel.value === 'human') {
      // P1 must be an agent
      const prev = p1sel.value;
      p1sel.innerHTML = agentOpts;
      if (agents[prev]) p1sel.value = prev;
      else {
        const firstAgent = Object.keys(agents)[0];
        if (firstAgent) p1sel.value = firstAgent;
      }
    } else {
      // P2 is agent — P1 can be human or agent
      const prev = p1sel.value;
      p1sel.innerHTML = humanOpt + agentOpts;
      p1sel.value = prev || 'human';
    }
  }
}

function updatePlayerUI(prefix) {
  const el = document.getElementById(`${prefix}-type`);
  if (!el) return;
  const sel = el.value;
  const container = document.getElementById(`${prefix}-params`);

  if (sel === 'human' || !agents[sel]) {
    container.innerHTML = '';
    return;
  }

  const params = agents[sel].params || {};
  if (Object.keys(params).length === 0) {
    container.innerHTML = '<div style="font-size:0.6rem;color:var(--muted)">No parameters</div>';
    return;
  }

  let html = '';
  for (const [key, spec] of Object.entries(params)) {
    html += `<div class="param-row">`;
    html += `<label>${spec.label}</label>`;
    if (spec.type === 'int' || spec.type === 'float') {
      const step = spec.type === 'float' ? '0.01' : '1';
      html += `<div class="range-row">
        <input type="range" id="${prefix}-${key}"
          min="${spec.min}" max="${spec.max}" step="${step}" value="${spec.default}"
          oninput="document.getElementById('${prefix}-${key}-val').textContent=this.value">
        <span class="range-val" id="${prefix}-${key}-val">${spec.default}</span>
      </div>`;
    } else {
      html += `<input type="text" id="${prefix}-${key}" placeholder="${spec.default}" value="">`;
    }
    html += `</div>`;
  }
  container.innerHTML = html;
}

function getPlayerConfig(prefix) {
  const el = document.getElementById(`${prefix}-type`);
  if (!el) return {};
  const sel = el.value;
  if (sel === 'human' || !agents[sel]) return {};
  const params = agents[sel].params || {};
  const config = {};
  for (const [key, spec] of Object.entries(params)) {
    const inputEl = document.getElementById(`${prefix}-${key}`);
    if (!inputEl) continue;
    const val = inputEl.value || spec.default;
    config[key] = (spec.type === 'int') ? parseInt(val) :
                  (spec.type === 'float') ? parseFloat(val) : val;
  }
  return config;
}

// ── Tournament Logic ────────────────────────────────────────────────── //
async function runTournament() {
  const numGames = parseInt(document.getElementById('tourney-games').value);
  if (!numGames || numGames < 1) {
    alert('Please enter a valid number of games.');
    return;
  }

  const p1type = document.getElementById('tourney-p1-type').value;
  const p2type = document.getElementById('tourney-p2-type').value;
  const p1cfg  = getPlayerConfig('tourney-p1');
  const p2cfg  = getPlayerConfig('tourney-p2');

  cancelTourney = false;

  // Setup UI for running
  document.getElementById('btn-run-tourney').style.display = 'none';
  document.getElementById('btn-stop-tourney').style.display = 'block';
  document.getElementById('tourney-results-card').style.display = 'block';

  document.getElementById('tourney-label-p1').textContent = `${agentLabel(p1type)} WINS`;
  document.getElementById('tourney-label-p2').textContent = `${agentLabel(p2type)} WINS`;

  let results = { p1: 0, p2: 0, draw: 0 };
  document.getElementById('tourney-res-p1').textContent = '0';
  document.getElementById('tourney-res-p2').textContent = '0';
  document.getElementById('tourney-res-draw').textContent = '0';

  // Track tournament start time for grouping in history
  const tournamentId = 'tourney_' + Date.now();

  for (let i = 1; i <= numGames; i++) {
    if (cancelTourney) break;
    document.getElementById('tourney-status').innerHTML = `Running game <b><span style="color:var(--accent)">${i}</span></b> of ${numGames}...`;

    try {
      const res = await fetch(`${API}/games`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          mode: 'ava',
          player1_type: p1type,
          player2_type: p2type,
          player1_config: p1cfg,
          player2_config: p2cfg,
          tournament_id: tournamentId,
        })
      });

      if (!res.ok) throw new Error('Failed to start tournament game');
      let state = await res.json();
      let tGameId = state.game_id;

      // Play out the game loop
      while (state.status === 'active') {
        if (cancelTourney) {
          await fetch(`${API}/games/${tGameId}`, {method:'DELETE'}).catch(()=>{});
          break;
        }
        const stepRes = await fetch(`${API}/games/${tGameId}/agent-move`, {method:'POST'});
        if (!stepRes.ok) break;
        state = await stepRes.json();
      }

      if (state.status === 'finished') {
        if (state.winner === 'player1') results.p1++;
        else if (state.winner === 'player2') results.p2++;
        else results.draw++;

        document.getElementById('tourney-res-p1').textContent = results.p1;
        document.getElementById('tourney-res-p2').textContent = results.p2;
        document.getElementById('tourney-res-draw').textContent = results.draw;
      }
    } catch(e) {
      console.error(e);
      document.getElementById('tourney-status').textContent = `Error during tournament: ${e.message}`;
      break;
    }
  }

  // Teardown UI
  document.getElementById('btn-run-tourney').style.display = 'block';
  document.getElementById('btn-stop-tourney').style.display = 'none';
  if (cancelTourney) {
    document.getElementById('tourney-status').textContent = 'Tournament stopped early.';
  } else {
    document.getElementById('tourney-status').innerHTML = `<span style="color:var(--accent)">Tournament Complete!</span>`;
  }

  // Store tournament metadata in sessionStorage for history grouping
  try {
    const existing = JSON.parse(sessionStorage.getItem('tournaments') || '[]');
    existing.push({
      id: tournamentId,
      p1type, p2type,
      games: numGames,
      results,
      timestamp: new Date().toISOString(),
    });
    sessionStorage.setItem('tournaments', JSON.stringify(existing));
  } catch(e) {}
}

function stopTourney() {
  cancelTourney = true;
  document.getElementById('tourney-status').textContent = 'Stopping tournament (finishing current move)...';
}

// ── Start standard game ─────────────────────────────────────────────── //
async function startGame() {
  const p1type = document.getElementById('p1-type').value;
  const p2type = document.getElementById('p2-type').value;
  const p1cfg  = getPlayerConfig('p1');
  const p2cfg  = getPlayerConfig('p2');

  try {
    const res = await fetch(`${API}/games`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        mode: currentMode,
        player1_type: p1type,
        player2_type: p2type,
        player1_config: p1cfg,
        player2_config: p2cfg,
      })
    });

    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || 'Failed to start game');
      return;
    }

    gameState = await res.json();
    gameId = gameState.game_id;
    stopAutoPlay();
    _lastRenderedBoard = null;  // force full board rebuild for new game

    showView('game');
    document.getElementById('nav-game').style.display = '';
    renderGame(gameState);

    // If AvA, set up controls
    if (currentMode === 'ava') {
      document.getElementById('ava-controls').style.display = 'flex';
    } else {
      document.getElementById('ava-controls').style.display = 'none';
    }

    if (gameState.status === 'finished') {
      handleGameEnd(gameState);
    }
  } catch(e) {
    alert('Error starting game: ' + e.message);
  }
}

// ── Board rendering ─────────────────────────────────────────────────── //
function renderGame(state) {
  gameState = state;
  renderBoard(state);
  renderSidebar(state);
  updateStatusBar(state);
}

// Cache the last rendered board for incremental updates
let _lastRenderedBoard = null;

function renderBoard(state) {
  const board = state.board;
  const ROWS = 6, COLS = 7;
  const boardEl = document.getElementById('board');
  const arrowsEl = document.getElementById('col-arrows');

  // Build arrows only once per game (not every move)
  if (arrowsEl.children.length !== COLS) {
    arrowsEl.innerHTML = '';
    for (let c = 0; c < COLS; c++) {
      const arr = document.createElement('div');
      arr.className = 'col-arrow';
      arr.textContent = '▼';
      arr.dataset.col = c;
      arr.addEventListener('mouseenter', () => setHoverCol(c));
      arr.addEventListener('mouseleave', () => setHoverCol(null));
      arr.addEventListener('click', () => humanMove(c));
      arrowsEl.appendChild(arr);
    }
  }

  // Full rebuild only on first render of a new game
  if (boardEl.children.length !== ROWS * COLS) {
    boardEl.innerHTML = '';
    _lastRenderedBoard = null;
    for (let r = ROWS - 1; r >= 0; r--) {
      for (let c = 0; c < COLS; c++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.row = r;
        cell.dataset.col = c;
        cell.addEventListener('mouseenter', () => setHoverCol(c));
        cell.addEventListener('mouseleave', () => setHoverCol(null));
        cell.addEventListener('click', () => humanMove(c));
        boardEl.appendChild(cell);
      }
    }
  }

  // Incremental update: only touch cells that actually changed
  const cells = boardEl.querySelectorAll('.cell');
  for (let r = ROWS - 1; r >= 0; r--) {
    for (let c = 0; c < COLS; c++) {
      const val = board[r][c];
      if (_lastRenderedBoard && _lastRenderedBoard[r][c] === val) continue;
      const visualRow = (ROWS - 1 - r);
      const cellIdx = visualRow * COLS + c;
      const cell = cells[cellIdx];
      if (!cell) continue;
      cell.classList.remove('p1', 'p2', 'winning', 'ghost-p1', 'ghost-p2');
      if (val === 1) cell.classList.add('p1');
      else if (val === 2) cell.classList.add('p2');
    }
  }
  _lastRenderedBoard = board.map(row => row.slice());

  // Highlight winning cells if finished
  if (state.status === 'finished' && state.winner !== 'draw') {
    highlightWinners(state);
  }
}

function setHoverCol(col) {
  hoveredCol = col;
  if (!gameState || gameState.status !== 'active') return;
  const current = gameState.current_player;
  const isHumanTurn = (
    (current === 1 && gameState.player1_type === 'human') ||
    (current === 2 && gameState.player2_type === 'human')
  );
  if (!isHumanTurn) return;

  // Clear ghosts
  document.querySelectorAll('.cell.ghost-p1, .cell.ghost-p2').forEach(c => {
    c.classList.remove('ghost-p1', 'ghost-p2');
  });
  document.querySelectorAll('.col-arrow').forEach(a => a.classList.remove('hovered'));

  if (col === null) return;

  document.querySelectorAll(`.col-arrow[data-col="${col}"]`).forEach(a =>
    a.classList.add('hovered'));

  const board = gameState.board;
  let dropRow = -1;
  for (let r = 0; r < 6; r++) {
    if (board[r][col] === 0) { dropRow = r; break; }
  }
  if (dropRow === -1) return;

  const cells = document.querySelectorAll('.cell');
  cells.forEach(cell => {
    if (parseInt(cell.dataset.col) === col && parseInt(cell.dataset.row) === dropRow) {
      cell.classList.add(current === 1 ? 'ghost-p1' : 'ghost-p2');
    }
  });
}

function highlightWinners(state) {
  const board = state.board;
  const winner = state.winner;
  const piece = winner === 'player1' ? 1 : 2;

  const cells = document.querySelectorAll('.cell');
  const ROWS = 6, COLS = 7;
  const winning = new Set();

  const check = (r, c, dr, dc) => {
    const line = [];
    for (let i = 0; i < 4; i++) {
      const nr = r + i*dr, nc = c + i*dc;
      if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) return;
      if (board[nr][nc] !== piece) return;
      line.push(`${nr},${nc}`);
    }
    if (line.length === 4) line.forEach(k => winning.add(k));
  };

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c] === piece) {
        check(r,c,0,1); check(r,c,1,0);
        check(r,c,1,1); check(r,c,1,-1);
      }
    }
  }

  cells.forEach(cell => {
    const key = `${cell.dataset.row},${cell.dataset.col}`;
    if (winning.has(key)) cell.classList.add('winning');
  });
}

function renderSidebar(state) {
  document.getElementById('label-p1').textContent = 'PLAYER 1';
  document.getElementById('label-p2').textContent = 'PLAYER 2';
  document.getElementById('type-p1').textContent = agentLabel(state.player1_type);
  document.getElementById('type-p2').textContent = agentLabel(state.player2_type);
  document.getElementById('info-mode').textContent = state.mode.toUpperCase();

  const moves = state.move_history || [];
  const m1 = moves.filter(m => m.player === 1).length;
  const m2 = moves.filter(m => m.player === 2).length;
  document.getElementById('moves-p1').textContent = m1;
  document.getElementById('moves-p2').textContent = m2;
  document.getElementById('info-moves-p1').textContent = m1;
  document.getElementById('info-moves-p2').textContent = m2;

  const isActive = state.status === 'active';
  const cp = state.current_player;
  const isHvH = state.mode === 'hvh';
  document.getElementById('turn-p1').style.display = (isActive && cp === 1 && isHvH) ? 'block' : 'none';
  document.getElementById('turn-p2').style.display = (isActive && cp === 2 && isHvH) ? 'block' : 'none';

  document.getElementById('card-p1').classList.toggle('active-turn', isActive && cp === 1 && isHvH);
  document.getElementById('card-p1').classList.remove('p2-active');
  document.getElementById('card-p2').classList.toggle('active-turn', isActive && cp === 2 && isHvH);
  document.getElementById('card-p2').classList.toggle('p2-active', isActive && cp === 2 && isHvH);
}

function agentLabel(type) {
  if (type === 'human') return 'Human';
  return agents[type]?.label || type;
}

function updateStatusBar(state) {
  const bar = document.getElementById('status-bar');
  bar.className = 'status-bar';

  if (state.status === 'active') {
    const cp = state.current_player;
    const type = cp === 1 ? state.player1_type : state.player2_type;
    const isHuman = type === 'human';
    bar.textContent = isHuman
      ? `▼ PLAYER ${cp}'S TURN — CLICK A COLUMN`
      : `◈ PLAYER ${cp} (${agentLabel(type)}) IS THINKING...`;
  } else if (state.winner === 'draw') {
    bar.textContent = '◈ DRAW! — BOARD IS FULL';
    bar.classList.add('win');
  } else if (state.winner === 'player1') {
    bar.textContent = '★ PLAYER 1 WINS!';
    bar.classList.add('win');
  } else if (state.winner === 'player2') {
    bar.textContent = '★ PLAYER 2 WINS!';
    bar.classList.add('win', 'p2-win');
  }
}

// ── Human move ────────────────────────────────────────────────────── //
async function humanMove(col) {
  if (!gameState || gameState.status !== 'active') return;
  const cp = gameState.current_player;
  const isHuman = (
    (cp === 1 && gameState.player1_type === 'human') ||
    (cp === 2 && gameState.player2_type === 'human')
  );
  if (!isHuman) return;

  const agentWillRespond = (
    (cp === 1 && gameState.player2_type !== 'human') ||
    (cp === 2 && gameState.player1_type !== 'human')
  );
  if (agentWillRespond) setThinking(true, cp === 1 ? 2 : 1);
  try {
    const res = await fetch(`${API}/games/${gameId}/move`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({column: col})
    });
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail);
      return;
    }
    const state = await res.json();
    renderGame(state);
    if (state.status === 'finished') handleGameEnd(state);
  } catch(e) {
    console.error(e);
  } finally {
    if (agentWillRespond) setThinking(false, cp === 1 ? 2 : 1);
  }
}

// ── AvA controls ──────────────────────────────────────────────────── //
async function stepAva() {
  if (!gameState || gameState.status !== 'active') return;
  setThinking(true, gameState.current_player);
  try {
    const res = await fetch(`${API}/games/${gameId}/agent-move`, {method:'POST'});
    if (!res.ok) return;
    const state = await res.json();
    renderGame(state);
    if (state.status === 'finished') {
      stopAutoPlay();
      handleGameEnd(state);
    }
  } catch(e) {
    console.error(e);
  } finally {
    setThinking(false, gameState.current_player);
  }
}

function toggleAutoPlay() {
  if (isAutoPlaying) stopAutoPlay();
  else startAutoPlay();
}

function startAutoPlay() {
  if (!gameState || gameState.status !== 'active') return;
  isAutoPlaying = true;
  document.getElementById('btn-auto').textContent = '⏸ PAUSE';
  scheduleNextMove();
}

function stopAutoPlay() {
  isAutoPlaying = false;
  if (autoPlayTimer) { clearTimeout(autoPlayTimer); autoPlayTimer = null; }
  document.getElementById('btn-auto').textContent = '⏵ AUTO';
}

function scheduleNextMove() {
  if (!isAutoPlaying) return;
  const speed = parseInt(document.getElementById('ava-speed').value);
  autoPlayTimer = setTimeout(async () => {
    if (!isAutoPlaying || !gameState || gameState.status !== 'active') {
      stopAutoPlay(); return;
    }
    await stepAva();
    if (gameState.status === 'active') scheduleNextMove();
    else stopAutoPlay();
  }, speed);
}

// ── Game end ──────────────────────────────────────────────────────── //
function handleGameEnd(state) {
  stopAutoPlay();
  document.getElementById('btn-abandon').style.display = 'none';
  setTimeout(() => showEndModal(state), 600);
}

function showEndModal(state) {
  const title = document.getElementById('modal-title');
  const sub   = document.getElementById('modal-subtitle');

  if (state.winner === 'draw') {
    title.textContent = "IT'S A DRAW!";
    sub.textContent   = `${state.move_history.length} moves — board is full`;
  } else {
    const p = state.winner === 'player1' ? 1 : 2;
    title.textContent = `PLAYER ${p} WINS!`;
    const type = p === 1 ? state.player1_type : state.player2_type;
    sub.textContent = `${agentLabel(type)} wins in ${state.move_history.length} moves`;
  }

  const mini = document.getElementById('modal-board');
  mini.innerHTML = '';
  const board = state.board;
  for (let r = 5; r >= 0; r--) {
    for (let c = 0; c < 7; c++) {
      const cell = document.createElement('div');
      cell.className = 'mini-cell';
      if (board[r][c] === 1) cell.classList.add('p1');
      else if (board[r][c] === 2) cell.classList.add('p2');
      mini.appendChild(cell);
    }
  }

  document.getElementById('modal-move-log').style.display = 'none';
  document.getElementById('end-modal').classList.add('open');
}

function closeModal() {
  document.getElementById('end-modal').classList.remove('open');
}

// ── Misc actions ──────────────────────────────────────────────────── //
function setupNewGame() {
  closeModal();
  stopAutoPlay();
  if (gameId) {
    fetch(`${API}/games/${gameId}`, {method:'DELETE'}).catch(() => {});
    gameId = null;
  }
  document.getElementById('btn-abandon').style.display = '';
  showView('setup');
  document.getElementById('nav-game').style.display = 'none';
}

async function abandonGame() {
  if (!gameId) return;
  if (!confirm('Abandon current game?')) return;
  await fetch(`${API}/games/${gameId}`, {method:'DELETE'});
  setupNewGame();
}

function setThinking(on, playerNum) {
  if (playerNum === 1) {
    document.getElementById('thinking-p1').classList.toggle('visible', on);
    document.getElementById('thinking-p2').classList.remove('visible');
  } else if (playerNum === 2) {
    document.getElementById('thinking-p2').classList.toggle('visible', on);
    document.getElementById('thinking-p1').classList.remove('visible');
  } else {
    document.getElementById('thinking-p1').classList.remove('visible');
    document.getElementById('thinking-p2').classList.remove('visible');
  }
}

// ── Navigation ────────────────────────────────────────────────────── //
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(`view-${name}`).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
}

// ── History ──────────────────────────────────────────────────────── //
async function loadHistory() {
  // Load stats
  try {
    const res = await fetch(`${API}/stats`);
    const stats = await res.json();
    renderStats(stats);
  } catch(e) {}

  const histEl = document.getElementById('history-content');
  histEl.innerHTML = '<div class="empty-state"><div class="icon">◈</div>Loading...</div>';

  try {
    const res = await fetch(`${API}/history?limit=200`);
    const records = await res.json();
    if (!records.length) {
      histEl.innerHTML = '<div class="empty-state"><div class="icon">◈</div>No games played yet</div>';
      return;
    }

    // Load tournament metadata from sessionStorage
    let tournamentMeta = [];
    try {
      tournamentMeta = JSON.parse(sessionStorage.getItem('tournaments') || '[]');
    } catch(e) {}

    // Group records: detect tournament sequences by matching tournament_id or
    // heuristic (consecutive AVA games with same matchup in rapid succession).
    // We use tournament_id from the record if available, otherwise fall back to
    // grouping consecutive same-matchup AvA games played within a 5-minute window.
    const groups = groupRecords(records);

    let html = '';
    let globalIdx = records.length;

    for (const group of groups) {
      if (group.type === 'tournament') {
        html += renderTournamentFolder(group, globalIdx);
        globalIdx -= group.games.length;
      } else {
        const r = group.game;
        html += renderHistoryRow(r);
        globalIdx--;
      }
    }

    histEl.innerHTML = `
      <table class="history-table" id="history-outer-table">
        <thead>
          <tr>
            <th>#</th><th>MODE</th><th>PLAYER 1</th><th>PLAYER 2</th>
            <th>WINNER</th><th>MOVES</th><th>DATE</th>
          </tr>
        </thead>
        <tbody>${html}</tbody>
      </table>
    `;
  } catch(e) {
    histEl.innerHTML = '<div class="empty-state"><div class="icon">⚠</div>Error loading history</div>';
  }
}

/**
 * Group history records. Consecutive AvA games with the same matchup
 * played within 5 minutes of each other are bundled as a tournament folder.
 * Minimum 3 games to form a folder.
 */
function groupRecords(records) {
  const groups = [];
  let i = 0;

  while (i < records.length) {
    const r = records[i];

    // Try to start a tournament group
    if (r.mode === 'ava') {
      let j = i + 1;
      const p1 = r.player1_type, p2 = r.player2_type;
      let prevTime = new Date(r.created_at).getTime();

      while (j < records.length) {
        const nr = records[j];
        const nTime = new Date(nr.created_at).getTime();
        const timeDiff = Math.abs(nTime - prevTime);
        if (nr.mode === 'ava' && nr.player1_type === p1 && nr.player2_type === p2 && timeDiff < 5 * 60 * 1000) {
          prevTime = nTime;
          j++;
        } else {
          break;
        }
      }

      const count = j - i;
      if (count >= 3) {
        groups.push({ type: 'tournament', games: records.slice(i, j) });
        i = j;
        continue;
      }
    }

    groups.push({ type: 'single', game: r });
    i++;
  }

  return groups;
}

function renderTournamentFolder(group, startIdx) {
  const games = group.games;
  const first = games[0];
  const p1wins = games.filter(g => g.winner === 'player1').length;
  const p2wins = games.filter(g => g.winner === 'player2').length;
  const draws  = games.filter(g => g.winner === 'draw').length;
  const folderId = 'folder_' + first.id;
  const date = new Date(first.created_at).toLocaleDateString();

  const rowsHtml = games.map((r, idx) => `
    <tr onclick="showHistoryDetail(${r.id})" style="cursor:pointer">
      <td style="color:var(--muted)">${r.id}</td>
      <td><span class="badge ava">AVA</span></td>
      <td>${agentLabel(r.player1_type)}</td>
      <td>${agentLabel(r.player2_type)}</td>
      <td>${winnerBadge(r.winner)}</td>
      <td>${r.total_moves}</td>
      <td style="color:var(--muted)">${new Date(r.created_at).toLocaleDateString()}</td>
    </tr>
  `).join('');

  return `
    <tr>
      <td colspan="7" style="padding: 0.4rem 0;">
        <div class="tournament-folder" id="${folderId}">
          <div class="tournament-folder-header" onclick="toggleFolder('${folderId}')">
            <span class="folder-icon">🏆</span>
            <div class="folder-info">
              <span class="badge tournament">TOURNAMENT</span>
              <span style="color:var(--text); font-size:0.7rem;">${agentLabel(first.player1_type)} <span style="color:var(--muted)">vs</span> ${agentLabel(first.player2_type)}</span>
              <span style="color:var(--muted); font-size:0.65rem;">${games.length} games &nbsp;·&nbsp; <span style="color:var(--p1)">${p1wins}W</span> / <span style="color:var(--p2)">${p2wins}W</span> / <span style="color:var(--muted)">${draws}D</span></span>
              <span style="color:var(--muted); font-size:0.65rem;">${date}</span>
            </div>
            <span class="folder-toggle">▶</span>
          </div>
          <div class="tournament-folder-games">
            <table class="history-table" style="margin:0">
              <thead>
                <tr>
                  <th>#</th><th>MODE</th><th>PLAYER 1</th><th>PLAYER 2</th>
                  <th>WINNER</th><th>MOVES</th><th>DATE</th>
                </tr>
              </thead>
              <tbody>${rowsHtml}</tbody>
            </table>
          </div>
        </div>
      </td>
    </tr>
  `;
}

function renderHistoryRow(r) {
  return `
    <tr onclick="showHistoryDetail(${r.id})">
      <td style="color:var(--muted)">${r.id}</td>
      <td><span class="badge ${r.mode}">${r.mode.toUpperCase()}</span></td>
      <td>${agentLabel(r.player1_type)}</td>
      <td>${agentLabel(r.player2_type)}</td>
      <td>${winnerBadge(r.winner)}</td>
      <td>${r.total_moves}</td>
      <td style="color:var(--muted)">${new Date(r.created_at).toLocaleDateString()}</td>
    </tr>
  `;
}

function toggleFolder(folderId) {
  const el = document.getElementById(folderId);
  if (el) el.classList.toggle('open');
}

function renderStats(stats) {
  const sec = document.getElementById('stats-section');
  const modes = stats.by_mode || {};
  const winners = stats.by_winner || {};
  sec.innerHTML = `
    <div class="stats-grid">
      <div class="stat-card">
        <span class="stat-val">${stats.total_games || 0}</span>
        <span class="stat-label">TOTAL GAMES</span>
      </div>
      <div class="stat-card">
        <span class="stat-val">${modes.hvh || 0}</span>
        <span class="stat-label">HUMAN VS HUMAN</span>
      </div>
      <div class="stat-card">
        <span class="stat-val">${modes.hva || 0}</span>
        <span class="stat-label">HUMAN VS AGENT</span>
      </div>
      <div class="stat-card">
        <span class="stat-val">${modes.ava || 0}</span>
        <span class="stat-label">AGENT VS AGENT</span>
      </div>
      <div class="stat-card">
        <span class="stat-val" style="color:var(--p1)">${winners.player1 || 0}</span>
        <span class="stat-label">P1 WINS</span>
      </div>
      <div class="stat-card">
        <span class="stat-val" style="color:var(--p2)">${winners.player2 || 0}</span>
        <span class="stat-label">P2 WINS</span>
      </div>
      <div class="stat-card">
        <span class="stat-val" style="color:var(--muted)">${winners.draw || 0}</span>
        <span class="stat-label">DRAWS</span>
      </div>
    </div>
  `;
}

function winnerBadge(w) {
  if (w === 'player1') return `<span class="badge win1">P1 WINS</span>`;
  if (w === 'player2') return `<span class="badge win2">P2 WINS</span>`;
  if (w === 'draw') return `<span class="badge draw">DRAW</span>`;
  return `<span style="color:var(--muted)">—</span>`;
}

async function showHistoryDetail(id) {
  try {
    const res = await fetch(`${API}/history/${id}`);
    const r = await res.json();

    document.getElementById('modal-title').textContent = `GAME #${r.id}`;
    document.getElementById('modal-subtitle').textContent =
      `${r.mode.toUpperCase()} | ${agentLabel(r.player1_type)} vs ${agentLabel(r.player2_type)} | ${r.total_moves} moves`;

    const mini = document.getElementById('modal-board');
    mini.innerHTML = '';
    const board = r.final_board;
    for (let row = 5; row >= 0; row--) {
      for (let c = 0; c < 7; c++) {
        const cell = document.createElement('div');
        cell.className = 'mini-cell';
        if (board[row][c] === 1) cell.classList.add('p1');
        else if (board[row][c] === 2) cell.classList.add('p2');
        mini.appendChild(cell);
      }
    }

    const logEl = document.getElementById('modal-move-log');
    if (r.move_history && r.move_history.length > 0) {
      logEl.style.display = 'block';
      logEl.innerHTML = `<div class="move-log-title" style="border-bottom: 1px solid var(--border); margin-bottom: 5px;">MOVE LOG</div>` +
      r.move_history.map((m, i) =>
        `<div class="move-entry">
          <div class="dot p${m.player}"></div>
          <span class="move-num">#${i+1}</span>
          <span>P${m.player} → Col ${m.col + 1}</span>
        </div>`
      ).reverse().join('');
    } else {
      logEl.style.display = 'none';
    }

    document.getElementById('end-modal').classList.add('open');
  } catch(e) {}
}

boot();
