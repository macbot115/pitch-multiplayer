#!/usr/bin/env node
// ══════════════════════════════════════════════════════
// PITCH MULTIPLAYER SERVER — Node.js + ws + express
// ══════════════════════════════════════════════════════
const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');
const path = require('path');

const PORT = process.env.PORT || 3000;

// ── HTTP SERVER ──
const app = express();
app.use(express.static(__dirname));
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'pitch_multiplayer.html')));

const server = http.createServer(app);

// ── WEBSOCKET SERVER ──
const wss = new WebSocketServer({ server });

// ══════════════════════════════════════════
// GAME ENGINE (copied from pitch_game.html)
// ══════════════════════════════════════════
const SUIT_NAMES = ['Clubs','Diamonds','Hearts','Spades'];
const SUIT_SYMBOLS = ['♣','♦','♥','♠'];
const RANK_NAMES = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A'};
const GAME_POINTS = {14:4,13:3,12:2,11:1,10:10};

class Card {
  constructor(r,s) { this.rank = r; this.suit = s; }
  get gamePoints() { return GAME_POINTS[this.rank] || 0; }
  get id() { return this.suit * 100 + this.rank; }
  get name() { return (RANK_NAMES[this.rank] || this.rank) + SUIT_SYMBOLS[this.suit]; }
  eq(o) { return o && this.rank === o.rank && this.suit === o.suit; }
  toJSON() { return { rank: this.rank, suit: this.suit }; }
  static from(o) { return new Card(o.rank, o.suit); }
}

function makeDeck() { const d = []; for (let s = 0; s < 4; s++) for (let r = 2; r <= 14; r++) d.push(new Card(r,s)); return d; }
function shuffle(a) { for (let i = a.length-1; i > 0; i--) { const j = Math.floor(Math.random()*(i+1)); [a[i],a[j]] = [a[j],a[i]]; } return a; }
function dealHands() { const d = shuffle(makeDeck()); const h = []; for (let i = 0; i < 4; i++) h.push(d.slice(i*6, i*6+6).sort((a,b) => a.suit-b.suit || a.rank-b.rank)); return h; }
function teamOf(p) { return p % 2; }

function legalPlays(h, tc, tr) {
  if (tc.length === 0) return [...h];
  const ls = tc[0].card.suit;
  if (ls === tr) { const f = h.filter(c => c.suit === tr); return f.length > 0 ? f : [...h]; }
  const f = h.filter(c => c.suit === ls);
  if (f.length > 0) { return h.filter(c => c.suit === ls || c.suit === tr); }
  return [...h];
}

function beats(ch, best, tr) {
  if (ch.suit === tr && best.suit !== tr) return true;
  if (ch.suit !== tr && best.suit === tr) return false;
  if (ch.suit === best.suit) return ch.rank > best.rank;
  return false;
}

function trickWinner(cards, tr) {
  let b = 0;
  for (let i = 1; i < cards.length; i++) if (beats(cards[i].card, cards[b].card, tr)) b = i;
  return cards[b].player;
}

class BidState {
  constructor(d) { this.dealer = d; this.currentBid = 0; this.currentBidder = null; this.bids = []; this.complete = false; }
  biddingOrder() { return [0,1,2,3].map(i => (this.dealer + 1 + i) % 4); }
  legalBids(p) {
    if (p === this.dealer) {
      if (this.currentBid === 0) return [2, 5];
      return [0, this.currentBid, 5].filter((v,i,a) => a.indexOf(v) === i).sort((a,b) => a - b);
    } else { const o = [0]; for (let b = Math.max(2, this.currentBid + 1); b <= 5; b++) o.push(b); return o; }
  }
  placeBid(p, v) {
    this.bids.push({ player: p, bid: v });
    if (v > 0) { this.currentBid = v; this.currentBidder = p; }
    if (this.bids.length === 4) { this.complete = true; if (this.currentBidder === null) { this.currentBid = 2; this.currentBidder = this.dealer; } }
  }
}

function computeScoring(tricks, tr) {
  const cap = {0:[], 1:[]}, tw = {0:0, 1:0};
  for (const t of tricks) { const w = trickWinner(t.cards, tr); const tm = teamOf(w); tw[tm]++; t.cards.forEach(tc => cap[tm].push(tc.card)); }
  const trc = {0: cap[0].filter(c => c.suit === tr), 1: cap[1].filter(c => c.suit === tr)};
  const at = [...trc[0], ...trc[1]];
  let hi = null, hiT = null, lo = null, loT = null, jT = null, jC = false, gT = null;
  if (at.length) {
    const h = at.reduce((a,b) => a.rank > b.rank ? a : b); hi = h; hiT = trc[0].some(c => c.eq(h)) ? 0 : 1;
    const l = at.reduce((a,b) => a.rank < b.rank ? a : b); lo = l; loT = trc[0].some(c => c.eq(l)) ? 0 : 1;
  }
  const j = new Card(11, tr);
  for (let t = 0; t < 2; t++) if (cap[t].some(c => c.eq(j))) { jC = true; jT = t; }
  const gp = {0: cap[0].reduce((s,c) => s + c.gamePoints, 0), 1: cap[1].reduce((s,c) => s + c.gamePoints, 0)};
  if (gp[0] > gp[1]) gT = 0; else if (gp[1] > gp[0]) gT = 1;
  const pts = {0:0, 1:0};
  if (hiT !== null) pts[hiT]++;
  if (loT !== null) pts[loT]++;
  if (jC) pts[jT]++;
  if (gT !== null) pts[gT]++;
  return { high: hi, highTeam: hiT, low: lo, lowTeam: loT, jackCaptured: jC, jackTeam: jT, gameTeam: gT, gp, points: pts, tricksWon: tw, captured: cap };
}

// ══════════════════════════════════════════
// GAME ROOM
// ══════════════════════════════════════════
const room = {
  players: [],       // [{ws, name, seat, connected}]
  teams: [0,1,0,1],  // seat→team mapping
  phase: 'lobby',    // lobby | bidding | playing | hand-result | game-over
  dealer: 0,
  hands: [null,null,null,null],
  initialHands: [null,null,null,null],  // saved at deal for training data
  allBids: [],       // all bids placed this hand [{seat, bid}]
  bidState: null,
  bidTurnIdx: 0,
  trump: null,
  tricks: [],
  currentTrick: [],
  leader: null,
  bidder: null,
  bidAmount: 0,
  scores: {0:0, 1:0},
  handNum: 0,
  trickNum: 0,
  playTurnIdx: 0,
  hostSeat: 0,
};

function broadcast(msg) {
  const s = JSON.stringify(msg);
  for (const p of room.players) { if (p && p.ws && p.ws.readyState === 1) p.ws.send(s); }
}

function sendTo(seat, msg) {
  const p = room.players[seat];
  if (p && p.ws && p.ws.readyState === 1) p.ws.send(JSON.stringify(msg));
}

function broadcastLobby() {
  broadcast({
    type: 'lobby',
    players: room.players.map((p, i) => p ? { name: p.name, seat: i, connected: p.connected, team: room.teams[i] } : null),
    hostSeat: room.hostSeat,
    dealer: room.dealer,
  });
}

function playerCount() { return room.players.filter(p => p && p.connected).length; }

// ── GAME ACTIONS ──

function startGame() {
  room.phase = 'playing';
  room.scores = {0:0, 1:0};
  room.handNum = 0;
  startHand();
}

function startHand() {
  room.handNum++;
  room.hands = dealHands();
  room.initialHands = room.hands.map(h => h.map(c => ({ rank: c.rank, suit: c.suit })));
  room.allBids = [];
  room.bidState = new BidState(room.dealer);
  room.trump = null;
  room.tricks = [];
  room.currentTrick = [];
  room.bidder = null;
  room.bidAmount = 0;
  room.trickNum = 0;
  room.phase = 'bidding';
  room.bidTurnIdx = 0;

  for (let s = 0; s < 4; s++) {
    sendTo(s, {
      type: 'hand-dealt',
      hand: room.hands[s],
      dealer: room.dealer,
      handNum: room.handNum,
      scores: room.scores,
      teams: room.teams,
      players: room.players.map((p, i) => p ? { name: p.name, seat: i, team: room.teams[i] } : null),
    });
  }

  advanceBid();
}

function advanceBid() {
  if (room.bidState.complete) {
    room.bidder = room.bidState.currentBidder;
    room.bidAmount = room.bidState.currentBid;
    room.leader = room.bidder;
    room.phase = 'playing';
    room.trickNum = 0;

    broadcast({
      type: 'bid-won',
      bidder: room.bidder,
      bidAmount: room.bidAmount,
      bidderName: room.players[room.bidder].name,
    });

    setTimeout(() => advancePlay(), 500);
    return;
  }

  const order = room.bidState.biddingOrder();
  const seat = order[room.bidTurnIdx];
  const legal = room.bidState.legalBids(seat);

  broadcast({
    type: 'bid-turn',
    seat: seat,
    playerName: room.players[seat].name,
  });

  sendTo(seat, {
    type: 'your-bid',
    legalBids: legal,
    currentBid: room.bidState.currentBid,
  });
}

function handleBid(seat, bid) {
  if (room.phase !== 'bidding') return;
  const order = room.bidState.biddingOrder();
  if (order[room.bidTurnIdx] !== seat) return;
  const legal = room.bidState.legalBids(seat);
  if (!legal.includes(bid)) return;

  room.bidState.placeBid(seat, bid);
  room.allBids.push({ seat, bid });
  broadcast({
    type: 'bid-placed',
    seat: seat,
    bid: bid,
    playerName: room.players[seat].name,
  });

  room.bidTurnIdx++;
  setTimeout(() => advanceBid(), 400);
}

function advancePlay() {
  if (room.trickNum >= 6) {
    finishHand();
    return;
  }

  if (room.currentTrick.length >= 4) {
    const winner = trickWinner(room.currentTrick, room.trump);
    room.tricks.push({ cards: [...room.currentTrick] });

    broadcast({
      type: 'trick-complete',
      winner: winner,
      winnerName: room.players[winner].name,
      cards: room.currentTrick,
      trickNum: room.trickNum,
    });

    room.leader = winner;
    room.currentTrick = [];
    room.trickNum++;
    room.playTurnIdx = 0;

    setTimeout(() => advancePlay(), 1200);
    return;
  }

  const seat = (room.leader + room.currentTrick.length) % 4;
  const hand = room.hands[seat];
  const tc = room.currentTrick.map(x => ({ player: x.player, card: Card.from(x.card) }));

  const isPitch = room.trickNum === 0 && room.currentTrick.length === 0 && room.trump === null;
  const legal = isPitch ? [...hand] : legalPlays(hand, tc, room.trump);

  broadcast({
    type: 'play-turn',
    seat: seat,
    playerName: room.players[seat].name,
    trickNum: room.trickNum,
    currentTrick: room.currentTrick,
  });

  sendTo(seat, {
    type: 'your-turn',
    legalPlays: legal,
    isPitch: isPitch,
    currentTrick: room.currentTrick,
  });
}

function handlePlay(seat, cardData) {
  if (room.phase !== 'playing') return;
  const expectedSeat = (room.leader + room.currentTrick.length) % 4;
  if (seat !== expectedSeat) return;

  const card = Card.from(cardData);
  const hand = room.hands[seat];
  const handIdx = hand.findIndex(c => c.eq(card));
  if (handIdx === -1) return;

  const isPitch = room.trickNum === 0 && room.currentTrick.length === 0 && room.trump === null;
  const tc = room.currentTrick.map(x => ({ player: x.player, card: Card.from(x.card) }));

  if (isPitch) {
    room.trump = card.suit;
    broadcast({ type: 'trump-set', trump: room.trump, pitcherName: room.players[seat].name });
  }

  const legal = isPitch ? [...hand] : legalPlays(hand, tc, room.trump);
  if (!legal.some(c => c.eq(card))) return;

  room.hands[seat].splice(handIdx, 1);
  room.currentTrick.push({ player: seat, card: card });

  broadcast({
    type: 'card-played',
    seat: seat,
    card: card,
    playerName: room.players[seat].name,
    cardsLeft: room.hands[seat].length,
  });

  sendTo(seat, { type: 'hand-update', hand: room.hands[seat] });

  setTimeout(() => advancePlay(), 400);
}

function finishHand() {
  const scoring = computeScoring(room.tricks, room.trump);
  const bidTeam = teamOf(room.bidder);
  const bidPts = scoring.points[bidTeam];
  const made = room.bidAmount === 5 ? scoring.tricksWon[bidTeam] >= 6 : bidPts >= room.bidAmount;

  if (made) {
    room.scores[bidTeam] += room.bidAmount === 5 ? 5 : bidPts;
  } else {
    room.scores[bidTeam] -= room.bidAmount;
  }
  const defTeam = 1 - bidTeam;
  room.scores[defTeam] += scoring.points[defTeam];

  let winner = null;
  if (room.scores[bidTeam] >= 21 && made) {
    winner = bidTeam;
  } else if (room.scores[defTeam] >= 21 && !(room.scores[bidTeam] >= 21 && made)) {
    winner = defTeam;
  }

  room.phase = 'hand-result';

  // Build trick history for training data
  const trickHistory = room.tricks.map(t => t.cards.map(tc => ({ player: tc.player, card: tc.card })));

  broadcast({
    type: 'hand-result',
    scoring: {
      high: scoring.high, highTeam: scoring.highTeam,
      low: scoring.low, lowTeam: scoring.lowTeam,
      jackCaptured: scoring.jackCaptured, jackTeam: scoring.jackTeam,
      gameTeam: scoring.gameTeam, gp: scoring.gp,
      points: scoring.points, tricksWon: scoring.tricksWon,
    },
    bidder: room.bidder,
    bidAmount: room.bidAmount,
    made: made,
    scores: room.scores,
    winner: winner,
    teams: room.teams,
    players: room.players.map((p,i) => p ? { name: p.name, seat: i, team: room.teams[i] } : null),
    // Full training data — all 4 hands, all bids, all tricks
    trainingData: {
      initialHands: room.initialHands,
      allBids: room.allBids,
      tricks: trickHistory,
      trump: room.trump,
      dealer: room.dealer,
    },
  });

  if (winner !== null) {
    room.phase = 'game-over';
  }
}

function nextHand() {
  if (room.phase === 'game-over') return;
  room.dealer = (room.dealer + 1) % 4;
  startHand();
}

function newGame() {
  room.scores = {0:0, 1:0};
  room.handNum = 0;
  room.dealer = Math.floor(Math.random() * 4);
  room.phase = 'lobby';
  broadcastLobby();
}

// ══════════════════════════════════════════
// WEBSOCKET CONNECTION HANDLER
// ══════════════════════════════════════════
wss.on('connection', (ws) => {
  let mySeat = -1;
  console.log('  → New WebSocket connection');

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch(e) { return; }

    switch (msg.type) {
      case 'join': {
        let seat = room.players.findIndex(p => p && p.name === msg.name && !p.connected);
        if (seat === -1) {
          seat = room.players.findIndex(p => p === null);
          if (seat === -1 && room.players.length < 4) seat = room.players.length;
          if (seat === -1 || seat >= 4) { ws.send(JSON.stringify({ type: 'error', message: 'Game is full' })); return; }
          while (room.players.length <= seat) room.players.push(null);
          room.players[seat] = { ws, name: msg.name, seat, connected: true };
        } else {
          room.players[seat].ws = ws;
          room.players[seat].connected = true;
        }
        mySeat = seat;
        console.log(`  → ${msg.name} joined as Seat ${seat+1} (${playerCount()}/4 players)`);
        ws.send(JSON.stringify({ type: 'joined', seat: mySeat, name: msg.name }));
        broadcastLobby();

        // Notify everyone this player reconnected
        broadcast({ type: 'player-reconnected', seat: mySeat, name: msg.name });

        if (room.phase !== 'lobby') {
          // Resend full game state to reconnected player
          sendTo(mySeat, {
            type: 'hand-dealt',
            hand: room.hands[mySeat],
            dealer: room.dealer,
            handNum: room.handNum,
            scores: room.scores,
            teams: room.teams,
            players: room.players.map((p,i) => p ? { name: p.name, seat: i, team: room.teams[i] } : null),
          });
          if (room.trump !== null) sendTo(mySeat, { type: 'trump-set', trump: room.trump, pitcherName: room.players[room.bidder] ? room.players[room.bidder].name : '' });
          if (room.bidder !== null) sendTo(mySeat, { type: 'bid-won', bidder: room.bidder, bidAmount: room.bidAmount, bidderName: room.players[room.bidder].name });
          if (room.currentTrick.length > 0) {
            sendTo(mySeat, { type: 'trick-state', currentTrick: room.currentTrick, trickNum: room.trickNum });
          }

          // Re-trigger current turn if it was waiting on someone
          if (room.phase === 'bidding') {
            const order = room.bidState.biddingOrder();
            const currentBidSeat = order[room.bidTurnIdx];
            // Re-broadcast whose turn it is
            broadcast({ type: 'bid-turn', seat: currentBidSeat, playerName: room.players[currentBidSeat].name });
            if (currentBidSeat === mySeat) {
              const legal = room.bidState.legalBids(mySeat);
              sendTo(mySeat, { type: 'your-bid', legalBids: legal, currentBid: room.bidState.currentBid });
            }
          } else if (room.phase === 'playing') {
            const currentPlaySeat = (room.leader + room.currentTrick.length) % 4;
            broadcast({ type: 'play-turn', seat: currentPlaySeat, playerName: room.players[currentPlaySeat].name, trickNum: room.trickNum, currentTrick: room.currentTrick });
            if (currentPlaySeat === mySeat) {
              const hand = room.hands[mySeat];
              const tc = room.currentTrick.map(x => ({ player: x.player, card: Card.from(x.card) }));
              const isPitch = room.trickNum === 0 && room.currentTrick.length === 0 && room.trump === null;
              const legal = isPitch ? [...hand] : legalPlays(hand, tc, room.trump);
              sendTo(mySeat, { type: 'your-turn', legalPlays: legal, isPitch, currentTrick: room.currentTrick });
            }
          } else if (room.phase === 'hand-result') {
            // Resend hand result so they see the score screen
            sendTo(mySeat, { type: 'hand-result', scoring: { highTeam: null, lowTeam: null, jackCaptured: false, jackTeam: null, gameTeam: null, gp: {0:0,1:0}, points: {0:0,1:0}, tricksWon: {0:0,1:0} }, bidder: room.bidder, bidAmount: room.bidAmount, made: true, scores: room.scores, winner: null, teams: room.teams, players: room.players.map((p,i) => p ? { name: p.name, seat: i, team: room.teams[i] } : null) });
          }
        }
        break;
      }

      case 'randomize-teams': {
        if (mySeat !== room.hostSeat) return;
        room.teams = shuffle([0,1,0,1]);
        broadcastLobby();
        break;
      }

      case 'randomize-dealer': {
        if (mySeat !== room.hostSeat) return;
        room.dealer = Math.floor(Math.random() * 4);
        broadcastLobby();
        break;
      }

      case 'start-game': {
        if (mySeat !== room.hostSeat) return;
        if (playerCount() < 4) { ws.send(JSON.stringify({ type: 'error', message: 'Need 4 players' })); return; }
        startGame();
        break;
      }

      case 'bid': {
        handleBid(mySeat, msg.bid);
        break;
      }

      case 'play': {
        handlePlay(mySeat, msg.card);
        break;
      }

      case 'next-hand': {
        nextHand();
        break;
      }

      case 'new-game': {
        if (mySeat !== room.hostSeat) return;
        newGame();
        break;
      }
    }
  });

  ws.on('close', () => {
    console.log(`  → Connection closed (Seat ${mySeat >= 0 ? mySeat+1 : '?'})`);
    if (mySeat >= 0 && room.players[mySeat]) {
      room.players[mySeat].connected = false;
      broadcast({ type: 'player-disconnected', seat: mySeat, name: room.players[mySeat].name });
      if (room.phase === 'lobby') {
        room.players[mySeat] = null;
        while (room.players.length > 0 && room.players[room.players.length-1] === null) room.players.pop();
        broadcastLobby();
      }
    }
  });

  ws.on('error', (err) => {
    console.log(`  → WebSocket error: ${err.message}`);
  });
});

// ══════════════════════════════════════════
// START SERVER
// ══════════════════════════════════════════
server.listen(PORT, '0.0.0.0', () => {
  console.log(`\n  Pitch Multiplayer Server running on port ${PORT}`);
  console.log(`  Local:   http://localhost:${PORT}`);
  const os = require('os');
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        console.log(`  Network: http://${net.address}:${PORT}`);
      }
    }
  }
  console.log(`\n  Waiting for 4 players to join...\n`);
});
