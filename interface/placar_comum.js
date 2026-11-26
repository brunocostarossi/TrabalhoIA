// Código comum do placar: usado por placar.html e simulador.html.
//
// "estado" é a descrição de um momento do round, no mesmo formato dos totais da base:
// { mapa: "de_mirage", placar_ct, placar_t, tempo (s), bomba (bool),
//   ct: { vivos, vida, colete, capacetes, kits, itens: { ak47: 2, flashbang: 3, ... } },
//   t:  { vivos, vida, colete, capacetes, itens } }
(function () {
  "use strict";

  const LOGO = { ct: "logo_ct.png", t: "logo_t.png" };
  const NOME_TIME = { ct: "Contra-Terroristas", t: "Terroristas" };

  const MAPAS = {
    de_dust2: "Dust2", de_mirage: "Mirage", de_inferno: "Inferno", de_nuke: "Nuke",
    de_overpass: "Overpass", de_vertigo: "Vertigo", de_train: "Train", de_cache: "Cache",
  };
  // Armas principais por categoria (na ordem em que aparecem no simulador)
  const CATEGORIAS = [
    ["Fuzis", { ak47: "AK-47", m4a4: "M4A4", m4a1s: "M4A1-S", galilar: "Galil AR", famas: "FAMAS", sg553: "SG 553", aug: "AUG" }],
    ["Snipers", { awp: "AWP", ssg08: "SSG 08", g3sg1: "G3SG1", scar20: "SCAR-20" }],
    ["SMGs", { mac10: "MAC-10", mp9: "MP9", mp7: "MP7", mp5sd: "MP5-SD", ump45: "UMP-45", p90: "P90", bizon: "PP-Bizon" }],
    ["Pesadas", { nova: "Nova", xm1014: "XM1014", sawedoff: "Sawed-Off", mag7: "MAG-7", m249: "M249", negev: "Negev" }],
  ];
  const ARMAS = Object.assign({}, ...CATEGORIAS.map(([, armas]) => armas));
  const PISTOLAS = {
    glock: "Glock-18", usps: "USP-S", p2000: "P2000", p250: "P250", fiveseven: "Five-SeveN",
    tec9: "Tec-9", cz75auto: "CZ75-Auto", deagle: "Desert Eagle", elite: "Dual Berettas", r8revolver: "R8",
  };
  // [sigla, nome, classe, máximo por jogador]
  const GRANADAS = {
    hegrenade: ["HE", "Explosiva (HE)", "he", 1],
    flashbang: ["FB", "Flashbang", "fb", 2],
    smokegrenade: ["SM", "Fumaça", "sm", 1],
    molotovgrenade: ["MO", "Molotov", "mo", 1],
    incendiarygrenade: ["IN", "Incendiária", "in", 1],
    decoygrenade: ["DC", "Decoy", "dc", 1],
  };
  const ORDEM_GRANADAS = Object.keys(GRANADAS);
  const MAX_GRANADAS = 4;            // por jogador
  const ORDEM_MORTES = [2, 4, 1, 5, 3]; // número do jogador, na ordem em que morre

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const limitar = (v, min, max) => Math.min(Math.max(v, min), max);
  const ehFogo = (g) => g === "molotovgrenade" || g === "incendiarygrenade";

  function fmtTempo(seg) {
    const t = Math.max(0, Math.ceil(seg));
    return Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
  }

  // --- Sorteio reproduzível (mulberry32) ---
  function geradorAleatorio(semente) {
    let a = semente >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function embaralhar(lista, rng) {
    const v = lista.slice();
    for (let i = v.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [v[i], v[j]] = [v[j], v[i]];
    }
    return v;
  }
  // {ak47: 2, glock: 1} -> ["ak47", "ak47", "glock"], só com as chaves de `catalogo`
  function expandir(itens, catalogo) {
    const lista = [];
    for (const k in itens) if (catalogo[k]) for (let i = 0; i < itens[k]; i++) lista.push(k);
    return lista;
  }

  // Divide `total` pontos (vida ou colete) entre `n` jogadores que começam com 100 cada:
  // o dano (100·n − total) é dividido igualmente, arredondado para cima.
  function dividirPontos(total, n) {
    if (n === 0) return 0;
    const dano = Math.max(0, 100 * n - total);
    return limitar(100 - Math.ceil(dano / n), 1, 100);
  }

  // Quem está vivo com `n` vivos: morrem os primeiros 5 − n de ORDEM_MORTES.
  function vivosPorJogador(n) {
    const mortos = ORDEM_MORTES.slice(0, 5 - limitar(n, 0, 5));
    return [1, 2, 3, 4, 5].map((num) => !mortos.includes(num));
  }

  /** Transforma os totais de um time (o que a base tem) em 5 jogadores. */
  function distribuir(d, lado, rng) {
    const n = limitar(d.vivos, 0, 5);
    const jogadores = vivosPorJogador(n).map((vivo) => ({
      vivo, vida: 0, colete: 0, capacete: false, kit: false, arma: null, tipoArma: null, granadas: [],
    }));
    if (n === 0) return jogadores;
    const vivos = jogadores.filter((j) => j.vivo);

    // Vida: igual para todos os vivos
    const vida = dividirPontos(limitar(d.vida, 0, 100 * n), n);
    vivos.forEach((j) => { j.vida = vida; });

    // Colete: dividido igualmente entre os vivos, como a vida. Se cada um ficaria com
    // mais de 0 e até 80, o maior número possível de jogadores fica com 100 e o
    // que sobrar vai para um único outro jogador. Quem recebe é sorteado.
    const totalColete = limitar(d.colete, 0, 100 * n);
    const igual = dividirPontos(totalColete, n);
    let coletes;
    if (totalColete === 0) coletes = [];
    else if (igual > 80) coletes = Array(n).fill(igual);
    else {
      const cheios = Math.floor(totalColete / 100), resto = totalColete - 100 * cheios;
      coletes = Array(cheios).fill(100);
      if (resto > 0) coletes.push(resto);
    }
    // O capacete só vai para quem tem colete; a ordem já é sorteada,
    // então os primeiros `capacetes` com colete também são um sorteio.
    const capacetes = Math.min(limitar(d.capacetes, 0, n), coletes.length);
    embaralhar(vivos, rng).slice(0, coletes.length).forEach((j, i) => {
      j.colete = coletes[i]; j.capacete = i < capacetes;
    });

    // Armas: uma por jogador. Principais primeiro; pistola só para quem ficou sem principal.
    const primarias = embaralhar(expandir(d.itens, ARMAS), rng);
    const pistolas = embaralhar(expandir(d.itens, PISTOLAS), rng);
    embaralhar(vivos, rng).forEach((j) => {
      if (primarias.length) { j.arma = ARMAS[primarias.pop()]; j.tipoArma = "primaria"; }
      else if (pistolas.length) { j.arma = PISTOLAS[pistolas.pop()]; j.tipoArma = "pistola"; }
      else { j.arma = "Faca"; j.tipoArma = "faca"; }
    });

    // Granadas: sorteio respeitando o limite do jogo (4 por jogador, 2 flashes,
    // 1 de cada outro tipo e só uma entre molotov/incendiária).
    const cabe = (j, g) => {
      if (j.granadas.length >= MAX_GRANADAS) return false;
      if (ehFogo(g)) return !j.granadas.some(ehFogo);
      return j.granadas.filter((x) => x === g).length < GRANADAS[g][3];
    };
    embaralhar(expandir(d.itens, GRANADAS), rng).forEach((g) => {
      const candidatos = vivos.filter((j) => cabe(j, g));
      if (candidatos.length) candidatos[Math.floor(rng() * candidatos.length)].granadas.push(g);
    });
    vivos.forEach((j) => j.granadas.sort((a, b) => ORDEM_GRANADAS.indexOf(a) - ORDEM_GRANADAS.indexOf(b)));

    // Kits (só CT)
    if (lado === "ct") {
      embaralhar(vivos, rng).slice(0, limitar(d.kits || 0, 0, n)).forEach((j) => { j.kit = true; });
    }
    return jogadores;
  }

  // --- Modelo (árvores exportadas pelo passo 12) ---
  function vetorDoModelo(modelo, e) {
    const x = { time_left: e.tempo, ct_score: e.placar_ct, t_score: e.placar_t, bomb_planted: e.bomba ? 1 : 0 };
    x["map_" + e.mapa] = 1;
    for (const lado of ["ct", "t"]) {
      const d = e[lado];
      x[lado + "_health"] = d.vida;
      x[lado + "_armor"] = d.colete;
      x[lado + "_helmets"] = d.capacetes;
      x[lado + "_players_alive"] = d.vivos;
      let valor = d.colete / 100 * modelo.preco_colete + d.capacetes * modelo.preco_capacete;
      for (const item in d.itens) {
        x[lado + (GRANADAS[item] ? "_grenade_" : "_weapon_") + item] = d.itens[item];
        valor += d.itens[item] * (modelo.precos[item] || 0);
      }
      if (lado === "ct") {
        x.ct_defuse_kits = d.kits;
        valor += d.kits * modelo.preco_kit;
      }
      x[lado + "_equip_valor"] = valor;
    }
    // Mesmos atributos de criar_atributos (preparacao.py), sem o de dinheiro
    x.dif_jogadores = x.ct_players_alive - x.t_players_alive;
    x.dif_vida = x.ct_health - x.t_health;
    x.dif_armadura = x.ct_armor - x.t_armor;
    x.dif_equip_valor = x.ct_equip_valor - x.t_equip_valor;
    x.bomba_x_tempo = x.bomb_planted * x.time_left;
    return modelo.colunas.map((c) => x[c] || 0);
  }

  function probabilidadeCT(modelo, estado) {
    const v = vetorDoModelo(modelo, estado);
    let bruto = modelo.base;
    for (const a of modelo.arvores) {
      let no = 0;
      while (!a.folha[no]) {
        const valor = v[a.f[no]];
        const esquerda = Number.isNaN(valor) ? a.m[no] === 1 : valor <= a.t[no];
        no = esquerda ? a.e[no] : a.d[no];
      }
      bruto += a.v[no];
    }
    return 1 / (1 + Math.exp(-bruto));
  }

  // --- Desenho do cartão (placar + tabelas + chance) ---
  const SIMBOLOS = `
  <svg width="0" height="0" style="position:absolute" aria-hidden="true">
    <symbol id="ic-caveira" viewBox="0 0 16 16">
      <path d="M8 1.5C4.4 1.5 2 3.9 2 7c0 1.9.9 3.3 2.2 4.1V13a1 1 0 0 0 1 1h5.6a1 1 0 0 0 1-1v-1.9C13.1 10.3 14 8.9 14 7c0-3.1-2.4-5.5-6-5.5Z" fill="currentColor"/>
      <circle cx="5.7" cy="7.4" r="1.5" fill="#1b1f24"/><circle cx="10.3" cy="7.4" r="1.5" fill="#1b1f24"/>
      <path d="M7 11.5h2" stroke="#1b1f24" stroke-width="1.2"/>
    </symbol>
    <!-- Capacete: quadrado escuro com o capacete branco (desenho enviado pelo usuário) -->
    <symbol id="ic-capacete" viewBox="0 0 20 20">
      <rect x="0.5" y="0.5" width="19" height="19" rx="4" fill="#1d2c3d"/>
      <path d="M3.6 13.6V10.4C3.6 6.6 6.4 4 10 4s6.4 2.6 6.4 6.4v3.2h-3.9L10 10.6l-2.5 3H3.6Z" fill="#f4f6f8"/>
    </symbol>
    <!-- Kit de desarme: alicate de bico (desenho enviado pelo usuário) -->
    <symbol id="ic-kit" viewBox="0 0 48 41">
      <path d="M6.6 2.4 15 11" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>
      <path d="M2.4 7.6 14 12.6" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>
      <circle cx="16" cy="12.4" r="5" fill="currentColor"/>
      <path d="M17 10.2C25 8.4 33 12.6 44.6 17.4" fill="none" stroke="currentColor" stroke-width="6.6" stroke-linecap="round"/>
      <path d="M16.4 15.4C17 24.4 21.4 32 28.4 37.8" fill="none" stroke="currentColor" stroke-width="6.6" stroke-linecap="round"/>
      <path d="M23.4 17.6l3 .9M23 20.1l2.6 1.1M22.8 22.4l2 1.1" stroke="currentColor" stroke-width="1.1" stroke-linecap="round"/>
    </symbol>
    <symbol id="ic-bomba" viewBox="0 0 24 24">
      <rect x="2" y="8" width="20" height="13" rx="2" fill="currentColor"/>
      <rect x="5" y="11" width="8" height="4.5" rx=".8" fill="#0b0d10"/>
      <path d="M15.5 11h3M15.5 13.3h3M15.5 15.6h3" stroke="#0b0d10" stroke-width="1.3" stroke-linecap="round"/>
      <path d="M7 8c0-3 2-5 5-5s4 1.6 5 3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    </symbol>
  </svg>`;

  function incluirSimbolos() {
    if (!$("ic-bomba")) document.body.insertAdjacentHTML("beforeend", SIMBOLOS);
  }

  function criarCartao(container) {
    incluirSimbolos();
    container.innerHTML = `
      <div class="placar" id="placar">
        <div class="lado ct">
          <span class="nome"><span class="longo">Contra-Terroristas</span><span class="curto">CT</span></span>
          <span class="pontos" id="pts-ct">0</span>
          <img class="logo" src="${LOGO.ct}" alt="">
        </div>
        <div class="centro">
          <div class="tempo">
            <svg viewBox="0 0 24 24" class="icone-bomba" id="icone-bomba" role="img"><use href="#ic-bomba"/></svg>
            <span class="relogio" id="relogio">0:00</span>
          </div>
          <span class="rodada" id="rodada">Rodada 1</span>
        </div>
        <div class="lado t">
          <img class="logo" src="${LOGO.t}" alt="">
          <span class="pontos" id="pts-t">0</span>
          <span class="nome"><span class="longo">Terroristas</span><span class="curto">TR</span></span>
        </div>
      </div>
      <div class="rolagem"><section class="time t" id="time-t" aria-label="Terroristas"></section></div>
      <div class="rolagem"><section class="time ct" id="time-ct" aria-label="Contra-Terroristas"></section></div>
      <div class="chance" aria-live="polite">
        <div class="chance-titulo">Chance de vencer a rodada</div>
        <div class="chance-linha">
          <div class="fora ct" id="fora-ct"></div>
          <div class="barra" id="barra">
            <div class="seg ct" id="seg-ct"><span id="lbl-ct"></span></div>
            <div class="seg t" id="seg-t"><span id="lbl-t"></span></div>
          </div>
          <div class="fora t" id="fora-t"></div>
        </div>
      </div>`;
    new ResizeObserver(() => { if (ultimaProb !== null) renderChance(ultimaProb); }).observe($("barra"));
  }

  const icone = (id, classe, titulo) =>
    `<svg class="${classe}" role="img" aria-label="${titulo}"><title>${titulo}</title><use href="#${id}"/></svg>`;

  function linhaJogador(j, i) {
    const icones = (j.kit ? icone("ic-kit", "kit", "Kit de desarme") : "") +
                   (j.vivo ? "" : icone("ic-caveira", "caveira", "Morto"));
    const arma = j.vivo
      ? `<div class="c arma ${j.tipoArma === "primaria" ? "" : j.tipoArma}">${esc(j.arma)}</div>`
      : '<div class="c"></div>';
    const gran = j.granadas.map((g) => {
      const [sigla, nome, classe] = GRANADAS[g];
      return `<span class="gr ${classe}" title="${nome}">${sigla}</span>`;
    }).join("");
    const colete = j.colete > 0
      ? `<div class="c colete"><span class="num">${j.colete}</span>${j.capacete ? icone("ic-capacete", "", "Capacete") : ""}</div>`
      : '<div class="c colete sem"><span class="num">0</span></div>';

    return `<div class="linha jog${j.vivo ? "" : " morto"}">` +
      `<div class="c nome"><span>Jogador ${i + 1}</span><span class="icones">${icones}</span></div>` +
      arma +
      `<div class="c gran">${gran}</div>` +
      `<div class="c"><div class="hp${j.vida > 0 && j.vida <= 20 ? " baixa" : ""}"><i style="width:${j.vida}%"></i><span>${j.vida}</span></div></div>` +
      colete +
      "</div>";
  }

  function renderTime(lado, d, semente) {
    const jogadores = distribuir(d, lado, geradorAleatorio(semente));
    $("time-" + lado).innerHTML =
      `<div class="linha cab"><div class="c equipe"><img class="logo" src="${LOGO[lado]}" alt="">${NOME_TIME[lado]}</div>` +
      '<div class="c">Arma</div><div class="c">Granadas</div><div class="c">Vida</div><div class="c">Colete</div></div>' +
      jogadores.map(linhaJogador).join("");
  }

  // --- Barra de chance: número dentro da cor; se não couber, "< x%" do lado de fora ---
  const medidor = document.createElement("canvas").getContext("2d");
  const FOLGA = 8; // espaço mínimo (px) entre o número e cada borda da cor
  let ultimaProb = null;

  function larguraTexto(texto, el) {
    const cs = getComputedStyle(el);
    medidor.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    return medidor.measureText(texto).width;
  }

  // Menor x tal que TODO valor de x% a 100% cabe dentro da própria cor.
  // (Testa de cima para baixo porque "10%" é mais largo que "9%": sem isso,
  // 9% poderia caber e 10% não.)
  function limiar(larguraBarra, el) {
    let x = 101;
    for (let k = 100; k >= 1; k--) {
      if (larguraBarra * k / 100 >= larguraTexto(k + "%", el) + 2 * FOLGA) x = k;
      else break;
    }
    return x;
  }

  function renderChance(probCt) {
    ultimaProb = probCt;
    const pCt = Math.round(probCt * 100), pT = 100 - pCt;
    $("seg-ct").style.width = pCt + "%";

    const largura = $("barra").clientWidth;
    for (const [lado, p, nome] of [["ct", pCt, "CT"], ["t", pT, "TR"]]) {
      const lbl = $("lbl-" + lado), fora = $("fora-" + lado);
      lbl.textContent = p + "%";
      const x = limiar(largura, lbl);
      if (p >= x) {
        lbl.hidden = false;
        fora.innerHTML = `<img class="logo" src="${LOGO[lado]}" alt="${nome}">`;
        fora.title = "";
      } else {
        lbl.hidden = true;
        fora.textContent = "< " + x + "%";
        fora.title = `${nome}: ${p}%`;
      }
    }
    $("barra").setAttribute("aria-label", `Chance de vitória: CT ${pCt}%, TR ${pT}%`);
  }

  /** Mostra um estado no cartão criado por criarCartao. */
  function mostrar(estado, probCt, semente) {
    $("pts-ct").textContent = estado.placar_ct;
    $("pts-t").textContent = estado.placar_t;
    $("relogio").textContent = fmtTempo(estado.tempo);
    $("rodada").textContent = `Rodada ${estado.placar_ct + estado.placar_t + 1} · ${MAPAS[estado.mapa] || estado.mapa}`;
    $("placar").classList.toggle("bomba", estado.bomba);
    $("icone-bomba").setAttribute("aria-label", estado.bomba ? "Bomba plantada" : "Bomba não plantada");
    renderTime("t", estado.t, semente);
    renderTime("ct", estado.ct, semente + 1);
    renderChance(probCt);
  }

  window.Placar = {
    LOGO, NOME_TIME, MAPAS, CATEGORIAS, ARMAS, PISTOLAS, GRANADAS, MAX_GRANADAS, ORDEM_MORTES,
    $, esc, limitar, ehFogo, fmtTempo, dividirPontos, vivosPorJogador, distribuir,
    incluirSimbolos, criarCartao, mostrar, probabilidadeCT,
  };
})();
