/**
 * Scraper de disciplinas da FSP/USP via Jupiterweb.
 *
 * Fluxo:
 *   1. Acessa jupDisciplinaLista?tipo=D&codcg=6 e coleta todas as siglas
 *   2. Para cada sigla, acessa obterDisciplina?sgldis=<sigla> e extrai campos
 *   3. Salva data/saude_publica.json
 *
 * Uso:
 *   node scripts/scrape_jupiterweb.js
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE = 'https://uspdigital.usp.br/jupiterweb';
const CODCG = 6; // Faculdade de Saúde Pública
const DATA_DIR = path.join(__dirname, '..', 'data');
const OUT = path.join(DATA_DIR, 'saude_publica.json');
const DELAY_MS = 400; // pausa entre requests para nao sobrecarregar

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function textoBlocoApos(tds, label) {
  const idx = tds.findIndex(t => t.trim() === label);
  if (idx === -1) return '';
  // Pega os tds seguintes até o proximo label conhecido ou fim
  const labels = ['Ementa', 'Objetivos', 'Programa', 'Avaliacao', 'Avaliação', 'Bibliografia'];
  const partes = [];
  for (let i = idx + 1; i < tds.length; i++) {
    if (labels.includes(tds[i].trim())) break;
    if (tds[i].trim()) partes.push(tds[i].trim());
  }
  return partes.join(' ').trim();
}

async function extrairDisciplina(page, sigla) {
  try {
    await page.goto(`${BASE}/obterDisciplina?sgldis=${sigla}`, { waitUntil: 'domcontentloaded', timeout: 20000 });
    await sleep(DELAY_MS);

    const data = await page.evaluate(() => {
      const tds = Array.from(document.querySelectorAll('td')).map(td => td.innerText.trim());

      // Valor logo após um label exato (td[i] = label, td[i+1] = valor)
      const apos = (label) => {
        const idx = tds.findIndex(t => t === label);
        return idx !== -1 ? (tds[idx + 1] || '') : '';
      };

      // Bloco de texto a partir de um label de secao, ate o proximo label de secao
      const secoes = ['Ementa', 'Objetivos', 'Programa', 'Avaliação', 'Avaliacao', 'Bibliografia', 'Critérios de Avaliação'];
      const blocoApos = (label) => {
        const idx = tds.findIndex(t => t === label);
        if (idx === -1) return '';
        const partes = [];
        for (let i = idx + 1; i < tds.length; i++) {
          if (secoes.includes(tds[i])) break;
          // Ignora traducao em ingles (nao há como distinguir de forma confiavel, pega tudo)
          if (tds[i]) partes.push(tds[i]);
        }
        return partes.join(' ').replace(/\s+/g, ' ').trim();
      };

      // Unidade = td[10], Departamento = td[12] (posicoes fixas conforme HTML)
      const unidade = tds[10] || '';
      const departamento = tds[12] || '';

      // Linha "Disciplina: HSA0131 - Nome" = td[14]
      const linhaDisc = tds[14] || '';
      const match = linhaDisc.match(/Disciplina:\s*(\S+)\s*-\s*(.+)/);
      const codigo = match?.[1] || '';
      const nome = match?.[2]?.trim() || '';

      return {
        Codigo: codigo,
        Nome: nome,
        Unidade: unidade,
        Departamento: departamento,
        Creditos_aula: apos('Créditos Aula:'),
        Creditos_trabalho: apos('Créditos Trabalho:'),
        Carga_horaria: apos('Carga Horária Total:'),
        Tipo: apos('Tipo:'),
        Ativacao: apos('Ativação:'),
        Ementa: blocoApos('Ementa'),
        Objetivos: blocoApos('Objetivos'),
        Programa: blocoApos('Programa'),
        Avaliacao: blocoApos('Avaliação') || blocoApos('Avaliacao') || blocoApos('Critérios de Avaliação'),
        Bibliografia: blocoApos('Bibliografia'),
      };
    });

    return data;
  } catch (err) {
    console.warn(`  Erro em ${sigla}: ${err.message}`);
    return null;
  }
}

async function listarSiglas(page) {
  await page.goto(`${BASE}/jupDisciplinaLista?tipo=D&codcg=${CODCG}`, { waitUntil: 'domcontentloaded', timeout: 20000 });
  await sleep(500);

  const siglas = await page.evaluate(() => {
    const rows = Array.from(document.querySelectorAll('table tr'));
    return rows
      .map(r => r.cells[0]?.innerText.trim())
      .filter(s => s && /^[A-Z0-9]{5,10}$/.test(s));
  });

  // Remover duplicatas mantendo ordem
  return [...new Set(siglas)];
}

async function scrape() {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    locale: 'pt-BR',
  });
  const page = await ctx.newPage();

  console.log('Coletando lista de siglas...');
  const siglas = await listarSiglas(page);
  console.log(`${siglas.length} siglas encontradas.`);

  const disciplinas = [];
  for (let i = 0; i < siglas.length; i++) {
    const sigla = siglas[i];
    process.stdout.write(`[${i + 1}/${siglas.length}] ${sigla} ... `);
    const disc = await extrairDisciplina(page, sigla);
    if (disc && disc.Nome) {
      disciplinas.push(disc);
      console.log(`OK: ${disc.Nome.slice(0, 50)}`);
    } else {
      console.log('ignorada (sem nome)');
    }
  }

  await browser.close();

  fs.writeFileSync(OUT, JSON.stringify(disciplinas, null, 2), 'utf-8');
  console.log(`\nSalvo: ${OUT} (${disciplinas.length} disciplinas)`);
}

scrape().catch(err => { console.error(err); process.exit(1); });
