/**
 * Scraper de disciplinas do Janus USP usando Playwright.
 *
 * Fluxo: carrega a pagina JSF (que inicializa o DWR), chama
 * Disciplina.listarDisciplinasOferecidasInglesCSV() via JS dentro
 * do contexto do browser, recebe um path de download temporario e
 * baixa o CSV com os cookies da sessao ativa.
 *
 * Uso:
 *   node scripts/scrape_disciplinas.js
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

const RAW_CSV = path.join(DATA_DIR, 'disciplinas_raw.csv');
const JSON_OUT = path.join(DATA_DIR, 'disciplinas.json');

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuote = false;
  for (const c of line) {
    if (c === '"') { inQuote = !inQuote; }
    else if (c === ';' && !inQuote) { result.push(current); current = ''; }
    else { current += c; }
  }
  result.push(current);
  return result;
}

function parsearCSV(raw) {
  const lines = raw.split('\n').filter(l => l.trim());
  const headers = parseCSVLine(lines[0]).map(h => h.trim());
  const records = [];
  for (const line of lines.slice(1)) {
    if (!line.trim()) continue;
    const vals = parseCSVLine(line);
    const obj = {};
    headers.forEach((h, i) => { if (h) obj[h] = (vals[i] || '').trim(); });
    records.push(obj);
  }
  return records;
}

async function scrape() {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    locale: 'pt-BR',
  });
  const page = await ctx.newPage();

  console.log('Carregando pagina Janus...');
  await page.goto(
    'https://uspdigital.usp.br/janus/componente/disciplinasOferecidasInicial.jsf',
    { waitUntil: 'networkidle', timeout: 30000 }
  );

  console.log('Chamando DWR listarDisciplinasOferecidasInglesCSV...');
  const downloadPath = await page.evaluate(() => {
    return new Promise((resolve, reject) => {
      Disciplina.listarDisciplinasOferecidasInglesCSV({
        callback: (d) => resolve(d),
        errorHandler: (e) => reject(new Error(String(e))),
      });
      setTimeout(() => reject(new Error('timeout DWR')), 20000);
    });
  });

  if (!downloadPath || !downloadPath.startsWith('/janus/dwr/download/')) {
    throw new Error('Resposta inesperada do DWR: ' + downloadPath);
  }

  console.log('Baixando CSV:', downloadPath);
  const csv = await page.evaluate(async (p) => {
    const res = await fetch('https://uspdigital.usp.br' + p, { credentials: 'include' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return res.text();
  }, downloadPath);

  await browser.close();

  fs.writeFileSync(RAW_CSV, csv, 'utf-8');
  console.log(`CSV raw salvo: ${RAW_CSV} (${csv.length} chars)`);

  const records = parsearCSV(csv);
  fs.writeFileSync(JSON_OUT, JSON.stringify(records, null, 2), 'utf-8');
  console.log(`JSON salvo: ${JSON_OUT} (${records.length} disciplinas)`);

  return records;
}

scrape().catch(err => { console.error(err); process.exit(1); });
