// Extra pages: Data Explorer, accuracy / feature-importance charts, history actions, new session.
(function () {
    const CH = { grid: 'rgba(125,150,200,0.15)', tick: '#8fa0c4' };
    let accChart = null, featChart = null, explChart = null;

    function barOptions(horizontal) {
        return {
            indexAxis: horizontal ? 'y' : 'x',
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: CH.tick }, grid: { color: CH.grid } },
                y: { ticks: { color: CH.tick }, grid: { color: CH.grid } }
            }
        };
    }

    async function drawVizCharts() {
        if (!window.Chart || typeof availableModels === 'undefined') return;
        if (accChart) accChart.destroy();
        accChart = new Chart(document.getElementById('vizAccChart'), {
            type: 'bar',
            data: {
                labels: availableModels.map(m => m.name),
                datasets: [
                    { label: 'Accuracy', data: availableModels.map(m => +(m.test_accuracy * 100).toFixed(1)), backgroundColor: '#22d3ee' },
                    { label: 'Weighted F1', data: availableModels.map(m => +(m.test_weighted_f1 * 100).toFixed(1)), backgroundColor: '#8b5cf6' }
                ]
            },
            options: Object.assign(barOptions(true), { plugins: { legend: { labels: { color: CH.tick } } } })
        });
        try {
            const res = await fetch(`${API_BASE}/feature-importance?top=10`);
            const rows = await res.json();
            if (featChart) featChart.destroy();
            featChart = new Chart(document.getElementById('vizFeatChart'), {
                type: 'bar',
                data: {
                    labels: rows.map(r => r.feature),
                    datasets: [{ data: rows.map(r => +(r.importance * 100).toFixed(2)), backgroundColor: '#6366f1' }]
                },
                options: barOptions(true)
            });
        } catch (e) { /* backend without the endpoint: leave chart empty */ }
    }

    // ---- Data Explorer (client-side CSV preview) ----
    function parseCsvLine(line) {
        const out = []; let cur = '', q = false;
        for (let i = 0; i < line.length; i++) {
            const c = line[i];
            if (c === '"') { if (q && line[i + 1] === '"') { cur += '"'; i++; } else q = !q; }
            else if (c === ',' && !q) { out.push(cur); cur = ''; }
            else cur += c;
        }
        out.push(cur);
        return out;
    }

    function esc(v) { return String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

    document.getElementById('explorerFile').addEventListener('change', async e => {
        const file = e.target.files[0];
        if (!file) return;
        const text = await file.text();
        const lines = text.split(/\r?\n/).filter(l => l.length);
        const header = parseCsvLine(lines[0]);
        const rows = lines.slice(1).map(parseCsvLine);
        let missing = 0;
        rows.forEach(r => header.forEach((_, i) => { if (r[i] === undefined || r[i] === '' || r[i] === 'NaN') missing++; }));

        document.getElementById('explorerHead').innerHTML =
            '<tr><th>#</th>' + header.slice(0, 10).map(h => `<th>${esc(h)}</th>`).join('') + '</tr>';
        document.getElementById('explorerBody').innerHTML = rows.slice(0, 5).map((r, i) =>
            `<tr><td>${i + 1}</td>${header.slice(0, 10).map((_, k) => `<td>${esc(r[k] ?? '')}</td>`).join('')}</tr>`).join('');
        document.getElementById('explorerStats').innerHTML =
            `<span class="badge ${missing ? 'warning' : 'safe'}">Missing values: ${missing}</span>
             <span class="badge safe">Rows: ${rows.length.toLocaleString()}</span>
             <span class="badge safe">Columns: ${header.length}</span>
             ${header.length > 10 ? '<small>Showing the first 10 columns.</small>' : ''}`;
        document.getElementById('explorerOut').classList.remove('hidden');
        document.getElementById('dsRows').textContent = rows.length.toLocaleString();

        const ci = header.indexOf('attack_cat');
        const wrap = document.getElementById('explorerChartWrap');
        if (ci >= 0 && window.Chart) {
            const counts = {};
            rows.forEach(r => { const k = r[ci] || 'Unknown'; counts[k] = (counts[k] || 0) + 1; });
            wrap.classList.remove('hidden');
            if (explChart) explChart.destroy();
            explChart = new Chart(document.getElementById('explorerChart'), {
                type: 'bar',
                data: { labels: Object.keys(counts), datasets: [{ data: Object.values(counts),
                    backgroundColor: Object.keys(counts).map(k => k === 'Normal' ? '#34d399' : '#fb7185') }] },
                options: barOptions(false)
            });
        } else wrap.classList.add('hidden');
    });

    // ---- Model history actions ----
    document.getElementById('historyTableBody').addEventListener('click', e => {
        const b = e.target.closest('[data-load-model]');
        if (!b) return;
        const id = b.dataset.loadModel;
        document.getElementById('predictModelSelect').value = id;
        updateCurrentModelStatus(id);
        document.querySelector('.nav-btn[data-page="page-predictions"]').click();
    });
    document.getElementById('btnCompareAll').addEventListener('click', () => {
        document.getElementById('predictModelSelect').value = 'compare_all';
        updateCurrentModelStatus('compare_all');
        document.querySelector('.nav-btn[data-page="page-predictions"]').click();
    });
    document.getElementById('btnClearHist').addEventListener('click', () => {
        document.getElementById('historyTableBody').innerHTML = '';
    });

    // ---- New session: clear inputs and results ----
    document.getElementById('btnNewSession').addEventListener('click', () => {
        ['batchFile', 'explorerFile'].forEach(id => { document.getElementById(id).value = ''; });
        ['singleResultArea', 'batchResults', 'explorerOut', 'singleError', 'batchError'].forEach(id =>
            document.getElementById(id).classList.add('hidden'));
        document.getElementById('jsonInput').value = '';
        if (typeof expectedColumns !== 'undefined') buildForm(expectedColumns);
        document.getElementById('dsRows').textContent = '257,673';
        document.getElementById('predictModelSelect').value = 'compare_all';
        updateCurrentModelStatus('compare_all');
        document.querySelector('.nav-btn[data-page="page-dashboard"]').click();
    });

    window.onAppReady = function () {
        document.getElementById('dsFeatures').textContent = expectedColumns.length;
        drawVizCharts();
        populateHistoryTable();
    };
})();
