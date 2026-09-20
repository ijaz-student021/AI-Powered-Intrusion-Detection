let API_BASE = localStorage.getItem('apiBaseUrl') || "http://127.0.0.1:8000/api";
let expectedColumns = [];
let availableModels = [];

document.addEventListener('DOMContentLoaded', initApp);

async function initApp() {
    document.getElementById('apiBaseUrl').value = API_BASE;
    
    // Sidebar Navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
            
            const target = e.currentTarget;
            target.classList.add('active');
            document.getElementById(target.dataset.page).classList.remove('hidden');
        });
    });

    document.querySelectorAll('.btn-nav').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetId = e.currentTarget.dataset.target;
            document.querySelector(`.nav-btn[data-page="${targetId}"]`).click();
        });
    });

    // Form interactions
    document.querySelectorAll('input[name="inputMode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'form') {
                document.getElementById('singleForm').classList.remove('hidden');
                document.getElementById('jsonInputMode').classList.add('hidden');
            } else {
                document.getElementById('singleForm').classList.add('hidden');
                document.getElementById('jsonInputMode').classList.remove('hidden');
                const data = getFormData();
                document.getElementById('jsonInput').value = JSON.stringify(data, null, 2);
            }
        });
    });

    document.getElementById('btnAnalyze').addEventListener('click', analyzeSingle);
    document.getElementById('btnBatch').addEventListener('click', analyzeBatch);
    document.getElementById('btnTestConnection').addEventListener('click', testConnection);
    document.getElementById('btnDownloadCSV').addEventListener('click', downloadCSV);

    document.getElementById('predictModelSelect').addEventListener('change', (e) => {
        updateCurrentModelStatus(e.target.value);
    });

    document.getElementById('vizModelSelect').addEventListener('change', (e) => {
        loadVisualizations(e.target.value);
    });

    await fetchInitialData();
}

async function fetchInitialData() {
    try {
        const modelsRes = await fetch(`${API_BASE}/models`);
        if (!modelsRes.ok) throw new Error('Failed to fetch models');
        availableModels = await modelsRes.json();
        
        populateModelSelectors();
        populateHistoryTable();
        
        document.getElementById('dashModelCount').textContent = availableModels.length;

        const healthRes = await fetch(`${API_BASE}/health`);
        if (!healthRes.ok) throw new Error('Backend health check failed');
        const healthData = await healthRes.json();
        
        if (!healthData.is_loaded) {
            throw new Error('Model artifacts not loaded properly on backend.');
        }
        
        expectedColumns = healthData.feature_columns;
        buildForm(expectedColumns);
        
        document.getElementById('loadingIndicator').classList.add('hidden');
        
        // Load default model
        document.getElementById('predictModelSelect').value = availableModels[0].id;
        updateCurrentModelStatus(availableModels[0].id);
        
        if(availableModels.length > 0) {
            document.getElementById('vizModelSelect').value = availableModels[0].id;
            loadVisualizations(availableModels[0].id);
        }

    } catch (err) {
        document.getElementById('loadingIndicator').classList.add('hidden');
        const errEl = document.getElementById('errorIndicator');
        errEl.textContent = err.message;
        errEl.classList.remove('hidden');
    }
}

function populateModelSelectors() {
    const predictSelect = document.getElementById('predictModelSelect');
    const vizSelect = document.getElementById('vizModelSelect');
    
    // reset select
    predictSelect.innerHTML = '<option value="compare_all">Compare All Models</option>';
    vizSelect.innerHTML = '';
    
    availableModels.forEach(m => {
        predictSelect.innerHTML += `<option value="${m.id}">${m.name}</option>`;
        vizSelect.innerHTML += `<option value="${m.id}">${m.name}</option>`;
    });
}

function populateHistoryTable() {
    const tbody = document.getElementById('historyTableBody');
    tbody.innerHTML = '';
    availableModels.forEach(m => {
        tbody.innerHTML += `
            <tr>
                <td>${m.name}</td>
                <td>${(m.test_accuracy * 100).toFixed(2)}%</td>
                <td>${(m.test_weighted_f1 * 100).toFixed(2)}%</td>
                <td>UNSW-NB15</td>
            </tr>
        `;
    });
}

function updateCurrentModelStatus(modelId) {
    if (modelId === 'compare_all') {
        document.getElementById('statusModelName').textContent = 'Comparing All Models';
        document.getElementById('statusModelAcc').textContent = '--';
        return;
    }
    
    const m = availableModels.find(x => x.id === modelId);
    if (m) {
        document.getElementById('statusModelName').textContent = m.name;
        document.getElementById('statusModelAcc').textContent = `Test Acc: ${(m.test_accuracy * 100).toFixed(1)}%`;
    }
}

function loadVisualizations(modelId) {
    const base = API_BASE.replace('/api', '');
    const prefix = modelId === 'lightgbm' ? 'lgb' : modelId;
    document.getElementById('vizConfVal').src = `${base}/figures/confusion_matrix_val_${prefix}.png`;
    document.getElementById('vizConfTest').src = `${base}/figures/confusion_matrix_test_${prefix}.png`;
    document.getElementById('vizRoc').src = `${base}/figures/roc_auc_test_${prefix}.png`;
}

function buildForm(columns) {
    const form = document.getElementById('singleForm');
    form.innerHTML = '';
    
    columns.forEach(col => {
        const group = document.createElement('div');
        group.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = col;
        
        const input = document.createElement('input');
        input.type = 'text';
        input.name = col;
        input.value = "0"; 
        
        group.appendChild(label);
        group.appendChild(input);
        form.appendChild(group);
    });
}

function getFormData() {
    const formData = new FormData(document.getElementById('singleForm'));
    const data = {};
    for (let [key, value] of formData.entries()) {
        let val = value;
        if (!isNaN(parseFloat(value)) && isFinite(value)) {
            val = parseFloat(value);
        }
        data[key] = val;
    }
    return data;
}

async function analyzeSingle() {
    const errorEl = document.getElementById('singleError');
    errorEl.classList.add('hidden');
    
    let data;
    const mode = document.querySelector('input[name="inputMode"]:checked').value;
    
    if (mode === 'form') {
        data = getFormData();
    } else {
        try {
            data = JSON.parse(document.getElementById('jsonInput').value);
        } catch (e) {
            errorEl.textContent = "Invalid JSON format.";
            errorEl.classList.remove('hidden');
            return;
        }
    }
    
    const model = document.getElementById('predictModelSelect').value;
    
    try {
        const res = await fetch(`${API_BASE}/predict?model=${model}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await res.json();
        if (!res.ok) throw new Error(result.detail || 'Prediction failed');
        
        displaySingleResult(result, model);
    } catch (err) {
        errorEl.textContent = err.message;
        errorEl.classList.remove('hidden');
    }
}

function displaySingleResult(result, modelId) {
    const area = document.getElementById('singleResultArea');
    const content = document.getElementById('singleResultContent');
    area.classList.remove('hidden');
    content.innerHTML = '';
    
    if (modelId === 'compare_all') {
        content.innerHTML += `<div class="mb-3 ${result.agreement ? 'badge safe' : 'badge warning'}">
            ${result.agreement ? '✅ All models agree' : '⚠️ Models disagree'}
        </div>`;
        
        Object.keys(result).forEach(key => {
            if (key === 'agreement') return;
            const res = result[key];
            const m = availableModels.find(x => x.id === key) || {name: key};
            const icon = res.low_confidence_category ? '⚠️' : (res.is_attack ? '🚨' : '✅');
            
            content.innerHTML += `
                <div class="compare-row">
                    <span class="compare-model-name">${m.name}</span>
                    <span class="badge ${res.is_attack ? 'danger' : 'safe'}">${res.predicted_class} ${icon}</span>
                    <span class="text-muted">${(res.confidence * 100).toFixed(1)}%</span>
                </div>
            `;
        });
    } else {
        const icon = result.low_confidence_category ? '⚠️' : (result.is_attack ? '🚨' : '✅');
        content.innerHTML = `
            <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">
                Result: ${result.predicted_class} <span class="badge ${result.is_attack ? 'danger' : 'safe'}">${icon}</span>
                <span style="font-size: 1rem; color: #64748b; font-weight: 400;">(${(result.confidence * 100).toFixed(1)}% confidence)</span>
            </div>
            <h4>Top 3 Predictions:</h4>
            <ul>
                ${result.top_3.map(t => `<li>${t.class_name}: ${(t.probability * 100).toFixed(1)}%</li>`).join('')}
            </ul>
        `;
    }
}

let chartInstance = null;
let currentBatchCsvContent = "";

async function analyzeBatch() {
    const errorEl = document.getElementById('batchError');
    errorEl.classList.add('hidden');
    
    const fileInput = document.getElementById('batchFile');
    if (!fileInput.files.length) {
        errorEl.textContent = "Please select a CSV file first.";
        errorEl.classList.remove('hidden');
        return;
    }
    
    const model = document.getElementById('predictModelSelect').value;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const res = await fetch(`${API_BASE}/predict-batch?model=${model}`, {
            method: 'POST',
            body: formData
        });
        
        const result = await res.json();
        if (!res.ok) throw new Error(result.detail || 'Batch prediction failed');
        
        displayBatchResults(result, model);
    } catch (err) {
        errorEl.textContent = err.message;
        errorEl.classList.remove('hidden');
    }
}

function displayBatchResults(result, modelId) {
    document.getElementById('batchResults').classList.remove('hidden');
    
    const headRow = document.getElementById('batchTableHeadRow');
    const tbody = document.querySelector('#batchTable tbody');
    headRow.innerHTML = '';
    tbody.innerHTML = '';
    
    let csvHeaders = ['Row #'];
    headRow.innerHTML = `<th>Row #</th>`;
    
    if (modelId === 'compare_all') {
        document.getElementById('batchChartContainer').classList.add('hidden');
        
        availableModels.forEach(m => {
            headRow.innerHTML += `<th>${m.name}</th>`;
            csvHeaders.push(m.name);
        });
        headRow.innerHTML += `<th>Agreement</th>`;
        csvHeaders.push('Agreement');
        
        let csvRows = [csvHeaders.join(',')];
        
        // Only display up to 100 rows in the UI to prevent browser freeze
        const displayLimit = 100;
        const displayResults = result.results.slice(0, displayLimit);
        
        // Build CSV string for all rows
        result.results.forEach(row => {
            let csvData = [row.row_index];
            availableModels.forEach(m => {
                const p = row.predictions[m.id];
                csvData.push(`${p.predicted_class} (${(p.confidence*100).toFixed(1)}%)`);
            });
            csvData.push(row.agreement ? 'Yes' : 'No');
            csvRows.push(csvData.join(','));
        });
        
        // Build UI Table for limited rows
        displayResults.forEach(row => {
            let tr = document.createElement('tr');
            let trHtml = `<td>${row.row_index}</td>`;
            
            availableModels.forEach(m => {
                const p = row.predictions[m.id];
                const warn = p.low_confidence_category ? `<span title="Low confidence">⚠️</span>` : '';
                trHtml += `<td>${p.predicted_class} (${(p.confidence*100).toFixed(1)}%) ${warn}</td>`;
            });
            
            const agreeText = row.agreement ? '✅ Yes' : '⚠️ No';
            trHtml += `<td>${agreeText}</td>`;
            
            tr.innerHTML = trHtml;
            tbody.appendChild(tr);
        });
        
        if (result.results.length > displayLimit) {
            let tr = document.createElement('tr');
            tr.innerHTML = `<td colspan="${csvHeaders.length}" style="text-align: center; color: #64748b; font-style: italic;">... Showing first ${displayLimit} of ${result.results.length} rows. Download CSV to see all. ...</td>`;
            tbody.appendChild(tr);
        }
        
        currentBatchCsvContent = csvRows.join('\n');
        
    } else {
        document.getElementById('batchChartContainer').classList.remove('hidden');
        
        headRow.innerHTML += `<th>Prediction</th><th>Confidence</th>`;
        csvHeaders.push('Prediction', 'Confidence');
        let csvRows = [csvHeaders.join(',')];
        
        const displayLimit = 100;
        const displayResults = result.results.slice(0, displayLimit);
        
        // Build CSV string for all rows
        result.results.forEach(row => {
            csvRows.push([row.row_index, row.predicted_class, (row.confidence * 100).toFixed(1) + '%'].join(','));
        });
        
        // Build UI Table for limited rows
        displayResults.forEach(row => {
            let tr = document.createElement('tr');
            const warn = row.low_confidence_category ? `<span title="Low confidence">⚠️</span>` : '';
            tr.innerHTML = `
                <td>${row.row_index}</td>
                <td><span class="badge ${row.is_attack ? 'danger' : 'safe'}">${row.predicted_class}</span> ${warn}</td>
                <td>${(row.confidence * 100).toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        });
        
        if (result.results.length > displayLimit) {
            let tr = document.createElement('tr');
            tr.innerHTML = `<td colspan="3" style="text-align: center; color: #64748b; font-style: italic;">... Showing first ${displayLimit} of ${result.results.length} rows. Download CSV to see all. ...</td>`;
            tbody.appendChild(tr);
        }
        
        currentBatchCsvContent = csvRows.join('\n');
        
        // Chart
        if (chartInstance) chartInstance.destroy();
        const ctx = document.getElementById('batchChart').getContext('2d');
        const labels = Object.keys(result.summary);
        const data = labels.map(l => result.summary[l].count);
        
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Count',
                    data: data,
                    backgroundColor: labels.map(l => l === 'Normal' ? '#10b981' : '#ef4444')
                }]
            },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });
    }
}

function downloadCSV() {
    if (!currentBatchCsvContent) return;
    const blob = new Blob([currentBatchCsvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions_export.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

async function testConnection() {
    const newBase = document.getElementById('apiBaseUrl').value;
    const resEl = document.getElementById('settingsResult');
    try {
        const res = await fetch(`${newBase}/health`);
        if (res.ok) {
            resEl.innerHTML = '<span style="color: #10b981;">✅ Connection successful! Reload to apply.</span>';
            localStorage.setItem('apiBaseUrl', newBase);
            API_BASE = newBase;
        } else {
            throw new Error('Bad response');
        }
    } catch(e) {
        resEl.innerHTML = '<span style="color: #ef4444;">❌ Connection failed. Check URL.</span>';
    }
}
