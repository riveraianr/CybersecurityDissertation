// RA5olver - Automated RA-5 Vulnerability Prioritization with XAI
// Copyright © 2025 Ian Rivera. All rights reserved.

// Sidebar toggle for mobile
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) sidebar.classList.toggle('sidebar-hidden');
}

// Chat visibility toggle
function toggleChat() {
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.classList.toggle('hidden');
}

// Chat expansion toggle
function toggleChatExpand() {
    const chatBox = document.getElementById('chat-box');
    const chatBody = document.getElementById('chat-body');
    if (chatBox && chatBody) {
        chatBox.classList.toggle('chat-box-expanded');
        chatBody.classList.toggle('chat-body-expanded');
    }
}

// Send chat message to server
function sendMessage() {
    const cveId = document.getElementById('chat-cve-select')?.value;
    const message = document.getElementById('chat-input')?.value;
    const chatBody = document.getElementById('chat-body');

    if (!message || !cveId || !chatBody) {
        console.warn('Chat input missing:', { cveId, message, chatBody });
        return;
    }

    const userMessage = document.createElement('div');
    userMessage.className = 'chat-message chat-message-user';
    userMessage.textContent = message;
    chatBody.appendChild(userMessage);

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `message=${encodeURIComponent(message)}&cve_id=${encodeURIComponent(cveId)}`
    })
    .then(response => {
        if (!response.ok) throw new Error(`Chat request failed: ${response.status}`);
        return response.json();
    })
    .then(data => {
        const botMessage = document.createElement('div');
        botMessage.className = 'chat-message chat-message-bot';
        botMessage.textContent = data.response;
        chatBody.appendChild(botMessage);
        chatBody.scrollTop = chatBody.scrollHeight;
    })
    .catch(error => console.error('Chat error:', error));

    document.getElementById('chat-input').value = '';
}

// Filter vulnerability table rows
function filterVulnerabilities() {
    const severityFilter = document.getElementById('severity-filter')?.value.toLowerCase() || 'all';
    const priorityFilter = document.getElementById('priority-filter')?.value.toLowerCase() || 'all';
    const exploitFilter = document.getElementById('exploit-filter')?.value.toLowerCase() || 'all';
    const searchQuery = document.getElementById('search-query')?.value.toLowerCase() || '';

    const rows = document.getElementsByClassName('vuln-row');
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        const severity = row.getAttribute('data-severity')?.toLowerCase() || '';
        const priority = row.getAttribute('data-priority')?.toLowerCase() || '';
        const hasExploit = row.getAttribute('data-exploit')?.toLowerCase() || '';
        const cveId = row.getAttribute('data-cve-id')?.toLowerCase() || '';
        const analysis = row.getAttribute('data-analysis')?.toLowerCase() || '';
        const conclusion = row.getAttribute('data-conclusion')?.toLowerCase() || '';

        let show = true;
        if (severityFilter !== 'all' && severity !== severityFilter) show = false;
        if (priorityFilter !== 'all' && priority !== priorityFilter) show = false;
        if (exploitFilter !== 'all' && hasExploit !== exploitFilter) show = false;
        if (searchQuery && !(cveId.includes(searchQuery) || analysis.includes(searchQuery) || conclusion.includes(searchQuery))) show = false;

        row.style.display = show ? '' : 'none';
    }
}

// Toggle between light and dark themes
function toggleTheme() {
    const html = document.documentElement;
    if (html.classList.contains('dark')) {
        html.classList.remove('dark');
        localStorage.setItem('theme', 'light');
    } else {
        html.classList.add('dark');
        localStorage.setItem('theme', 'dark');
    }
}

// Open vulnerability modal with charts
function openModal(cveId) {
    const modal = document.getElementById('vuln-modal');
    if (!modal) return;

    modal.style.display = 'flex';
    modal.setAttribute('aria-hidden', 'false');

    const vuln = vulnerabilities.find(v => v.CVE_ID === cveId);
    if (!vuln) {
        alert('Vulnerability not found.');
        return;
    }

    const vulnData = {
        cve_id: vuln.CVE_ID || 'N/A',
        reported_severity: vuln.Reported_Severity || 'Unknown',
        priority: vuln.Priority || 'Unknown',
        has_exploit: vuln.Has_Exploit ? 'Has' : 'No',
        analysis: vuln.Analysis || 'Not specified',
        priority_justification: vuln.Priority_Justification || 'Not specified',
        severity_rationale: vuln.Severity_Rationale || 'Not specified',
        rf_feature_importance: vuln.RF_Feature_Importance || {},
        shap_values: vuln.SHAP_Values || {},
        suggested_remediation_steps: vuln.Suggested_Remediation_Steps || 'Not specified'
    };

    document.getElementById('modal-cve-id').textContent = vulnData.cve_id;
    document.getElementById('modal-severity').textContent = `${vulnData.reported_severity} Severity`;
    document.getElementById('modal-severity').className = `inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold badge-${vulnData.reported_severity.toLowerCase()}`;
    document.getElementById('modal-priority').textContent = `Priority: ${vulnData.priority}`;
    document.getElementById('modal-priority').className = `inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold badge-priority-${vulnData.priority.toLowerCase()}`;
    document.getElementById('modal-exploit').textContent = `${vulnData.has_exploit} Exploit`;
    document.getElementById('modal-exploit').className = `inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold badge-exploit-${vulnData.has_exploit.toLowerCase()}`;
    document.getElementById('modal-analysis').textContent = vulnData.analysis;
    document.getElementById('modal-priority-justification').textContent = vulnData.priority_justification;
    document.getElementById('modal-severity-rationale').textContent = vulnData.severity_rationale;
    document.getElementById('modal-remediation').textContent = vulnData.suggested_remediation_steps;

    if (window.featureChart) window.featureChart.destroy();
    if (window.shapChart) window.shapChart.destroy();

    const featureLabels = [];
    const featureValues = [];
    for (const [key, value] of Object.entries(vulnData.rf_feature_importance)) {
        featureLabels.push(key);
        const match = value.match(/(\d+\.\d+)/);
        featureValues.push(match ? parseFloat(match[0]) : 0);
    }
    const featureImportanceCtx = document.getElementById('modal-feature-importance-chart')?.getContext('2d');
    if (featureImportanceCtx) {
        window.featureChart = new Chart(featureImportanceCtx, {
            type: 'bar',
            data: {
                labels: featureLabels,
                datasets: [{
                    label: 'Feature Importance (%)',
                    data: featureValues,
                    backgroundColor: '#3b82f6',
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                scales: { x: { max: 100, beginAtZero: true }, y: { beginAtZero: true } },
                plugins: { legend: { display: false } }
            }
        });
    }

    const shapLabels = [];
    const shapValues = [];
    for (const [key, value] of Object.entries(vulnData.shap_values)) {
        shapLabels.push(key);
        const match = value.match(/([-]?\d+\.\d+)/);
        shapValues.push(match ? parseFloat(match[0]) : 0);
    }
    const shapValuesCtx = document.getElementById('modal-shap-values-chart')?.getContext('2d');
    if (shapValuesCtx) {
        window.shapChart = new Chart(shapValuesCtx, {
            type: 'pie',
            data: {
                labels: shapLabels,
                datasets: [{
                    data: shapValues.map(v => Math.abs(v)),
                    backgroundColor: ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'],
                    borderWidth: 1
                }]
            },
            options: {
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = shapValues[context.dataIndex];
                                return `${context.label}: ${(value * 100).toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    const shapList = document.getElementById('shap-values-list');
    if (shapList) {
        shapList.innerHTML = '';
        shapLabels.forEach((label, index) => {
            const value = shapValues[index];
            const div = document.createElement('div');
            div.className = 'flex items-center justify-between text-sm';
            div.innerHTML = `
                <div class="flex items-center gap-2">
                    <div class="w-3 h-3 rounded-full" style="background-color: ${['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'][index % 5]}"></div>
                    <span>${label}</span>
                </div>
                <span class="${value >= 0 ? 'text-green-500' : 'text-red-500'}">
                    ${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%
                </span>
            `;
            shapList.appendChild(div);
        });
    }
}

// Close vulnerability modal
function closeModal() {
    const modal = document.getElementById('vuln-modal');
    if (modal) {
        modal.style.display = 'none';
        modal.setAttribute('aria-hidden', 'true');
    }
}

// Switch modal tabs
function switchTab(tabId) {
    const tabs = document.querySelectorAll('.tab-button');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
    });
    contents.forEach(c => c.classList.remove('active'));

    const activeTab = document.querySelector(`.tab-button[data-tab="${tabId}"]`);
    const activeContent = document.getElementById(tabId);
    if (activeTab && activeContent) {
        activeTab.classList.add('active');
        activeTab.setAttribute('aria-selected', 'true');
        activeContent.classList.add('active');
    }
}

// Populate vulnerability table dynamically
function populateVulnTable() {
    const tbody = document.querySelector('#vuln-table-body');
    if (!tbody) {
        console.error('Table body not found');
        return;
    }
    if (!Array.isArray(vulnerabilities) || vulnerabilities.length === 0) {
        console.warn('No vulnerabilities data to populate table');
        tbody.innerHTML = '<tr><td colspan="6" class="px-4 py-2 text-center text-gray-400">No vulnerabilities available</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    vulnerabilities.forEach(vuln => {
        const row = document.createElement('tr');
        row.className = 'vuln-row hover:bg-gray-600 cursor-pointer';
        row.setAttribute('data-severity', vuln.Reported_Severity?.toLowerCase() || '');
        row.setAttribute('data-priority', vuln.Priority?.toLowerCase() || '');
        row.setAttribute('data-exploit', vuln.Has_Exploit ? 'true' : 'false');
        row.setAttribute('data-cve-id', vuln.CVE_ID?.toLowerCase() || '');
        row.setAttribute('data-analysis', vuln.Analysis?.toLowerCase() || '');
        row.setAttribute('data-conclusion', vuln.Conclusion?.toLowerCase() || '');
        row.onclick = () => openModal(vuln.CVE_ID);
        row.innerHTML = `
            <td class="px-4 py-2 font-medium text-gray-100">${vuln.CVE_ID || 'N/A'}</td>
            <td class="px-4 py-2">
                <span class="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold badge-${vuln.Reported_Severity?.toLowerCase() || 'unknown'}">
                    ${vuln.Reported_Severity || 'Unknown'}
                </span>
            </td>
            <td class="px-4 py-2">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold badge-priority-${vuln.Priority?.toLowerCase() || 'unknown'}">
                    ${vuln.Priority || 'Unknown'}
                </span>
            </td>
            <td class="px-4 py-2 hidden md:table-cell max-w-[200px] truncate text-gray-300">${vuln.Analysis || 'Not specified'}</td>
            <td class="px-4 py-2 hidden lg:table-cell max-w-[200px] truncate text-gray-300">${vuln.Suggested_Remediation_Steps || 'Not specified'}</td>
            <td class="px-4 py-2">
                <button onclick="event.stopPropagation(); openModal('${vuln.CVE_ID}')" class="text-blue-400 hover:text-blue-300">
                    <svg class="h-4 w-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                    </svg>
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Update sidebar counts
function updateSidebarCounts() {
    if (!Array.isArray(vulnerabilities)) return;
    document.getElementById('critical-count').textContent = vulnerabilities.filter(v => v.Reported_Severity === 'Critical').length;
    document.getElementById('high-count').textContent = vulnerabilities.filter(v => v.Reported_Severity === 'High').length;
    document.getElementById('medium-count').textContent = vulnerabilities.filter(v => v.Reported_Severity === 'Medium').length;
    document.getElementById('low-count').textContent = vulnerabilities.filter(v => v.Reported_Severity === 'Low').length;
}

// Update summary cards
function updateSummaryCards() {
    if (!Array.isArray(vulnerabilities) || vulnerabilities.length === 0) return;
    const criticalCount = vulnerabilities.filter(v => v.Reported_Severity === 'Critical').length;
    document.getElementById('critical-vulns-count').textContent = criticalCount;
    document.getElementById('critical-vulns-delta').textContent = `+${criticalCount - 20}`; // Placeholder delta
    const highPriorityCount = vulnerabilities.filter(v => v.Priority === 'Yes').length;
    document.getElementById('high-priority-count').textContent = highPriorityCount;
    document.getElementById('high-priority-attention').textContent = highPriorityCount;
    document.getElementById('active-exploits-count').textContent = vulnerabilities.filter(v => v.Has_Exploit).length;
    document.getElementById('active-exploits-mitigations').textContent = vulnerabilities.filter(v => v.Has_Exploit && v.Suggested_Remediation_Steps !== 'Not specified').length;
    const remediated = vulnerabilities.filter(v => v.Suggested_Remediation_Steps !== 'Not specified').length;
    document.getElementById('remediation-progress').textContent = `${Math.floor((remediated / vulnerabilities.length) * 100)}%`;
}

// Populate chat CVE dropdown
function populateChatCVE() {
    const select = document.getElementById('chat-cve-select');
    if (!select || !Array.isArray(vulnerabilities)) return;
    select.innerHTML = '<option value="">Select a Vulnerability</option>';
    vulnerabilities.forEach(vuln => {
        const option = document.createElement('option');
        option.value = vuln.CVE_ID;
        option.textContent = vuln.CVE_ID;
        select.appendChild(option);
    });
}

// Initialize after DOM load
document.addEventListener('DOMContentLoaded', () => {
    populateVulnTable();
    updateSidebarCounts();
    updateSummaryCards();
    populateChatCVE();
});