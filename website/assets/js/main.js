/* CAR-bench Challenge — shared behaviours */

// Mobile nav toggle
function toggleNav() {
    document.querySelector('.nav-links')?.classList.toggle('open');
}

// Mark active nav link based on current path
(function markActiveNav() {
    const path = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(a => {
        const href = a.getAttribute('href');
        if (!href) return;
        const target = href.split('/').pop();
        if (target === path || (path === '' && target === 'index.html')) {
            a.classList.add('active');
        }
    });
})();

// Countdown to next deadline (Jul 19, 2026 — final evaluation on hidden test set)
function startCountdown(targetISO, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const target = new Date(targetISO).getTime();
    const elD = container.querySelector('[data-u="d"]');
    const elH = container.querySelector('[data-u="h"]');
    const elM = container.querySelector('[data-u="m"]');
    const elS = container.querySelector('[data-u="s"]');

    function tick() {
        const now = Date.now();
        const diff = target - now;
        if (diff <= 0) {
            container.innerHTML = '<div class="countdown-label">Now Live</div><div class="countdown-event">Evaluation in progress</div>';
            return;
        }
        const d = Math.floor(diff / 86400000);
        const h = Math.floor((diff % 86400000) / 3600000);
        const m = Math.floor((diff % 3600000) / 60000);
        const s = Math.floor((diff % 60000) / 1000);
        if (elD) elD.textContent = d;
        if (elH) elH.textContent = String(h).padStart(2, '0');
        if (elM) elM.textContent = String(m).padStart(2, '0');
        if (elS) elS.textContent = String(s).padStart(2, '0');
    }
    tick();
    setInterval(tick, 1000);
}

// Copy BibTeX to clipboard
function copyCitation(btn) {
    const box = btn.closest('.citation-box');
    const text = box?.querySelector('pre')?.textContent || '';
    navigator.clipboard.writeText(text).then(() => {
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = original; }, 1600);
    });
}

// Copy code block
function copyCode(btn) {
    const block = btn.nextElementSibling || btn.closest('.code-block-wrap')?.querySelector('.code-block');
    if (!block) return;
    navigator.clipboard.writeText(block.textContent).then(() => {
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = original; }, 1400);
    });
}

// Leaderboard tabs
function initLeaderboardTabs() {
    document.querySelectorAll('.lb-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.target;
            document.querySelectorAll('.lb-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.lb-panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(target)?.classList.add('active');
        });
    });
}

// Auto-init when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Countdown: next deadline is Jul 1, 2026 AoE (registration closes)
    startCountdown('2026-07-01T23:59:59-12:00', 'countdown');
    initLeaderboardTabs();
});
