/* CAR-bench Challenge - shared behaviours */

// Mobile nav toggle
function toggleNav() {
    document.querySelector('.nav-links')?.classList.toggle('open');
}

function closeRegisterMenus() {
    document.querySelectorAll('.register-menu.open').forEach(menu => {
        menu.classList.remove('open');
        menu.querySelector('.register-toggle')?.setAttribute('aria-expanded', 'false');
    });
}

function toggleRegisterMenu(event) {
    event.preventDefault();
    event.stopPropagation();
    const menu = event.currentTarget.closest('.register-menu');
    if (!menu) return;
    const shouldOpen = !menu.classList.contains('open');
    closeRegisterMenus();
    menu.classList.toggle('open', shouldOpen);
    event.currentTarget.setAttribute('aria-expanded', String(shouldOpen));
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
    initLeaderboardTabs();
});

document.addEventListener('click', event => {
    if (!event.target.closest('.register-menu')) closeRegisterMenus();
});

document.addEventListener('keydown', event => {
    if (event.key === 'Escape') closeRegisterMenus();
});
