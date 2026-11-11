(() => {
    if (window.exmekhChatNavigation) return;
    let lastRequest = 0;
    let pendingRequest = null;

    function setBusy(busy) {
        const layout = document.querySelector('.chat-page-layout');
        if (!layout) return;
        const indicator = layout.querySelector('.chat-navigation-overlay');
        if (indicator) indicator.hidden = !busy;
        for (const child of layout.querySelectorAll(':scope > :not(.chat-navigation-overlay)')) {
            child.inert = busy;
        }
    }

    window.exmekhChatNavigation = {
        select(path, history = false) {
            if (history || path !== window.location.pathname || pendingRequest !== null) {
                pendingRequest = ++lastRequest;
                setBusy(true);
                if (!history) {
                    document.querySelector('.q-drawer--mobile .lm-sidebar .sidebar-close-button')?.click();
                }
            }
            emitEvent('exmekh_chat_navigate', {path, history, request_id: pendingRequest});
        },
        finish(requestId) {
            requestAnimationFrame(() => {
                if (pendingRequest !== requestId) return;
                pendingRequest = null;
                setBusy(false);
            });
        },
    };
    window.addEventListener('popstate', () => {
        window.exmekhChatNavigation.select(window.location.pathname, true);
    });
})();
