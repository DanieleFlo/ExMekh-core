(() => {
    if (window.exmekhChatNavigationInstalled) return;
    window.exmekhChatNavigationInstalled = true;
    window.addEventListener('popstate', () => {
        emitEvent('exmekh_chat_navigate', { path: window.location.pathname });
    });
})();
