(() => {
    const storageKey = 'exmekh-theme';
    const dark = document.getElementById('exmekh-theme-dark');
    const light = document.getElementById('exmekh-theme-light');
    let mode = 'dark';

    function apply(next) {
        mode = next;
        dark.media = mode === 'dark' ? 'all' : 'not all';
        light.media = mode === 'light' ? 'all' : 'not all';
        window.dispatchEvent(new Event('exmekh-theme-change'));
    }

    try {
        if (localStorage.getItem(storageKey) === 'light') apply('light');
    } catch {
        // The switch also works when browser storage is unavailable.
    }

    window.exmekhTheme = {
        get mode() { return mode; },
        set(next) {
            if (next === mode) return;
            apply(next);
            try {
                localStorage.setItem(storageKey, mode);
            } catch {
                // The current page already has the selected stylesheet.
            }
        },
    };
})();
