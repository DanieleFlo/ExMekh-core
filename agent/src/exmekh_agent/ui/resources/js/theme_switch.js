export default {
    template: `
        <button type="button" class="theme-switch" role="switch" :aria-label="label"
                :aria-checked="light" @click="toggle">
            <svg class="theme-sun" viewBox="0 0 24 24" aria-hidden="true">
                <defs>
                    <linearGradient id="theme-sun-fill" x2="0" y2="1">
                        <stop stop-color="#fff2ab"/>
                        <stop offset="1" stop-color="#f3ae2b"/>
                    </linearGradient>
                </defs>
                <circle cx="12" cy="12" r="4.25"/>
                <path d="M12 2.5v2m0 15v2m-9.5-9.5h2m15 0h2M5.3 5.3l1.4 1.4m10.6 10.6 1.4 1.4M5.3 18.7l1.4-1.4m10.6-10.6 1.4-1.4"/>
            </svg>
            <svg class="theme-moon" viewBox="0 0 24 24" aria-hidden="true">
                <defs>
                    <linearGradient id="theme-moon-fill" x2="0" y2="1">
                        <stop stop-color="#fff"/>
                        <stop offset="1" stop-color="#cbd8ec"/>
                    </linearGradient>
                </defs>
                <path d="M20.6 15.6A8.8 8.8 0 0 1 8.4 3.4a9 9 0 1 0 12.2 12.2Z"/>
            </svg>
        </button>
    `,
    props: ['label'],
    data() {
        return { light: window.exmekhTheme?.mode === 'light' };
    },
    mounted() {
        window.addEventListener('exmekh-theme-change', this.sync);
    },
    beforeUnmount() {
        window.removeEventListener('exmekh-theme-change', this.sync);
    },
    methods: {
        sync() {
            this.light = window.exmekhTheme.mode === 'light';
        },
        toggle() {
            window.exmekhTheme?.set(this.light ? 'dark' : 'light');
        },
    },
};
